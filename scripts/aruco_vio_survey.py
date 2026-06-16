#!/usr/bin/env python3
"""
ASCEND GPS-denied survey with an AprilTag/ArUco-bootstrapped takeoff (IRoC-U 2026).

SEPARATE, fully GPS-denied mission (the GPS-bootstrapped version lives in
autonomous_survey.py and is left untouched). Flow:

    ARUCO BOOTSTRAP -> the down-camera sees the dense ArUco pad; aruco_localizer
                       publishes the drone's pad-relative pose. THIS node relays it
                       to /mavros/vision_pose/pose so the EKF has an absolute
                       position WITHOUT GPS -> we can arm.
    TAKEOFF         -> stable vertical climb, holding over the pad (ArUco position).
    HOVER 5 s       -> let VINS-Mono VIO initialise in the air.
    HANDOVER        -> once VIO is healthy, switch the vision_pose source from ArUco
                       to VINS (offset-aligned so the estimate is continuous).
    SURVEY          -> slow lawnmower over the whole arena, on VIO only.
    RTL + LAND      -> return and land on the base station.

Genuinely GPS-denied: GPS is never used. The fiducial pad provides the bootstrap
position (replacing GPS) until VIO takes over for the free-flight survey.

Run with sitl_vins_nogps.parm (EKF3 ExternalNav primary, GPS off). This node is the
SOLE publisher of /mavros/vision_pose/pose, so launch VINS WITHOUT vins_to_mavros
(bringup_aruco.launch). The pad->ENU axis mapping may need tuning on first run
(aruco_localizer east_from/north_from) — see issue #18.
"""
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from geographic_msgs.msg import GeoPointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandLong, SetMode, CommandTOL, ParamGet

# VINS world (z-up, gravity-aligned) -> ENU : 90 deg about X (same as vins_to_mavros)
_R_VINS_ENU = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ArucoVioSurvey:
    def __init__(self):
        rospy.init_node("aruco_vio_survey")

        self.survey_alt = rospy.get_param("~survey_alt", 2.5)
        self.hover_secs = rospy.get_param("~hover_secs", 5.0)
        self.survey_speed = rospy.get_param("~survey_speed", 0.5)
        self.climb_speed = rospy.get_param("~climb_speed", 0.4)
        self.lookahead = rospy.get_param("~lookahead", 0.8)
        self.x_min = rospy.get_param("~x_min", -3.0)
        self.x_max = rospy.get_param("~x_max", 3.0)
        self.y_min = rospy.get_param("~y_min", -4.5)
        self.y_max = rospy.get_param("~y_max", 4.5)
        self.lane_spacing = rospy.get_param("~lane_spacing", 1.2)
        self.fence_x = rospy.get_param("~fence_x", 3.6)
        self.fence_y = rospy.get_param("~fence_y", 5.1)
        self.wp_tol = rospy.get_param("~wp_tol", 0.3)
        self.land_xy_tol = rospy.get_param("~land_xy_tol", 0.2)
        # Hard-fence abort: if the EKF position ever runs this far past the soft fence
        # the position loop has diverged (e.g. a vision-frame/axis mismatch) — LAND in
        # place rather than let the vehicle fly away. Pure safety; never hit in a
        # healthy survey, so it does not affect normal behaviour.
        self.abort_x = rospy.get_param("~abort_x", self.fence_x + 1.5)
        self.abort_y = rospy.get_param("~abort_y", self.fence_y + 1.5)
        self.aborted = False
        self.vins_settle = rospy.get_param("~vins_settle", 4.0)
        self.vision_rate = rospy.get_param("~vision_rate", 30.0)
        # VIO-init excitation: a monocular VIO never initialises from a static hover
        # ("IMU excitation not enough / not enough parallax"). We fly a small box,
        # still on ArUco vision_pose, at a slightly lower altitude where the pad fills
        # the down-camera (so the ArUco EKF feed stays solid through init).
        self.excite_alt = rospy.get_param("~excite_alt", 1.5)
        self.excite_radius = rospy.get_param("~excite_radius", 0.7)
        self.excite_speed = rospy.get_param("~excite_speed", 0.35)
        self.excite_cycles = rospy.get_param("~excite_cycles", 3)

        self.state = State()
        self.pose = PoseStamped()
        self.have_pose = False
        self.lp_recv = rospy.Time(0)   # wall-time of last local_position msg (liveness)
        self.lp_count = 0
        self.streaming = False
        self.sp = PoseStamped(); self.sp.header.frame_id = "map"
        self._last_fence_warn = rospy.Time(0)

        self.vision_source = "aruco"
        self.aruco = None
        self.aruco_visible = False
        self.aruco_last = rospy.Time(0)
        self.aruco_stamp = rospy.Time(0)
        # NOTE: No pad→earth rotation is needed here. The ArUco board is placed at the
        # world origin with no yaw (board +x = world East, board +y = world North), so
        # aruco_localizer with east_from="+x"/north_from="+y" already outputs correct
        # ENU. EK3_SRC1_YAW=1 (compass) is also earth ENU → no frame mismatch.
        self.vins_origin = None
        self.vins_enu = None
        self.vins_last = rospy.Time(0)
        self.vins_stamp = rospy.Time(0)
        self.vins_offset = np.zeros(3)
        # VIO->ENU similarity alignment solved at handover. VINS-Mono fixes only its
        # up-axis (gravity); the horizontal yaw is arbitrary and monocular scale can be
        # off, so a translation-only handover lets "go East" map to a rotated/scaled
        # direction -> the drone chases a target it never reaches and flies out of the
        # arena. vio_R = scale*R(2x2), vio_t = translation, vio_zoff = altitude offset.
        self.vio_aligned = False
        self.vio_R = np.eye(2)
        self.vio_t = np.zeros(2)
        self.vio_zoff = 0.0
        self._align_pairs = []

        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/aruco/pose", PoseStamped, self._aruco_cb)
        rospy.Subscriber("/aruco/visible", Bool, self._vis_cb)
        rospy.Subscriber("/vins_estimator/odometry", Odometry, self._vins_cb)
        self.sp_pub = rospy.Publisher("/mavros/setpoint_position/local",
                                      PoseStamped, queue_size=10)
        self.vision_pub = rospy.Publisher("/mavros/vision_pose/pose",
                                          PoseStamped, queue_size=10)
        # GPS-denied bootstrap: with no GPS the EKF never gets a global origin, so it
        # never publishes a LOCAL position and we'd hang forever waiting for it. We
        # set the origin ourselves so ExternalNav (ArUco/VIO) defines the local frame.
        self.origin_pub = rospy.Publisher("/mavros/global_position/set_gp_origin",
                                           GeoPointStamped, queue_size=1, latch=True)
        # ArduPilot SITL default home (CMAC). Any valid point works — we fly in the
        # local ENU frame; the origin only anchors that frame so local_position flows.
        self.origin_lat = rospy.get_param("~origin_lat", -35.3632621)
        self.origin_lon = rospy.get_param("~origin_lon", 149.1652374)
        self.origin_alt = rospy.get_param("~origin_alt", 584.0)

        rospy.loginfo("[aruco-survey] waiting for MAVROS services...")
        for s in ("/mavros/cmd/arming", "/mavros/set_mode", "/mavros/cmd/takeoff",
                  "/mavros/cmd/command"):
            rospy.wait_for_service(s)
        self.srv_arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.srv_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.srv_takeoff = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)
        self.srv_command = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        # ParamGet is optional (only for the pre-flight sanity check); don't hard-block
        # the mission on it in case the param plugin is slow/absent.
        try:
            rospy.wait_for_service("/mavros/param/get", timeout=5.0)
            self.srv_param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        except rospy.ROSException:
            self.srv_param_get = None

        self.rate = rospy.Rate(20.0)
        self.dt = 0.05
        rospy.Timer(rospy.Duration(self.dt), self._stream_cb)
        rospy.Timer(rospy.Duration(1.0 / self.vision_rate), self._vision_cb)

    # ---------- callbacks ----------
    def _state_cb(self, m): self.state = m

    def _pose_cb(self, m):
        self.pose = m; self.have_pose = True
        self.lp_recv = rospy.Time.now(); self.lp_count += 1
        self._fence_check()

    def _lp_live(self):
        # A LIVE EKF estimate keeps publishing local_position at the EKF rate even when
        # static. A single frozen sample (EKF not actually navigating) goes stale fast.
        return (rospy.Time.now() - self.lp_recv).to_sec() < 0.3

    def _aruco_cb(self, m):
        self.aruco = m; self.aruco_last = rospy.Time.now()
        self.aruco_stamp = m.header.stamp

    def _vis_cb(self, m): self.aruco_visible = m.data

    def _vins_cb(self, m):
        p = m.pose.pose.position
        v = np.array([p.x, p.y, p.z])
        if self.vins_origin is None:
            self.vins_origin = v.copy()
        self.vins_enu = _R_VINS_ENU @ (v - self.vins_origin)
        self.vins_last = rospy.Time.now()
        self.vins_stamp = m.header.stamp

    # ---------- vision_pose relay (sole publisher) ----------
    def _vision_cb(self, _evt):
        pose, stamp = None, rospy.Time(0)
        if self.vision_source == "aruco":
            if self.aruco is not None and (rospy.Time.now()-self.aruco_last).to_sec() < 0.5:
                a = self.aruco.pose
                pose = PoseStamped().pose
                pose.position.x = float(a.position.x)
                pose.position.y = float(a.position.y)
                pose.position.z = float(a.position.z)
                pose.orientation.w = 1.0
                stamp = self.aruco_stamp
        elif self.vins_enu is not None:
            if self.vio_aligned:
                xy = self.vio_R @ self.vins_enu[:2] + self.vio_t
                e = (float(xy[0]), float(xy[1]), float(self.vins_enu[2] + self.vio_zoff))
            else:                                # fallback: translation-only continuity
                v = self.vins_enu + self.vins_offset
                e = (float(v[0]), float(v[1]), float(v[2]))
            pose = PoseStamped().pose
            pose.position.x, pose.position.y, pose.position.z = e
            pose.orientation.w = 1.0
            stamp = self.vins_stamp
        if pose is not None:
            # Stamp with now(), NOT the source image/odom stamp. The camera runs at a
            # bursty ~11 Hz while this relay runs at vision_rate=30 Hz, so reusing the
            # source stamp republished the SAME (and sometimes stale) header.stamp 2-3x.
            # ArduPilot's EKF3 buffers ExternalNav by timestamp; non-advancing or too-old
            # stamps make it treat every frame as a fresh acquisition and reset its extnav
            # fusion -> "EKF3 IMU0 is using external nav / stopped aiding" cycling ->
            # position_ok() flickers false -> GUIDED arm rejected (result=4). In SITL the
            # camera->relay latency is ~0 and the FCU shares sim time, so now() is both
            # accurate and strictly increasing, giving the EKF a dense, gap-free, dup-free
            # 30 Hz stream. (On real hardware with real latency, switch to a properly
            # delay-compensated measurement stamp + set the vision delay param.)
            out = PoseStamped()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = "map"
            out.pose = pose
            self.vision_pub.publish(out)

    def _stream_cb(self, _evt):
        if not self.streaming:
            return
        self.sp.header.stamp = rospy.Time.now()
        self.sp_pub.publish(self.sp)

    # ---------- helpers ----------
    @property
    def px(self): return self.pose.pose.position.x
    @property
    def py(self): return self.pose.pose.position.y
    @property
    def pz(self): return self.pose.pose.position.z

    def _set_sp(self, x, y, z):
        self.sp.pose.position.x = clamp(x, -self.fence_x, self.fence_x)
        self.sp.pose.position.y = clamp(y, -self.fence_y, self.fence_y)
        self.sp.pose.position.z = z

    def _fence_check(self):
        if not self.aborted and (abs(self.px) > self.abort_x or abs(self.py) > self.abort_y):
            self._abort("hard-fence breach pose=(%.1f,%.1f)" % (self.px, self.py))
            return
        if abs(self.px) > self.fence_x + 0.5 or abs(self.py) > self.fence_y + 0.5:
            if (rospy.Time.now()-self._last_fence_warn).to_sec() > 2.0:
                rospy.logwarn("[aruco-survey] GEOFENCE breach pose=(%.1f,%.1f)",
                              self.px, self.py)
                self._last_fence_warn = rospy.Time.now()

    def _abort(self, reason):
        self.aborted = True
        self.streaming = False
        rospy.logerr("[aruco-survey] ABORT: %s — LANDing. The horizontal loop diverged: "
                     "check the vision frame matches the EKF heading (east_from/north_from "
                     "in bringup_aruco.launch vs EK3_SRC1_YAW).", reason)
        try:
            self.srv_mode(base_mode=0, custom_mode="LAND")
        except rospy.ServiceException as e:
            rospy.logwarn("[aruco-survey] abort set_mode: %s", e)

    def _dist(self, x, y, z):
        return math.sqrt((self.px-x)**2 + (self.py-y)**2 + (self.pz-z)**2)

    def _sleep(self, secs):
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now()-t0).to_sec() < secs:
            self.rate.sleep()

    def _vins_healthy(self):
        return (self.vins_enu is not None and
                (rospy.Time.now()-self.vins_last).to_sec() < 0.5)

    def _goto(self, tx, ty, tz, timeout=90.0, tol=None, speed=None):
        tol = self.wp_tol if tol is None else tol
        speed = self.survey_speed if speed is None else speed
        tx = clamp(tx, -self.fence_x, self.fence_x)
        ty = clamp(ty, -self.fence_y, self.fence_y)
        rospy.loginfo("[aruco-survey] -> E=%.2f N=%.2f U=%.2f", tx, ty, tz)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and not self.aborted:
            cx, cy, cz = (self.sp.pose.position.x, self.sp.pose.position.y,
                          self.sp.pose.position.z)
            if math.hypot(cx-self.px, cy-self.py) < self.lookahead:
                dx, dy = tx-cx, ty-cy
                d = math.hypot(dx, dy); step = speed*self.dt
                if d <= step:
                    cx, cy = tx, ty
                elif d > 1e-6:
                    cx += dx/d*step; cy += dy/d*step
            dz = tz-cz; vstep = self.climb_speed*self.dt
            cz = tz if abs(dz) <= vstep else cz + math.copysign(vstep, dz)
            self._set_sp(cx, cy, cz)
            if self._dist(tx, ty, tz) < tol:
                return True
            if (rospy.Time.now()-t0).to_sec() > timeout:
                rospy.logwarn("[aruco-survey] wp timeout d=%.2f", self._dist(tx, ty, tz))
                return False
            self.rate.sleep()
        return False

    def _lawnmower(self):
        wps = []
        n = max(1, int(round((self.x_max-self.x_min)/self.lane_spacing))+1)
        fwd = True
        for i in range(n):
            x = min(self.x_min + i*self.lane_spacing, self.x_max)
            ys = [self.y_min, self.y_max] if fwd else [self.y_max, self.y_min]
            for y in ys:
                wps.append((x, y))
            fwd = not fwd
        return wps

    def _ensure_mode(self, mode):
        if self.state.mode == mode:
            return
        rospy.loginfo("[aruco-survey] mode -> %s", mode)
        while not rospy.is_shutdown() and self.state.mode != mode:
            try: self.srv_mode(base_mode=0, custom_mode=mode)
            except rospy.ServiceException as e: rospy.logwarn("set_mode: %s", e)
            self._sleep(0.5)

    def _preflight_param_check(self):
        """Read the EKF/vision params straight off the FCU and fail FAST with a clear
        message if external-nav is not actually configured. This is the #1 cause of the
        endless GUIDED arm result=4: EK3_ENABLE/POSXY/VISO_TYPE come up at 0 because the
        .parm was never applied to this SITL boot (use sim_vehicle --add-param-file=...,
        or run set_vio_params.py). GUIDED does NOT need GPS; it needs a live ExternalNav
        position, which these params enable. Returns True if config looks sane (or if we
        cannot read params and must trust the user)."""
        if self.srv_param_get is None:
            rospy.logwarn("[aruco-survey] /mavros/param/get unavailable; skipping the EKF "
                          "param sanity check (cannot verify ExternalNav is configured).")
            return True
        want = {"EK3_ENABLE": 1, "EK3_SRC1_POSXY": 6, "VISO_TYPE": 1, "GPS_TYPE": 0}
        bad = {}
        for name, exp in want.items():
            try:
                r = self.srv_param_get(param_id=name)
                got = r.value.real if r.value.real != 0.0 else float(r.value.integer)
                if abs(got - exp) > 1e-3:
                    bad[name] = (got, exp)
            except rospy.ServiceException as e:
                rospy.logwarn("[aruco-survey] could not read %s (%s)", name, e)
        if bad:
            for name, (got, exp) in bad.items():
                rospy.logerr("[aruco-survey]   %s = %s  (need %s)", name, got, exp)
            rospy.logerr("[aruco-survey] EKF/ExternalNav NOT configured on the FCU -> the "
                         "EKF will never navigate on vision and the GUIDED arm will be "
                         "rejected (result=4). This is NOT a GPS/GUIDED limitation. FIX: "
                         "start SITL with --add-param-file=%s OR run `python3 "
                         "set_vio_params.py`, then relaunch.",
                         "/home/umang/catkin_ws/sitl_vins_nogps.parm")
            return False
        rospy.loginfo("[aruco-survey] preflight OK: EK3_ENABLE=1 POSXY=6(ExternalNav) "
                      "VISO_TYPE=1 GPS_TYPE=0 -> GPS-denied vision nav is configured.")
        return True

    def _set_ekf_origin(self):
        msg = GeoPointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.position.latitude = self.origin_lat
        msg.position.longitude = self.origin_lon
        msg.position.altitude = self.origin_alt
        self.origin_pub.publish(msg)

    def _ensure_armed(self, arm):
        if self.state.armed == arm:
            return
        rospy.loginfo("[aruco-survey] %s", "arming" if arm else "disarming")
        while not rospy.is_shutdown() and self.state.armed != arm:
            try:
                if arm:
                    # MAV_CMD_COMPONENT_ARM_DISARM (400) with param2=21196 is the
                    # force-arm magic number — equivalent to MAVProxy "arm throttle force".
                    # Needed because GUIDED mode's internal arm check calls position_ok()
                    # outside the ARMING_CHECK bitmask, so ARMING_CHECK=0 alone does not
                    # bypass it. Our own safety gates (GUIDED mode, EKF position acquired,
                    # fresh ArUco vision) run before we reach here.
                    resp = self.srv_command(command=400, param1=1.0, param2=21196.0)
                else:
                    resp = self.srv_arm(False)
                if not resp.success:
                    rospy.logwarn_throttle(3.0,
                        "[aruco-survey] FCU rejected %s (result=%s) — mode=%s "
                        "vision=%s ekf=(%.2f,%.2f,%.2f).",
                        "arm" if arm else "disarm",
                        getattr(resp, "result", "?"), self.state.mode,
                        "FRESH" if self._vision_fresh() else "STALE",
                        self.px, self.py, self.pz)
            except rospy.ServiceException as e:
                rospy.logwarn("arm/disarm: %s", e)
            self._sleep(0.5)
        rospy.loginfo("[aruco-survey] armed=%s mode=%s", self.state.armed, self.state.mode)

    # ---------- mission ----------
    def run(self):
        rospy.loginfo("[aruco-survey] waiting for FCU...")
        while not rospy.is_shutdown() and not self.state.connected:
            self.rate.sleep()

        rospy.loginfo("[aruco-survey] waiting to see the ArUco pad...")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and not self.aruco_visible:
            if (rospy.Time.now()-t0).to_sec() > 30:
                rospy.logwarn("[aruco-survey] pad not seen — check camera/markers/EKF.")
                t0 = rospy.Time.now()
            self.rate.sleep()
        rospy.loginfo("[aruco-survey] pad visible; relaying ArUco -> vision_pose.")

        # Fail FAST if the FCU isn't actually set up for GPS-denied vision nav, instead
        # of marching all the way to a doomed arm attempt. (GUIDED needs a live position,
        # not GPS -- this check verifies the ExternalNav source is enabled.)
        if not self._preflight_param_check():
            rospy.logerr("[aruco-survey] aborting before arm: fix the FCU params and relaunch.")
            return

        # Anchor the EKF local frame (no GPS => no automatic origin). Without this the
        # EKF never produces a local position and the wait below never finishes.
        rospy.loginfo("[aruco-survey] setting EKF origin (%.5f, %.5f) for GPS-denied bootstrap.",
                      self.origin_lat, self.origin_lon)
        for _ in range(5):
            self._set_ekf_origin(); self._sleep(0.3)

        # Wait for a LIVE EKF local position -- i.e. /mavros/local_position/pose updating
        # CONTINUOUSLY for 2 s, not a single frozen sample. The old check tripped on the
        # first message and printed "acquired", but when the EKF isn't really navigating
        # (ExternalNav off) that topic publishes once at z~0.11 and then stops -> the
        # mission armed against a dead estimate and got result=4 forever. Requiring a
        # sustained stream means "acquired" is only logged when the EKF truly navigates.
        rospy.loginfo("[aruco-survey] waiting for a LIVE EKF local position "
                      "(/mavros/local_position/pose updating, not frozen)...")
        t0 = rospy.Time.now(); last = rospy.Time.now(); live_since = None
        while not rospy.is_shutdown():
            if self._lp_live():
                live_since = live_since or rospy.Time.now()
                if (rospy.Time.now()-live_since).to_sec() >= 2.0:
                    break
            else:
                live_since = None
            if (rospy.Time.now()-last).to_sec() > 5.0:
                self._set_ekf_origin()      # re-assert origin in case it was missed
                rospy.logwarn("[aruco-survey] no LIVE EKF position after %.0fs (got %d msg(s), "
                              "last %.1fs ago). The EKF is NOT navigating on vision. Check: "
                              "params applied (EK3_ENABLE=1, EK3_SRC1_POSXY=6, VISO_TYPE=1) "
                              "via set_vio_params.py / --add-param-file, and "
                              "/mavros/vision_pose/pose flowing at ~30 Hz.",
                              (rospy.Time.now()-t0).to_sec(), self.lp_count,
                              (rospy.Time.now()-self.lp_recv).to_sec())
                last = rospy.Time.now()
            self.rate.sleep()
        rospy.loginfo("[aruco-survey] LIVE EKF local position: (%.2f, %.2f, %.2f) @ ~%d msgs.",
                      self.px, self.py, self.pz, self.lp_count)
        self._sleep(2.0)   # brief settle before arming

        self._ensure_mode("GUIDED")
        self._ensure_armed(True)

        if not self._takeoff():
            rospy.logerr("[aruco-survey] takeoff failed; aborting.")
            return
        self._set_sp(self.px, self.py, self.survey_alt)
        self.streaming = True

        # Centre over the pad, drop to the excitation altitude, then EXCITE so VINS-Mono
        # can initialise. The previous static hover here is what crashed the drone: a
        # monocular VIO never inits without motion, so the mission leaned on the ArUco
        # feed at altitude until the EKF variance failsafe tripped (-> LAND).
        self._goto(0.0, 0.0, self.excite_alt, timeout=25.0, tol=0.3)
        if self.aborted:
            return
        vio_ready = self._excite_vio()

        if vio_ready and self._handover_to_vio():
            rospy.loginfo("[aruco-survey] now on VIO (GPS-denied, pad out of view OK).")
        else:
            rospy.logwarn("[aruco-survey] VIO not ready — staying on ArUco (limited range).")

        # Climb to the survey altitude on whichever source is now active.
        self._goto(0.0, 0.0, self.survey_alt, timeout=25.0, tol=0.4)
        if self.aborted:
            return

        wps = self._lawnmower()
        rospy.loginfo("[aruco-survey] lawnmower: %d wps @ %.1f m/s", len(wps), self.survey_speed)
        for (x, y) in wps:
            if rospy.is_shutdown() or self.aborted:
                break
            self._goto(x, y, self.survey_alt)
        if self.aborted:
            return
        rospy.loginfo("[aruco-survey] survey complete.")

        rospy.loginfo("[aruco-survey] returning home...")
        self._goto(0.0, 0.0, self.survey_alt, timeout=120.0, tol=self.land_xy_tol)
        rospy.loginfo("[aruco-survey] descending for precision landing...")
        self._goto(0.0, 0.0, max(0.8, self.survey_alt*0.35), timeout=40.0, tol=0.25)
        self.streaming = False
        self._ensure_mode("LAND")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if not self.state.armed:
                break
            if (rospy.Time.now()-t0).to_sec() > 45:
                self._ensure_armed(False); break
            self.rate.sleep()
        rospy.loginfo("[aruco-survey] landed. Mission DONE (fully GPS-denied).")

    def _excite_vio(self):
        """Fly a small box (still on ArUco vision_pose, inside the fence) to give
        VINS-Mono the translation + IMU excitation it needs to initialise. VINS-Mono
        only publishes /vins_estimator/odometry once its solver reaches NON_LINEAR,
        i.e. after a successful metric init — so the first fresh odometry IS the
        ready signal (and the first thing that draws a trajectory in rviz)."""
        r = self.excite_radius
        box = [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]
        rospy.loginfo("[aruco-survey] exciting VINS-Mono (box r=%.1f m @ %.1f m) — a "
                      "monocular VIO needs motion, not a static hover, to initialise.",
                      r, self.excite_alt)
        moves = 0
        healthy = False
        for _ in range(self.excite_cycles):
            for (x, y) in box:
                if rospy.is_shutdown():
                    return False
                self._goto(x, y, self.excite_alt, timeout=20.0, tol=0.25,
                           speed=self.excite_speed)
                moves += 1
                if self._vins_healthy():
                    healthy = True
                    break
            if healthy:
                break
        if not healthy:
            rospy.logwarn("[aruco-survey] VINS-Mono still not publishing odometry after "
                          "excitation — check feature_tracker (texture/show_track) and "
                          "the /imu0 + /iris_demo/cam0/image_raw rates.")
            return False
        rospy.loginfo("[aruco-survey] VINS-Mono odometry live (VIO initialised) after "
                      "%d excitation move(s); running ENU alignment lap...", moves)
        # Alignment lap: still on ArUco (so self.px,py are TRUE ENU), settle at each
        # corner + centre and record (VINS_xy -> EKF_ENU_xy) pairs. These let the
        # handover solve the VIO->ENU yaw + scale, not just a translation.
        self._align_pairs = []
        for (x, y) in box + [(0.0, 0.0)]:
            if rospy.is_shutdown():
                break
            self._goto(x, y, self.excite_alt, timeout=20.0, tol=0.2,
                       speed=self.excite_speed)
            self._sleep(0.6)                     # let VINS + EKF settle, stay in sync
            if self._vins_healthy() and self._vision_fresh():
                self._align_pairs.append((self.vins_enu.copy(),
                                          np.array([self.px, self.py, self.pz])))
        # VINS health gates the handover; the pairs (if any) let it solve yaw+scale,
        # otherwise _handover_to_vio falls back to translation-only continuity.
        if len(self._align_pairs) < 4:
            rospy.logwarn("[aruco-survey] only %d alignment pair(s) captured (wanted "
                          ">=4) — ArUco may be dropping out at the box corners.",
                          len(self._align_pairs))
        return self._vins_healthy()

    def _fit_vio_alignment(self, pairs):
        """Least-squares 2-D similarity (rotation + uniform scale + translation) that
        maps VINS horizontal position onto true ENU, from the (VINS_xy, EKF_ENU_xy)
        pairs gathered on ArUco. Umeyama/Kabsch with scale. Z is gravity-aligned and
        metric, so it only needs an offset. Returns True on a well-conditioned fit."""
        src = np.array([p[0][:2] for p in pairs])      # VINS xy
        dst = np.array([p[1][:2] for p in pairs])      # EKF ENU xy
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        var_s = (sc ** 2).sum() / len(src)
        if var_s < 0.02:                               # corners too close -> ill-posed
            return False
        H = (dc.T @ sc) / len(src)
        U, D, Vt = np.linalg.svd(H)
        W = np.eye(2)
        if np.linalg.det(U @ Vt) < 0:
            W[1, 1] = -1.0
        R = U @ W @ Vt
        s = float((D * np.diag(W)).sum() / var_s)
        if not (0.3 < s < 3.0):                        # implausible scale -> reject
            return False
        t = mu_d - s * (R @ mu_s)
        res = dst - (src @ (s * R).T + t)
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        if rms > 0.5:                                  # poor fit -> don't trust it
            return False
        self.vio_R = s * R
        self.vio_t = t
        self.vio_zoff = float(np.mean([p[1][2] - p[0][2] for p in pairs]))
        self.vio_aligned = True
        rospy.loginfo("[aruco-survey] VIO->ENU aligned: yaw=%.1f deg scale=%.3f "
                      "rms=%.3f m (%d pairs).", math.degrees(math.atan2(R[1, 0],
                      R[0, 0])), s, rms, len(pairs))
        return True

    def _handover_to_vio(self):
        rospy.loginfo("[aruco-survey] waiting for VINS-Mono VIO to be steady...")
        t0 = rospy.Time.now(); steady = None
        while not rospy.is_shutdown():
            if self._vins_healthy():
                steady = steady or rospy.Time.now()
                if (rospy.Time.now()-steady).to_sec() >= self.vins_settle:
                    break
            else:
                steady = None
            if (rospy.Time.now()-t0).to_sec() > 30:
                return False
            self.rate.sleep()
        # Prefer the full similarity (yaw+scale+translation) from the alignment lap;
        # fall back to legacy translation-only continuity if it is unavailable/ill-posed.
        if not (len(self._align_pairs) >= 4 and self._fit_vio_alignment(self._align_pairs)):
            ref = self.aruco
            if ref is not None and self.vins_enu is not None:
                self.vins_offset = np.array([ref.pose.position.x, ref.pose.position.y,
                                             ref.pose.position.z]) - self.vins_enu
            rospy.logwarn("[aruco-survey] VIO alignment ill-posed — using translation-"
                          "only handover (heading/scale NOT corrected; watch for drift).")
        self.vision_source = "vio"
        self._sleep(2.0)
        return True

    def _vision_fresh(self):
        return (self.aruco is not None and
                (rospy.Time.now()-self.aruco_last).to_sec() < 0.5)

    def _takeoff(self):
        for attempt in range(1, 6):
            self._ensure_mode("GUIDED"); self._ensure_armed(True)
            # Report the inputs the FCU needs before each attempt: GUIDED takeoff is
            # silently REJECTED unless EKF position is healthy, which here means a
            # fresh ArUco vision_pose. Surface that so a rejection is diagnosable.
            rospy.loginfo("[aruco-survey] takeoff attempt %d -> %.1f m | mode=%s armed=%s "
                          "vision=%s ekf=(%.2f,%.2f,%.2f)", attempt, self.survey_alt,
                          self.state.mode, self.state.armed,
                          "FRESH" if self._vision_fresh() else "STALE",
                          self.px, self.py, self.pz)
            if not self._vision_fresh():
                rospy.logwarn("[aruco-survey] no fresh ArUco pose — EKF position is "
                              "likely unhealthy; pad must stay in view to take off.")
            ok = False
            try:
                resp = self.srv_takeoff(min_pitch=0, yaw=0, latitude=0, longitude=0,
                                        altitude=self.survey_alt)
                ok = bool(getattr(resp, "success", False))
                if not ok:
                    rospy.logwarn("[aruco-survey] takeoff REJECTED by FCU (result=%s). "
                                  "Usually EKF position not ready or not in GUIDED.",
                                  getattr(resp, "result", "?"))
            except rospy.ServiceException as e:
                rospy.logwarn("[aruco-survey] takeoff svc error: %s", e)
            # Give an accepted takeoff time to climb; keep watching even if the ack
            # was late so we don't restart a climb that is already underway.
            t0 = rospy.Time.now()
            window = 14.0 if ok else 8.0
            while not rospy.is_shutdown() and (rospy.Time.now()-t0).to_sec() < window:
                if self.pz > self.survey_alt - 0.3:
                    rospy.loginfo("[aruco-survey] reached alt z=%.2f", self.pz)
                    return True
                if self.pz > 0.5:               # climbing — extend the watch window
                    window = max(window, (rospy.Time.now()-t0).to_sec() + 12.0)
                self.rate.sleep()
        if self.pz > 0.8:
            return True
        rospy.logerr("[aruco-survey] takeoff did not climb after 5 attempts (z=%.2f). "
                     "Check: pad in camera view, /aruco/visible True, sitl_vins_nogps.parm "
                     "loaded (EK3_SRC1_POSXY=6), MAVROS vision_pose flowing.", self.pz)
        return False


if __name__ == "__main__":
    try:
        ArucoVioSurvey().run()
    except rospy.ROSInterruptException:
        pass
