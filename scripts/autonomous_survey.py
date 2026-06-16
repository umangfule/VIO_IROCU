#!/usr/bin/env python3
"""
ASCEND autonomous survey mission (IRoC-U 2026) — GPS-denied, VINS-Mono VIO.

Mission state machine (one start command, then fully autonomous):

    CONNECT  -> wait for FCU + a valid local-position estimate (from VIO/EKF)
    GUIDED   -> set GUIDED mode, arm
    TAKEOFF  -> vertical climb to survey altitude (NO setpoint streaming yet)
    HOVER    -> hold over home for 5 s (settle VIO)
    SURVEY   -> slow lawnmower (boustrophedon) across the WHOLE arena, following a
                rate-limited "carrot" setpoint -> gentle, controlled speed
    RTL      -> return to home (0,0) at survey altitude
    LAND     -> descend and land on the base station (home = VIO origin)
    DONE

Design notes (fixes from field testing)
---------------------------------------
* TAKEOFF: we do NOT stream position setpoints until the drone is airborne.
  Streaming setpoint_position/local while still on the ground puts ArduCopter in
  a guided-position sub-mode that ignores the takeoff climb, so the drone never
  lifted off autonomously. We arm in GUIDED, send the takeoff command (with
  retries), wait for the climb, THEN enable streaming for the survey.
* SPEED: the survey setpoint is a "carrot" advanced at `survey_speed` m/s and
  kept within `lookahead` of the drone, so the drone moves slowly and smoothly
  instead of darting to far waypoints.
* GEOFENCE: every commanded setpoint is clamped to the arena rectangle
  (fence_x/fence_y half-extents), so the drone cannot be commanded outside the
  arena. Breaches of the actual pose are logged.
* COVERAGE: the arena is centered on the home/base-station origin, so a SYMMETRIC
  lawnmower (x in [-x,+x], y in [-y,+y]) sweeps the whole arena regardless of the
  exact ENU<->Gazebo axis convention.

Frames: setpoint_position/local and local_position/pose are local ENU with the
origin at the EKF home (= base station = arena centre = VIO origin).
"""
import math
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AutonomousSurvey:
    def __init__(self):
        rospy.init_node("autonomous_survey")

        # ---- mission params ----
        self.survey_alt = rospy.get_param("~survey_alt", 2.5)     # m AGL (rulebook 2-6 m)
        self.hover_secs = rospy.get_param("~hover_secs", 5.0)
        self.survey_speed = rospy.get_param("~survey_speed", 0.5)  # m/s, slow
        self.climb_speed = rospy.get_param("~climb_speed", 0.4)    # m/s descent rate
        self.lookahead = rospy.get_param("~lookahead", 0.8)        # m carrot lead

        # Survey rectangle (ENU, symmetric about home). Long passes along Y.
        self.x_min = rospy.get_param("~x_min", -3.0)
        self.x_max = rospy.get_param("~x_max", 3.0)
        self.y_min = rospy.get_param("~y_min", -4.5)
        self.y_max = rospy.get_param("~y_max", 4.5)
        self.lane_spacing = rospy.get_param("~lane_spacing", 1.2)

        # Geofence half-extents (arena bounds, ENU). Setpoints clamped to these.
        self.fence_x = rospy.get_param("~fence_x", 3.6)
        self.fence_y = rospy.get_param("~fence_y", 5.1)

        self.wp_tol = rospy.get_param("~wp_tol", 0.3)             # m "reached" radius
        self.land_xy_tol = rospy.get_param("~land_xy_tol", 0.2)

        # ---- state ----
        self.state = State()
        self.pose = PoseStamped()
        self.have_pose = False
        self.streaming = False             # gate: stream setpoints only when True
        self.sp = PoseStamped()
        self.sp.header.frame_id = "map"
        self._last_fence_warn = rospy.Time(0)

        # ---- I/O ----
        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        self.sp_pub = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=10)

        rospy.loginfo("[survey] waiting for MAVROS services...")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/takeoff")
        rospy.wait_for_service("/mavros/cmd/land")
        self.srv_arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.srv_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.srv_takeoff = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)
        self.srv_land = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)

        self.rate = rospy.Rate(20.0)
        self.dt = 0.05
        rospy.Timer(rospy.Duration(self.dt), self._stream_cb)

    # ---------- callbacks ----------
    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg
        self.have_pose = True
        self._fence_check()

    def _stream_cb(self, _evt):
        if not self.streaming:
            return
        self.sp.header.stamp = rospy.Time.now()
        self.sp_pub.publish(self.sp)

    # ---------- geometry helpers ----------
    @property
    def px(self):
        return self.pose.pose.position.x

    @property
    def py(self):
        return self.pose.pose.position.y

    @property
    def pz(self):
        return self.pose.pose.position.z

    def _set_sp(self, x, y, z):
        # geofence clamp on every write
        self.sp.pose.position.x = clamp(x, -self.fence_x, self.fence_x)
        self.sp.pose.position.y = clamp(y, -self.fence_y, self.fence_y)
        self.sp.pose.position.z = z

    def _fence_check(self):
        if (abs(self.px) > self.fence_x + 0.5 or abs(self.py) > self.fence_y + 0.5):
            if (rospy.Time.now() - self._last_fence_warn).to_sec() > 2.0:
                rospy.logwarn("[survey] GEOFENCE breach: pose=(%.1f,%.1f) — pulling back",
                              self.px, self.py)
                self._last_fence_warn = rospy.Time.now()

    def _dist_xyz(self, x, y, z):
        return math.sqrt((self.px - x) ** 2 + (self.py - y) ** 2 + (self.pz - z) ** 2)

    def _sleep(self, secs):
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < secs:
            self.rate.sleep()

    # ---------- carrot-following motion ----------
    def _goto(self, tx, ty, tz, timeout=90.0, tol=None, speed=None):
        """Advance a rate-limited carrot setpoint toward (tx,ty,tz).
        Keeps the carrot within `lookahead` of the drone -> slow, smooth tracking."""
        tol = self.wp_tol if tol is None else tol
        speed = self.survey_speed if speed is None else speed
        tx = clamp(tx, -self.fence_x, self.fence_x)
        ty = clamp(ty, -self.fence_y, self.fence_y)
        rospy.loginfo("[survey] -> waypoint E=%.2f N=%.2f U=%.2f", tx, ty, tz)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            cx = self.sp.pose.position.x
            cy = self.sp.pose.position.y
            cz = self.sp.pose.position.z
            # only advance the carrot if the drone has kept up
            if math.hypot(cx - self.px, cy - self.py) < self.lookahead:
                dx, dy = tx - cx, ty - cy
                d = math.hypot(dx, dy)
                step = speed * self.dt
                if d <= step:
                    cx, cy = tx, ty
                elif d > 1e-6:
                    cx += dx / d * step
                    cy += dy / d * step
            # vertical carrot
            dz = tz - cz
            vstep = self.climb_speed * self.dt
            cz = tz if abs(dz) <= vstep else cz + math.copysign(vstep, dz)
            self._set_sp(cx, cy, cz)

            if self._dist_xyz(tx, ty, tz) < tol:
                return True
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn("[survey] waypoint timeout (d=%.2f m) — moving on",
                              self._dist_xyz(tx, ty, tz))
                return False
            self.rate.sleep()
        return False

    def _lawnmower(self):
        """Boustrophedon: long passes along Y (North), stepping lanes in X (East)."""
        wps = []
        n = max(1, int(round((self.x_max - self.x_min) / self.lane_spacing)) + 1)
        forward = True
        for i in range(n):
            x = min(self.x_min + i * self.lane_spacing, self.x_max)
            ys = [self.y_min, self.y_max] if forward else [self.y_max, self.y_min]
            for y in ys:
                wps.append((x, y))
            forward = not forward
        return wps

    # ---------- mission ----------
    def run(self):
        rospy.loginfo("[survey] waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.state.connected:
            self.rate.sleep()
        rospy.loginfo("[survey] FCU connected. Waiting for local-position estimate (VIO)...")
        while not rospy.is_shutdown() and not self.have_pose:
            self.rate.sleep()
        rospy.loginfo("[survey] position estimate OK (pose=%.2f,%.2f,%.2f).",
                      self.px, self.py, self.pz)

        # 1) GUIDED + ARM (no streaming yet)
        self._ensure_mode("GUIDED")
        self._ensure_armed(True)

        # 2) TAKEOFF — command-based vertical climb, streaming OFF
        if not self._takeoff():
            rospy.logerr("[survey] takeoff failed; aborting mission.")
            return

        # Now airborne over home. Seed the carrot at current pose and start streaming.
        self._set_sp(self.px, self.py, self.survey_alt)
        self.streaming = True
        rospy.loginfo("[survey] streaming enabled; holding over home.")

        # 3) HOVER
        rospy.loginfo("[survey] hovering %.0f s...", self.hover_secs)
        self._goto(0.0, 0.0, self.survey_alt, timeout=20.0, tol=0.4)
        self._sleep(self.hover_secs)

        # 4) SURVEY — slow lawnmower over the whole arena
        wps = self._lawnmower()
        rospy.loginfo("[survey] starting lawnmower: %d waypoints @ %.1f m/s.",
                      len(wps), self.survey_speed)
        for (x, y) in wps:
            if rospy.is_shutdown():
                break
            self._goto(x, y, self.survey_alt)
        rospy.loginfo("[survey] survey complete.")

        # 5) RTL
        rospy.loginfo("[survey] returning to home (0,0)...")
        self._goto(0.0, 0.0, self.survey_alt, timeout=120.0, tol=self.land_xy_tol)

        # 6) LAND on base station (home)
        rospy.loginfo("[survey] descending for precision landing on base station...")
        # Bring it low over the pad under guided control first...
        self._goto(0.0, 0.0, max(0.8, self.survey_alt * 0.35), timeout=40.0, tol=0.25)
        # ...then STOP streaming setpoints and hand fully to the autopilot LAND mode.
        # Streaming (0,0,0) during LAND made the node fight the descent and left the
        # estimate climbing after touchdown. LAND needs no setpoints.
        self.streaming = False
        self._ensure_mode("LAND")
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            if not self.state.armed:
                break
            if (rospy.Time.now() - t0).to_sec() > 45.0:
                rospy.logwarn("[survey] land timeout; force-disarming.")
                self._ensure_armed(False)
                break
            self.rate.sleep()
        rospy.loginfo("[survey] landed and disarmed. Mission DONE.")
        rospy.loginfo("[survey] (note: mono-VIO may drift slightly while static on the "
                      "ground — no zero-velocity update. Use reset_drone.py to recentre.)")

    # ---------- takeoff with retries ----------
    def _takeoff(self):
        target = self.survey_alt
        for attempt in range(1, 6):
            rospy.loginfo("[survey] takeoff attempt %d to %.1f m...", attempt, target)
            # keep GUIDED + armed in case they were lost
            self._ensure_mode("GUIDED")
            self._ensure_armed(True)
            try:
                resp = self.srv_takeoff(min_pitch=0, yaw=0, latitude=0,
                                        longitude=0, altitude=target)
                rospy.loginfo("[survey] takeoff cmd sent (success=%s)", resp.success)
            except rospy.ServiceException as e:
                rospy.logwarn("[survey] takeoff service error: %s", e)
            # wait up to 8 s for the climb to begin/reach
            t0 = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < 8.0:
                if self.pz > target - 0.3:
                    rospy.loginfo("[survey] reached altitude (z=%.2f).", self.pz)
                    return True
                self.rate.sleep()
            if self.pz > 0.5:   # climbing — give it more time
                rospy.loginfo("[survey] climbing (z=%.2f), waiting...", self.pz)
                t1 = rospy.Time.now()
                while not rospy.is_shutdown() and (rospy.Time.now() - t1).to_sec() < 12.0:
                    if self.pz > target - 0.3:
                        rospy.loginfo("[survey] reached altitude (z=%.2f).", self.pz)
                        return True
                    self.rate.sleep()
                if self.pz > target - 0.5:
                    return True
        return self.pz > 0.8

    # ---------- arming / mode with retries ----------
    def _ensure_mode(self, mode):
        if self.state.mode == mode:
            return
        rospy.loginfo("[survey] setting mode %s...", mode)
        while not rospy.is_shutdown() and self.state.mode != mode:
            try:
                self.srv_mode(base_mode=0, custom_mode=mode)
            except rospy.ServiceException as e:
                rospy.logwarn("[survey] set_mode error: %s", e)
            self._sleep(0.5)
        rospy.loginfo("[survey] mode = %s", mode)

    def _ensure_armed(self, arm):
        if self.state.armed == arm:
            return
        rospy.loginfo("[survey] %s...", "arming" if arm else "disarming")
        while not rospy.is_shutdown() and self.state.armed != arm:
            try:
                self.srv_arm(arm)
            except rospy.ServiceException as e:
                rospy.logwarn("[survey] arm error: %s", e)
            self._sleep(0.5)
        rospy.loginfo("[survey] armed = %s", self.state.armed)


if __name__ == "__main__":
    try:
        AutonomousSurvey().run()
    except rospy.ROSInterruptException:
        pass
