#!/usr/bin/env python3
"""
ArUco landing-pad localizer for the GPS-denied, fiducial-bootstrapped takeoff.

Detects the dense ArUco board on the base station (base_station/gen_aruco_pad.py)
in the down-camera image and publishes the drone's pose over the pad. This is an
ABSOLUTE position source while the drone is on / near the pad — enough to arm and
take off GPS-denied, before VINS-Mono VIO has initialised.

Publishes:
    /aruco/pose     geometry_msgs/PoseStamped   drone position in pad ENU (frame "map")
    /aruco/visible  std_msgs/Bool               true when >=1 board marker is seen

The board's marker corners are defined in the PAD frame (from aruco_layout.yaml).
cv2.aruco.estimatePoseBoard returns the board pose in the camera frame; inverting
it gives the camera (≈drone) position in the pad frame. ArUco reports each marker's
corners in its canonical orientation, so the result is a fixed pad frame regardless
of camera yaw.

Frame mapping (pad -> MAVROS local ENU) is CONFIGURABLE (east_from / north_from,
each one of +x,-x,+y,-y) because the pad frame may be rotated/mirrored vs the
autopilot's local ENU. Defaults follow the standard ArduPilot gazebo-iris mapping
(East = -pad_y, North = +pad_x). Flip if the ArUco takeoff drifts (issue #18).
"""
import os
import yaml
import numpy as np
import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

_AXIS = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}


class ArucoLocalizer:
    def __init__(self):
        rospy.init_node("aruco_localizer")
        layout_path = rospy.get_param(
            "~layout", os.path.expanduser(
                "~/catkin_ws/src/ascend_navigation/config/aruco_layout.yaml"))
        self.image_topic = rospy.get_param("~image_topic", "/iris_demo/cam0/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic",
                                          "/iris_demo/cam0/camera_info")
        self.east_from = rospy.get_param("~east_from", "-y")
        self.north_from = rospy.get_param("~north_from", "+x")

        with open(layout_path) as f:
            layout = yaml.safe_load(f)
        self.dict = aruco.Dictionary_get(getattr(aruco, layout["dictionary"]))
        self.det_params = aruco.DetectorParameters_create()
        self.board = self._build_board(layout["markers"])

        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.rvec = np.zeros((3, 1))
        self.tvec = np.zeros((3, 1))

        self.pub_pose = rospy.Publisher("/aruco/pose", PoseStamped, queue_size=10)
        self.pub_vis = rospy.Publisher("/aruco/visible", Bool, queue_size=10)
        rospy.Subscriber(self.info_topic, CameraInfo, self._info_cb)
        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1,
                         buff_size=2 ** 24)
        rospy.loginfo("[aruco] localizer up — board=%d markers, image=%s",
                      len(layout["markers"]), self.image_topic)

    def _build_board(self, markers):
        obj, ids = [], []
        for mid, m in markers.items():
            cx, cy, s = m["x"], m["y"], m["size"]
            h = s / 2.0
            # ArUco canonical corner order: TL, TR, BR, BL (marker +Y up, +X right)
            obj.append(np.array([[cx - h, cy + h, 0.0],
                                 [cx + h, cy + h, 0.0],
                                 [cx + h, cy - h, 0.0],
                                 [cx - h, cy - h, 0.0]], dtype=np.float32))
            ids.append(int(mid))
        return aruco.Board_create(obj, self.dict, np.array(ids, dtype=np.int32))

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.D, dtype=np.float64)
            rospy.loginfo("[aruco] camera intrinsics received.")

    def _enu(self, pad_xyz):
        ai, asign = _AXIS[self.east_from]
        bi, bsign = _AXIS[self.north_from]
        return asign * pad_xyz[ai], bsign * pad_xyz[bi], pad_xyz[2]

    def _image_cb(self, msg):
        if self.K is None:
            return
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception as e:                      # noqa: BLE001
            rospy.logwarn_throttle(5.0, "[aruco] cv_bridge: %s" % e)
            return
        corners, ids, _ = aruco.detectMarkers(gray, self.dict, parameters=self.det_params)
        if ids is None or len(ids) == 0:
            self.pub_vis.publish(Bool(data=False))
            return
        n, self.rvec, self.tvec = aruco.estimatePoseBoard(
            corners, ids, self.board, self.K, self.D, self.rvec, self.tvec)
        if n <= 0:
            self.pub_vis.publish(Bool(data=False))
            return
        R, _ = cv2.Rodrigues(self.rvec)
        C = (-R.T @ self.tvec).flatten()            # camera position in pad frame
        east, north, up = self._enu(C)

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "map"
        out.pose.position.x = float(east)
        out.pose.position.y = float(north)
        out.pose.position.z = float(abs(up))        # altitude above pad
        out.pose.orientation.w = 1.0                # yaw from compass (EKF), not vision
        self.pub_pose.publish(out)
        self.pub_vis.publish(Bool(data=True))


if __name__ == "__main__":
    try:
        ArucoLocalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
