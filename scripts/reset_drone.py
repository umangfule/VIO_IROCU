#!/usr/bin/env python3
"""
Reset the ASCEND drone for "lost drone" / diverged-VIO recovery.

What it does (run it any time):
  1. Teleports the Gazebo model back to the home pad (origin) with zero velocity
     (via /gazebo/set_model_state).
  2. Restarts the VINS-Mono estimator (publishes to /feature_tracker/restart),
     which clears the trajectory/feature cloud in RViz and re-initialises VIO.

Usage:
    rosrun ascend_navigation reset_drone.py
    rosrun ascend_navigation reset_drone.py _x:=0 _y:=0 _z:=0.2 _model:=iris_demo

Notes:
  * After a reset the autopilot's EKF may need a moment to re-converge on the
    teleported pose. If SITL state is badly diverged, also disarm and `reboot`
    in the MAVProxy console, then restart the mission.
  * This is a test/recovery tool — not part of the autonomous mission.
"""
import rospy
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


def main():
    rospy.init_node("reset_drone")
    model = rospy.get_param("~model", "iris_demo")
    x = rospy.get_param("~x", 0.0)
    y = rospy.get_param("~y", 0.0)
    z = rospy.get_param("~z", 0.2)

    # 1) Restart VINS first so the estimator is ready for the new pose.
    restart_pub = rospy.Publisher("/feature_tracker/restart", Bool, queue_size=1, latch=True)
    rospy.sleep(0.5)
    restart_pub.publish(Bool(data=True))
    rospy.loginfo("[reset] sent VINS restart (/feature_tracker/restart).")

    # 2) Teleport the Gazebo model home with zero velocity.
    rospy.loginfo("[reset] waiting for /gazebo/set_model_state...")
    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    ms = ModelState()
    ms.model_name = model
    ms.pose.position.x = x
    ms.pose.position.y = y
    ms.pose.position.z = z
    ms.pose.orientation.w = 1.0           # level, yaw 0
    # zero twist
    ms.twist.linear.x = ms.twist.linear.y = ms.twist.linear.z = 0.0
    ms.twist.angular.x = ms.twist.angular.y = ms.twist.angular.z = 0.0
    ms.reference_frame = "world"

    try:
        resp = set_state(ms)
        if resp.success:
            rospy.loginfo("[reset] teleported '%s' to home (%.2f, %.2f, %.2f).",
                          model, x, y, z)
        else:
            rospy.logwarn("[reset] set_model_state failed: %s", resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("[reset] set_model_state error: %s", e)

    rospy.loginfo("[reset] done. If the autopilot stays diverged, disarm + reboot SITL.")


if __name__ == "__main__":
    main()
