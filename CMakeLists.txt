cmake_minimum_required(VERSION 3.0.2)
project(ascend_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  mavros_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/autonomous_survey.py
  scripts/reset_drone.py
  scripts/aruco_localizer.py
  scripts/aruco_vio_survey.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
