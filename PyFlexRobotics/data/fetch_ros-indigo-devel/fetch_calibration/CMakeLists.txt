cmake_minimum_required(VERSION 2.8.3)
project(fetch_calibration)

find_package(catkin)
catkin_python_setup()
catkin_package()

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  PROGRAMS scripts/camera_reconfigure.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
