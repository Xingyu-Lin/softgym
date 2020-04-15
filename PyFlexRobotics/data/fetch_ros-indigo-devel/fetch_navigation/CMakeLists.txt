cmake_minimum_required(VERSION 2.8.3)
project(fetch_navigation)

find_package(catkin REQUIRED)
catkin_package()

install(
  DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  PROGRAMS scripts/tilt_head.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS roslaunch)
  roslaunch_add_file_check(launch/fetch_nav.launch)
  roslaunch_add_file_check(launch/freight_nav.launch)
endif()
