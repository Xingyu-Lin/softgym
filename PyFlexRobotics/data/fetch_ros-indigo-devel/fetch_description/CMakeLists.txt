cmake_minimum_required(VERSION 2.8.3)
project(fetch_description)

find_package(catkin)
catkin_package()

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
