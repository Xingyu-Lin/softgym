cmake_minimum_required(VERSION 2.8.3)
project(fetch_depth_layer)

find_package(catkin REQUIRED 
  COMPONENTS
    pluginlib
    roscpp
    image_transport
    cv_bridge
    sensor_msgs
    opencv_candidate
    costmap_2d
    nav_msgs
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(fetch_depth_layer src/depth_layer.cpp)

target_link_libraries(fetch_depth_layer
  ${catkin_LIBRARIES}
)

install(FILES costmap_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS fetch_depth_layer
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

