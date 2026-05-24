cmake_minimum_required(VERSION 3.8)
project(grid_mapper_2d)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation registration kdtree)
find_package(Eigen3 REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(grid_mapper_2d_node
  src/grid_mapper_2d_node.cpp
)

ament_target_dependencies(grid_mapper_2d_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)

target_link_libraries(grid_mapper_2d_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

install(TARGETS
  grid_mapper_2d_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()


