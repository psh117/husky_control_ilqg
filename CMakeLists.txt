cmake_minimum_required(VERSION 2.8.3)
project(husky_control_ilqg)

add_compile_options(-std=c++11)

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp eigen_conversions tf
)

catkin_package(
)


include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
add_executable(${PROJECT_NAME} src/${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

