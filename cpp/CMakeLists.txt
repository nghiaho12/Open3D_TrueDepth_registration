cmake_minimum_required(VERSION 3.0)

project(pose_graph)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_INSTALL_PREFIX .)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(pybind11)
pybind11_add_module(pose_graph pose_graph.cpp)
target_link_libraries(pose_graph PRIVATE ${CERES_LIBRARIES})

install (TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ..
)