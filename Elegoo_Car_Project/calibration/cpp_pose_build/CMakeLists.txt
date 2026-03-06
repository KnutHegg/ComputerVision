cmake_minimum_required(VERSION 3.20)
project(cpp_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(cpp_pose pose_bindings.cpp)
target_link_libraries(cpp_pose PRIVATE ${OpenCV_LIBS})
target_include_directories(cpp_pose PRIVATE ${OpenCV_INCLUDE_DIRS})

if (MSVC)
  target_compile_options(cpp_pose PRIVATE /EHsc)
endif()