cmake_minimum_required(VERSION 3.10)
project(sky-masking)
set(TARGET ${PROJECT_NAME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")
#add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset7")
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/result")

add_subdirectory(SkyDetector)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} SkyDetector)
