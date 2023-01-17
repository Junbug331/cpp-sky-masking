cmake_minimum_required(VERSION 3.10)
project(sky-masking)
set(TARGET ${PROJECT_NAME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_subdirectory(SkyDetector)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} SkyDetector)