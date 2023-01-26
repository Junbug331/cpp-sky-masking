
cmake_minimum_required(VERSION 3.10)
project(SkyDetector)
set(TARGET ${PROJECT_NAME})


set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/sky_detector.cpp
)

find_package(OpenCV REQUIRED)

add_library(${TARGET} ${SRC_FILES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
)