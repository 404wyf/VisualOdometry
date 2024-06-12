cmake_minimum_required(VERSION 3.0)
project(PoseEstimation)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Threads REQUIRED)

# # 库文件链接到可执行程序上
# # 链接线程库


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${Threads_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)  # 添加这行以包含 `slam/common.h` 的路径

add_executable(main src/main.cpp src/view.cpp src/image_process.cpp src/map.cpp src/config.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${EIGEN3_LIBS}  ${THREADS_LIBS})