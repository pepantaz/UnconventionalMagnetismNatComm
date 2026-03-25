cmake_minimum_required(VERSION 3.16)
project(UnconventionalFiguresCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(
    unconventional_figures
    src/main.cpp
    src/unconventional_figures.cpp
    src/multilayer_greens_function.cpp
)

target_include_directories(unconventional_figures PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    /Users/petros/anaconda3/lib/python3.10/site-packages/numpy/core/include
    /opt/homebrew/include
    /opt/homebrew/include/eigen3
    /opt/homebrew/opt/boost/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(unconventional_figures PRIVATE
    Python3::Python
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_compile_options(unconventional_figures PRIVATE
    -O2 -Wall -Wextra -Wpedantic
)