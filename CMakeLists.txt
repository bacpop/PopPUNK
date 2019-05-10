cmake_minimum_required(VERSION 3.0)
project(kmer_regression)

find_package(pybind11 REQUIRED)

add_definitions(-D DLIB_NO_GUI_SUPPORT=1 -D DLIB_USE_BLAS=1 -D DLIB_USE_LAPACK=1)
include_directories(${PROJECT_SOURCE_DIR}/dlib)
#add_subdirectory(dlib dlib_build)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

pybind11_add_module(kmer_regression src/link_function.cpp src/linear_regression.cpp src/regression_bindings.cpp)
target_link_libraries(kmer_regression PRIVATE Threads::Threads lapack blas)