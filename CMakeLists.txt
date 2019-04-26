cmake_minimum_required(VERSION 3.0)
project(kmer_regression)

find_package(pybind11 REQUIRED)
find_package(dlib REQUIRED)
# add_subdirectory(../dlib dlib_build)

pybind11_add_module(kmer_regression src/linear_regression.cpp src/regression_bindings.cpp)
target_link_libraries(kmer_regression PRIVATE pthread dlib::dlib)