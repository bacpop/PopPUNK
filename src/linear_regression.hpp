/*
 *
 * linear_regression.hpp
 * Header file for regression
 *
 */

// C/C++/C++11 headers
#include <string>
#include <random>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <functional>
#include <limits>

// dlib headers
#include <dlib/matrix.h>
#include <dlib/optimization.h>
typedef dlib::matrix<double,0,1> column_vector;

// pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "link_function.hpp"

namespace py = pybind11;

// Constants
extern const std::string VERSION;
const double convergence_limit = 1e-7;

// Function headers for each cpp file

// linear_regression.cpp
void fitKmers(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
              py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
              const column_vector& klist,
              const int num_threads = 1);
void fitKmerBlock(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
                  py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
                  const column_vector& klist,
                  const size_t start,
                  const size_t end);

// regression_bindings.cpp
void fit_all(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
             py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& klist,
             int num_threads);

