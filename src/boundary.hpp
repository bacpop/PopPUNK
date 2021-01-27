/*
 *
 * matrix.hpp
 * functions in matrix_ops.cpp
 *
 */
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<long>> network_coo;

Eigen::VectorXf assign_threshold(const NumpyMatrix &distMat,
                                 const int slope,
                                 const float x_max,
                                 const float y_max,
                                 unsigned int num_threads);

network_coo threshold_iterate_1D(const NumpyMatrix &distMat,
                                 const std::vector<double> &offsets,
                                 const int slope,
                                 const float x0,
                                 const float y0,
                                 const float x1,
                                 const float y1,
                                 const int num_threads = 1);

network_coo threshold_iterate_2D(const NumpyMatrix &distMat,
                                 const std::vector<float> &x_max,
                                 const float y_max);
