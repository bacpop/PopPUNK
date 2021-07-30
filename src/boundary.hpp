/*
 *
 * matrix.hpp
 * functions in matrix_ops.cpp
 *
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>
    network_coo;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<float>>
    sparse_coo;
typedef std::vector<std::tuple<long, long>> edge_tuple;

template <typename T>
std::vector<long> sort_indexes(const T &v, const uint32_t n_threads)

Eigen::VectorXf assign_threshold(const NumpyMatrix &distMat, const int slope,
                                 const float x_max, const float y_max,
                                 unsigned int num_threads);

edge_tuple edge_iterate(const NumpyMatrix &distMat, const int slope,
                        const float x_max, const float y_max);

edge_tuple generate_tuples(const std::vector<int> &assignments,
                           const int within_label,
                           bool self,
                           const int num_ref,
                           const int int_offset);

edge_tuple generate_all_tuples(const int num_ref,
                                const int num_queries,
                                bool self,
                                const int int_offset);

network_coo threshold_iterate_1D(const NumpyMatrix &distMat,
                                 const std::vector<double> &offsets,
                                 const int slope, const float x0,
                                 const float y0, const float x1, const float y1,
                                 const int num_threads);

network_coo threshold_iterate_2D(const NumpyMatrix &distMat,
                                 const std::vector<float> &x_max,
                                 const float y_max);
