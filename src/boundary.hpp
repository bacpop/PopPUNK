/*
 *
 * boundary.hpp
 * prototypes for boundary fns
 *
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

// Parallel sort
#include <boost/sort/sort.hpp>
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    NumpyMatrix;
typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<long>>
    network_coo;
typedef std::vector<std::tuple<long, long>> edge_tuple;

// https://stackoverflow.com/a/12399290
template <typename T>
std::vector<long> sort_indexes(const T &v, const uint32_t n_threads) {
  // initialize original index locations
  std::vector<long> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  boost::sort::parallel_stable_sort(
      idx.begin(), idx.end(), [&v](long i1, long i2) { return v[i1] < v[i2]; },
      n_threads);

  return idx;
}

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
