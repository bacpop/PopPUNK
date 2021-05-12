/*
 *
 * boundary.cpp
 * Functions to move a network boundary
 *
 */
#include <algorithm>
#include <cassert> // assert
#include <cmath>   // floor/sqrt
#include <cstddef> // size_t
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

// Parallel sort
#include <boost/sort/sort.hpp>

#include "boundary.hpp"

const float epsilon = 1E-10;

template <class T> inline size_t rows_to_samples(const T &longMat) {
  return 0.5 * (1 + sqrt(1 + 8 * (longMat.rows())));
}

inline size_t calc_row_idx(const uint64_t k, const size_t n) {
  return n - 2 -
         std::floor(
             std::sqrt(static_cast<double>(-8 * k + 4 * n * (n - 1) - 7)) / 2 -
             0.5);
}

inline size_t calc_col_idx(const uint64_t k, const size_t i, const size_t n) {
  return k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
}

inline size_t square_to_condensed(const size_t i, const size_t j,
                                  const size_t n) {
  assert(j > i);
  return (n * i - ((i * (i + 1)) >> 1) + j - 1 - i);
}

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

// Unnormalised (signed_ distance between a point (x0, y0) and a line defined
// by the two points (xmax, 0) and (0, ymax)
// Divide by 1/sqrt(xmax^2 + ymax^2) to get distance
inline float line_dist(const float x0, const float y0, const float x_max,
                       const float y_max, const int slope) {
  float boundary_side = 0;
  if (slope == 2) {
    boundary_side = y0 * x_max + x0 * y_max - x_max * y_max;
  } else if (slope == 0) {
    boundary_side = x0 - x_max;
  } else if (slope == 1) {
    boundary_side = y0 - y_max;
  }

  return boundary_side;
}

Eigen::VectorXf assign_threshold(const NumpyMatrix &distMat, const int slope,
                                 const float x_max, const float y_max,
                                 unsigned int num_threads) {
  Eigen::VectorXf boundary_test(distMat.rows());

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (long row_idx = 0; row_idx < distMat.rows(); row_idx++) {
    float in_tri = line_dist(distMat(row_idx, 0), distMat(row_idx, 1), x_max,
                             y_max, slope);
    float boundary_side;
    if (in_tri == 0) {
      boundary_side = 0;
    } else if (in_tri > 0) {
      boundary_side = 1;
    } else {
      boundary_side = -1;
    }
    boundary_test[row_idx] = boundary_side;
  }
  return (boundary_test);
}

edge_tuple edge_iterate(const NumpyMatrix &distMat, const int slope,
                        const float x_max, const float y_max) {
  const size_t n_samples = rows_to_samples(distMat);
  edge_tuple edge_vec;
  for (long row_idx = 0; row_idx < distMat.rows(); row_idx++) {
    if (line_dist(distMat(row_idx, 0), distMat(row_idx, 1), x_max, y_max,
                  slope) <= 0) {
      long i = calc_row_idx(row_idx, n_samples);
      long j = calc_col_idx(row_idx, i, n_samples);
      edge_vec.push_back(std::make_tuple(i, j));
    }
  }
  return edge_vec;
}

edge_tuple generate_tuples(const std::vector<int> &assignments, const int within_label) {
    const size_t n_rows = assignments.size();
    const size_t n_samples = 0.5 * (1 + sqrt(1 + 8 * (n_rows)));
    edge_tuple edge_vec;
    for (long row_idx = 0; row_idx < n_rows; row_idx++) {
        if (assignments[row_idx] == within_label) {
            long i = calc_row_idx(row_idx, n_samples);
            long j = calc_col_idx(row_idx, i, n_samples);
            edge_vec.push_back(std::make_tuple(i, j));
        }
    }
    return edge_vec;
}

// Line defined between (x0, y0) and (x1, y1)
// Offset is distance along this line, starting at (x0, y0)
network_coo threshold_iterate_1D(const NumpyMatrix &distMat,
                                 const std::vector<double> &offsets,
                                 const int slope, const float x0,
                                 const float y0, const float x1, const float y1,
                                 const int num_threads) {
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  std::vector<long> offset_idx;
  const float gradient = (y1 - y0) / (x1 - x0); // == tan(theta)
  const size_t n_samples = rows_to_samples(distMat);

  std::vector<float> boundary_dist(distMat.rows());
  std::vector<long> boundary_order;
  long sorted_idx = 0;
  for (int offset_nr = 0; offset_nr < offsets.size(); ++offset_nr) {
    float x_intercept = x0 + offsets[offset_nr] * (1 / std::sqrt(1 + gradient));
    float y_intercept =
        y0 + offsets[offset_nr] * (gradient / std::sqrt(1 + gradient));
    float x_max, y_max;
    if (slope == 2) {
      x_max = x_intercept + y_intercept * gradient;
      y_max = y_intercept + x_intercept / gradient;
    } else if (slope == 0) {
      x_max = x_intercept;
      y_max = 0;
    } else {
      x_max = 0;
      y_max = y_intercept;
    }

    // Calculate the distances and sort them on the first loop entry
    if (offset_nr == 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (long row_idx = 0; row_idx < distMat.rows(); row_idx++) {
        boundary_dist[row_idx] = line_dist(
            distMat(row_idx, 0), distMat(row_idx, 1), x_max, y_max, slope);
      }
      boundary_order = sort_indexes(boundary_dist, num_threads);
    }

    long row_idx = boundary_order[sorted_idx];
    while (sorted_idx < boundary_order.size() &&
           line_dist(distMat(row_idx, 0), distMat(row_idx, 1), x_max, y_max,
                     slope) <= 0) {
      long i = calc_row_idx(row_idx, n_samples);
      long j = calc_col_idx(row_idx, i, n_samples);
      i_vec.push_back(i);
      j_vec.push_back(j);
      offset_idx.push_back(offset_nr);
      sorted_idx++;
      row_idx = boundary_order[sorted_idx];
    }
  }
  return (std::make_tuple(i_vec, j_vec, offset_idx));
}

network_coo threshold_iterate_2D(const NumpyMatrix &distMat,
                                 const std::vector<float> &x_max,
                                 const float y_max) {
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  std::vector<long> offset_idx;
  const size_t n_samples = rows_to_samples(distMat);

  for (int offset_nr = 0; offset_nr < x_max.size(); ++offset_nr) {
    for (long row_idx = 0; row_idx < distMat.rows(); row_idx++) {
      if (line_dist(distMat(row_idx, 0), distMat(row_idx, 1), x_max[offset_nr],
                    y_max, 2) <= 0) {
        if (offset_nr == 0 ||
            (line_dist(distMat(row_idx, 0), distMat(row_idx, 1),
                       x_max[offset_nr - 1], y_max, 2) > 0)) {
          long i = calc_row_idx(row_idx, n_samples);
          long j = calc_col_idx(row_idx, i, n_samples);
          i_vec.push_back(i);
          j_vec.push_back(j);
          offset_idx.push_back(offset_nr);
        }
      }
    }
  }
  return (std::make_tuple(i_vec, j_vec, offset_idx));
}
