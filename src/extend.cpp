/*
 *
 * extend.cpp
 * Functions to extend a sparse distance matrix
 *
 */
#include "extend.hpp"
#include <cstddef> // size_t
#include <cstdint>
#include <pybind11/pybind11.h>
#include <vector>
#include <iostream>

// Get indices where each row starts in the sparse matrix
std::vector<long> row_start_indices(const sparse_coo &sparse_rr_mat,
                                    const size_t nr_samples) {
  const std::vector<long> i_vec = std::get<0>(sparse_rr_mat);
  std::vector<long> row_start_idx(nr_samples + 1);
  size_t i_idx = 0;
  row_start_idx[0] = 0;
  for (long i = 1; i < nr_samples; ++i) {
    while (i_vec[i_idx] < i && i_idx <= i_vec.size()) {
      i_idx++;
    }
    row_start_idx[i] = i_idx;

    // Set to end of vector, if reached
    if (i_idx == i_vec.size()) {
      for (long j = i + 1; j <= nr_samples; ++j) {
        row_start_idx[j] = i_idx;
      }
      break;
    }
  }
  // Ensure last range end is end of vector
  row_start_idx[nr_samples] = i_vec.size();
  return row_start_idx;
}

template <typename T>
std::vector<T> combine_vectors(const std::vector<std::vector<T>> &vec,
                               const size_t len) {
  std::vector<T> all(len);
  auto all_it = all.begin();
  for (size_t i = 0; i < vec.size(); ++i) {
    std::copy(vec[i].cbegin(), vec[i].cend(), all_it);
    all_it += vec[i].size();
  }
  return all;
}

sparse_coo extend(const sparse_coo &sparse_rr_mat,
                  const NumpyMatrix &qq_mat_square,
                  const NumpyMatrix &qr_mat_rect, const size_t kNN,
                  const size_t num_threads) {
  const size_t nr_samples = qr_mat_rect.rows();
  const size_t nq_samples = qr_mat_rect.cols();

  std::vector<long> row_start_idx =
      row_start_indices(sparse_rr_mat, nr_samples);

  // ijv vectors
  std::vector<std::vector<float>> dists(nr_samples + nq_samples);
  std::vector<std::vector<long>> i_vec(nr_samples + nq_samples);
  std::vector<std::vector<long>> j_vec(nr_samples + nq_samples);
  size_t len = 0;

  std::vector<float> dist_vec = std::get<2>(sparse_rr_mat);
#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:len)
  for (long i = 0; i < nr_samples + nq_samples; ++i) {
    // Extract the dists for the row from the qr (dense) and rr (sparse)
    // matrices
    Eigen::VectorXf rr_dists, qr_dists;
    if (i < nr_samples) {
      qr_dists = qr_mat_rect.row(i);
      long start_difference = row_start_idx[i + 1] - row_start_idx[i];
      long n_rr_dists = std::max(0L, start_difference);
      if (n_rr_dists > 0) {
        Eigen::Map<Eigen::VectorXf> rr_map(dist_vec.data() + row_start_idx[i],
                                           n_rr_dists);
        rr_dists = rr_map;
      }
    } else {
      rr_dists = qr_mat_rect.col(i - nr_samples);
      qr_dists = qq_mat_square.row(i - nr_samples);
    }

    // Sort these. Then do a merge below
    std::vector<long> qr_ordered_idx = sort_indexes(qr_dists, 1);
    std::vector<long> rr_ordered_idx = sort_indexes(rr_dists, 1);
    // See sparsify_dists in pp_sketchlib.
    // This is very similar, but merging two lists as input
    auto rr_it = rr_ordered_idx.cbegin();
    auto qr_it = qr_ordered_idx.cbegin();
    while (qr_it != qr_ordered_idx.cend() || rr_it != rr_ordered_idx.cend()) {
      // Get the next smallest dist, and corresponding j
      long j;
      float dist;
      if (rr_it == rr_ordered_idx.cend() ||
          (!(qr_it == qr_ordered_idx.cend()) &&
           qr_dists[*qr_it] <= rr_dists[*rr_it])) {
        j = *qr_it + nr_samples;
        dist = qr_dists[*qr_it];
        ++qr_it;
      } else if (!(rr_it == rr_ordered_idx.cend())) {
        if (i < nr_samples) {
          j = std::get<1>(sparse_rr_mat)[row_start_idx[i] + *rr_it];
        } else {
          j = *rr_it;
        }
        dist = rr_dists[*rr_it];
        ++rr_it;
      } else {
        std::cerr << "Insufficient distances for specified kNN value; try reducing the maximum search depth" << std::endl;
        pybind11::key_error();
      }

      if (j == i) {
        continue;
      }
      if (j_vec[i].size() < kNN) {
        dists[i].push_back(dist);
        i_vec[i].push_back(i);
        j_vec[i].push_back(j);
      } else {
        break; // move to next i
      }
    }
    len += dists[i].size();
  }

  // Combine the lists from each thread
  std::vector<float> dists_all = combine_vectors(dists, len);
  std::vector<long> i_vec_all = combine_vectors(i_vec, len);
  std::vector<long> j_vec_all = combine_vectors(j_vec, len);
  return (std::make_tuple(i_vec_all, j_vec_all, dists_all));
}

struct pair_hash {
  std::size_t operator()(const std::pair<long,long> & v) const {
    long hash_value = v.first;
    hash_value ^= v.second + 0x9e3779b9 + (v.first << 6) + (v.first >> 2);
    return hash_value;
  }
};

sparse_coo lower_rank(const sparse_coo &sparse_rr_mat, const size_t n_samples,
                      const size_t kNN, bool reciprocal_only,
                      bool count_unique_distances, float epsilon,
                      size_t num_threads = 1) {
  // Data structures for iteration
  size_t len = 0;
  std::vector<long> row_start_idx = row_start_indices(sparse_rr_mat, n_samples);
  std::vector<float> dist_vec = std::get<2>(sparse_rr_mat);
  const std::vector<long> j_sparse = std::get<1>(sparse_rr_mat);
  // ijv vectors
  std::vector<std::vector<float>> dists(n_samples);
  std::vector<std::vector<long>> i_vec(n_samples);
  std::vector<std::vector<long>> j_vec(n_samples);
#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:len)
  for (long i = 0; i < n_samples; ++i) {
    long start_difference = row_start_idx[i + 1] - row_start_idx[i];
    long n_rr_dists = std::max(0L, start_difference);
    if (n_rr_dists > 0) {
      Eigen::Map<Eigen::VectorXf> rr_dists(dist_vec.data() + row_start_idx[i],
                                           row_start_idx[i + 1] -
                                               row_start_idx[i]);
      std::vector<long> rr_ordered_idx = sort_indexes(rr_dists, 1);

      long unique_neighbors = 0;
      float prev_value = -1;
      bool new_val;
      for (auto rr_it = rr_ordered_idx.cbegin(); rr_it != rr_ordered_idx.cend();
           ++rr_it) {
        long j = std::get<1>(sparse_rr_mat)[row_start_idx[i] + *rr_it];
        float dist = rr_dists[*rr_it];
        if (j == i) {
          continue;
        }
        if (unique_neighbors < kNN) {
          dists[i].push_back(dist);
          i_vec[i].push_back(i);
          j_vec[i].push_back(j);
          if (count_unique_distances) {
            new_val = abs(dist - prev_value) >= epsilon;
            if (new_val) {
              unique_neighbors++;
              prev_value = dist;
            } else {
              continue; // next j
            }
          } else {
            unique_neighbors = j_vec[i].size();
          }
        } else {
          break; // next i
        }
      }
    }
    len += dists[i].size();
  }
  // Combine the lists from each thread
  std::vector<float> dists_all = combine_vectors(dists, len);
  std::vector<long> i_vec_all = combine_vectors(i_vec, len);
  std::vector<long> j_vec_all = combine_vectors(j_vec, len);
  // Only count reciprocal matches
  if (reciprocal_only) {
    std::unordered_set< std::pair<long, long>, pair_hash> kNN_pairs;
    for (long vector_index = 0; vector_index < i_vec_all.size(); ++vector_index) {
      if (i_vec_all[vector_index] > j_vec_all[vector_index]) {
        kNN_pairs.insert(std::make_pair(i_vec_all[vector_index],j_vec_all[vector_index]));
      }
    }
    std::vector<std::vector<float>> filtered_dists(n_samples);
    std::vector<std::vector<long>> filtered_i_vec(n_samples);
    std::vector<std::vector<long>> filtered_j_vec(n_samples);
    len = 0;
#pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:len)
    for (long i = 0; i < n_samples; ++i) {
      for (long j_index = 0; j_index < j_vec[i].size(); ++j_index) {
        long j = (j_vec[i])[j_index];
        if (i < j) {
          if (kNN_pairs.find(std::make_pair(j,i)) != kNN_pairs.end()) {
            filtered_dists[i].push_back((dists[i])[j_index]);
            filtered_j_vec[i].push_back(j);
          }
        }
      }
      filtered_i_vec[i].resize(filtered_j_vec[i].size(), i);
      len += filtered_i_vec[i].size();
    }
    // Combine the lists from each thread
    std::vector<float> filtered_dists_all =
        combine_vectors(filtered_dists, len);
    std::vector<long> filtered_i_vec_all = combine_vectors(filtered_i_vec, len);
    std::vector<long> filtered_j_vec_all = combine_vectors(filtered_j_vec, len);
    // Overwrite vectors
    dists_all = std::move(filtered_dists_all);
    i_vec_all = std::move(filtered_i_vec_all);
    j_vec_all = std::move(filtered_j_vec_all);
  }
  return (std::make_tuple(i_vec_all, j_vec_all, dists_all));
}

sparse_coo get_kNN_distances(const NumpyMatrix &distMat, const int kNN,
                             const size_t dist_col, const size_t num_threads) {

  size_t dist_rows = distMat.rows();
  std::vector<float> dists(dist_rows * kNN);
  std::vector<long> i_vec(dist_rows * kNN);
  std::vector<long> j_vec(dist_rows * kNN);

  bool interrupt = false;

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (size_t i = 0; i < dist_rows; ++i) {
    std::vector<float> row_dists(distMat.row(i).data(), distMat.row(i).data() + distMat.row(i).size());
    if (interrupt || PyErr_CheckSignals() != 0) {
      interrupt = true;
    } else {
      long offset = i * kNN;
      std::vector<long> j_ordered_by_dists = sort_indexes(row_dists, 1);
      std::fill_n(i_vec.begin() + offset, kNN, i);
      int k = 0;
      for (long j_index = 0; j_index < j_ordered_by_dists.size(); ++j_index) {
        if (j_ordered_by_dists[j_index] != i) {
          j_vec[offset + k] = j_ordered_by_dists[j_index];
          dists[offset + k] = row_dists[j_ordered_by_dists[j_index]];
          ++k;
          if (k == kNN) {
            break;
          }
        }
      }
    }
  }

  // Handle Ctrl-C from python
  if (interrupt) {
    throw pybind11::error_already_set();
  }

  return (std::make_tuple(i_vec, j_vec, dists));
}
