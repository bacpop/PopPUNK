/*
 *
 * extend.cpp
 * Functions to extend a sparse distance matrix
 *
 */
#include <cstddef> // size_t
#include <cstdint>
#include <vector>

#include "extend.hpp"

const float epsilon = 1E-10;

// Get indices where each row starts in the sparse matrix
std::vector<long> row_start_indices(const sparse_coo &sparse_rr_mat,
                                    const size_t nr_samples) {
  const std::vector<long> i_vec = std::get<0>(sparse_rr_mat);
  std::vector<long> row_start_idx(nr_samples + 1);
  size_t i_idx = 0;
  row_start_idx[0] = 0;
  row_start_idx[nr_samples] = i_vec.size();
  for (long i = 1; i < nr_samples; ++i) {
    while (i_vec[i_idx] < i) {
      i_idx++;
    }
    row_start_idx[i] = i_idx;
  }
  return row_start_idx;
}

sparse_coo extend(const sparse_coo &sparse_rr_mat, const NumpyMatrix &qq_mat_square,
                  const NumpyMatrix &qr_mat_rect, const size_t kNN) {
  const size_t nr_samples = qr_mat_rect.rows();
  const size_t nq_samples = qr_mat_rect.cols();

  std::vector<long> row_start_idx =
      row_start_indices(sparse_rr_mat, nr_samples);

  std::cout << qq_mat_square << std::endl;
  std::cout << qr_mat_rect << std::endl;

  // ijv vectors
  std::vector<float> dists;
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  std::vector<float> dist_vec = std::get<2>(sparse_rr_mat);
  for (long i = 0; i < nr_samples + nq_samples; ++i) {
    // Extract the dists for the row from the qr (dense) and rr (sparse)
    // matrices
    Eigen::VectorXf rr_dists, qr_dists;
    if (i < nr_samples) {
      qr_dists = qr_mat_rect.row(i);
      Eigen::Map<Eigen::VectorXf> rr_map(dist_vec.data() + row_start_idx[i],
                                         row_start_idx[i + 1] -
                                             row_start_idx[i]);
      rr_dists = rr_map;
    } else {
      rr_dists = qr_mat_rect.col(i - nr_samples);
      qr_dists = qq_mat_square.row(i - nr_samples);
    }

    // Sort these. Then do a merge below
    std::vector<long> qr_ordered_idx = sort_indexes(qr_dists, 1);
    std::vector<long> rr_ordered_idx = sort_indexes(rr_dists, 1);
    for (int z = 0; z < qr_ordered_idx.size(); ++z) {
      std::cout << z << "\t" << qr_ordered_idx[z] << "\t" << qr_dists[qr_ordered_idx[z]] << std::endl;
    }
    for (int z = 0; z < rr_ordered_idx.size(); ++z) {
      std::cout << z << "\t" << rr_ordered_idx[z] << "\t" << rr_dists[rr_ordered_idx[z]] << std::endl;
    }

    // See sparsify_dists in pp_sketchlib.
    // This is very similar, but merging two lists as input
    long unique_neighbors = 0;
    float prev_value = -1;
    auto rr_it = rr_ordered_idx.cbegin();
    auto qr_it = qr_ordered_idx.cbegin();
    while (qr_it != qr_ordered_idx.cend() && rr_it != rr_ordered_idx.cend()) {
      // Get the next smallest dist, and corresponding j
      long j;
      float dist;
      if (rr_it == rr_ordered_idx.cend() ||
          (!(qr_it == qr_ordered_idx.cend()) &&
           qr_dists[*qr_it] <= rr_dists[*rr_it])) {
        j = *qr_it + nr_samples;
        dist = qr_dists[*qr_it];
        ++qr_it;
      } else {
        j = *rr_it;
        dist = rr_dists[*rr_it];
        ++rr_it;
      }

      if (j == i) {
        continue;
      }
      bool new_val = abs(dist - prev_value) < epsilon;
      if (unique_neighbors < kNN || new_val) {
        dists.push_back(dist);
        i_vec.push_back(i);
        j_vec.push_back(j);
        if (!new_val) {
          unique_neighbors++;
          prev_value = dist;
        }
      } else {
        break; // next i
      }
    }
  }
  return (std::make_tuple(i_vec, j_vec, dists));
}

sparse_coo lower_rank(const sparse_coo &sparse_rr_mat, const size_t n_samples,
                      const size_t kNN) {
  std::vector<long> row_start_idx =
      row_start_indices(sparse_rr_mat, n_samples);

  // ijv vectors
  std::vector<float> dists;
  std::vector<long> i_vec;
  std::vector<long> j_vec;
  std::vector<float> dist_vec = std::get<2>(sparse_rr_mat);
  const std::vector<long> j_sparse = std::get<1>(sparse_rr_mat);
  for (long i = 0; i < n_samples; ++i) {
    Eigen::Map<Eigen::VectorXf> rr_dists(dist_vec.data() + row_start_idx[i],
                                         row_start_idx[i + 1] -
                                             row_start_idx[i]);
    std::vector<long> rr_ordered_idx = sort_indexes(rr_dists, 1);

    long unique_neighbors = 0;
    float prev_value = -1;
    for (auto rr_it = rr_ordered_idx.cbegin(); rr_it != rr_ordered_idx.cend();
         ++rr_it) {
      long j = std::get<1>(sparse_rr_mat)[row_start_idx[i] + *rr_it];
      float dist = rr_dists[*rr_it];
      if (j == i) {
        continue;
      }
      bool new_val = abs(dist - prev_value) < epsilon;
      if (unique_neighbors < kNN || new_val) {
        dists.push_back(dist);
        i_vec.push_back(i);
        j_vec.push_back(j);
        if (!new_val) {
          unique_neighbors++;
          prev_value = dist;
        }
      } else {
        break; // next i
      }
    }
  }
  return (std::make_tuple(i_vec, j_vec, dists));
}
