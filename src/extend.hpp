#pragma once

#include "boundary.hpp"

typedef std::tuple<std::vector<long>, std::vector<long>, std::vector<float>>
    sparse_coo;

sparse_coo extend(const sparse_coo &sparse_rr_mat,
                  const NumpyMatrix &qq_mat_square,
                  const NumpyMatrix &qr_mat_rect,
                  const size_t kNN,
                  const size_t num_threads);

sparse_coo lower_rank(const sparse_coo &sparse_rr_mat,
                      const size_t n_samples,
                      const size_t kNN,
                      bool reciprocal_only,
                      bool count_neighbours,
                      float epsilon,
                      const size_t num_threads);

sparse_coo get_kNN_distances(const NumpyMatrix &distMat,
                             const int kNN,
                             const size_t dist_col,
                             const size_t num_threads);
