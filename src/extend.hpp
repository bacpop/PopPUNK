#pragma once

#include "boundary.hpp"

sparse_coo extend(const sparse_coo &sparse_rr_mat,
                  const NumpyMatrix &qq_mat_square,
                  const NumpyMatrix &qr_mat_rect, const size_t kNN);

sparse_coo lower_rank(const sparse_coo &sparse_rr_mat, const size_t kNN);
