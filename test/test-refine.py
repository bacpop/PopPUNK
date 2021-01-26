import os, sys
import numpy as np
from math import sqrt

# testing without install
#sys.path.insert(0, '../build/lib.macosx-10.9-x86_64-3.8')
import poppunk_refine

# Original PopPUNK function (with some improvements)
def withinBoundary(dists, x_max, y_max, slope=2):
    boundary_test = np.ones((dists.shape[0]))
    for row in range(boundary_test.size):
        if slope == 2:
            in_tri = dists[row, 1] * x_max + dists[row, 0] * y_max - x_max * y_max
        elif slope == 0:
            in_tri = dists[row, 0] - x_max
        elif slope == 1:
            in_tri = dists[row, 1] - y_max
        if abs(in_tri) < np.finfo(np.float32).eps:
          boundary_test[row] = 0
        elif in_tri < 0:
            boundary_test[row] = -1
    return(boundary_test)

def check_res(res, expected):
    if (not np.all(res == expected)):
        print(res)
        print(expected)
        raise RuntimeError("Results don't match")

# assigning
x = np.arange(0, 1, 0.1, dtype=np.float32)
y = np.arange(0, 1, 0.1, dtype=np.float32)
xv, yv = np.meshgrid(x, y)
distMat = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
assign0 = poppunk_refine.assignThreshold(distMat, 0, 0.5, 0.5, 2)
assign1 = poppunk_refine.assignThreshold(distMat, 1, 0.5, 0.5, 2)
assign2 = poppunk_refine.assignThreshold(distMat, 2, 0.5, 0.5, 2)

assign0_res = withinBoundary(distMat, 0.5, 0.5, 0)
assign1_res = withinBoundary(distMat, 0.5, 0.5, 1)
assign2_res = withinBoundary(distMat, 0.5, 0.5, 2)

check_res(assign0, assign0_res)
check_res(assign1, assign1_res)
check_res(assign2, assign2_res)

# move boundary 1D
# example is symmetrical at points (0.1, 0.1); (0.2, 0.2); (0.3, 0.3)
samples = 100
distMat = np.random.rand(int(0.5 * samples * (samples - 1)), 2)
distMat = np.array(distMat, dtype = np.float32)
offsets = [x * sqrt(2) for x in [-0.1, 0.0, 0.1]]
i_vec, j_vec, idx_vec = poppunk_refine.thresholdIterate1D(distMat, offsets, 2, 0.2, 0.2, 0.3, 0.3)
sketchlib_i = []
sketchlib_j = []
for offset_idx, offset in enumerate(offsets):
  for i, j, idx in zip(i_vec, j_vec, idx_vec):
    if idx > offset_idx:
      break
    elif idx == offset_idx:
      sketchlib_i.append(i)
      sketchlib_j.append(j)

  py_i = []
  py_j = []
  xmax = 0.4 + (2 * (offset/sqrt(2)))
  assign = poppunk_refine.assignThreshold(distMat, 2, xmax, xmax, 1)
  dist_idx = 0
  for i in range(samples):
    for j in range(i + 1, samples):
      if assign[dist_idx] <= 0:
        py_i.append(i)
        py_j.append(j)
      dist_idx += 1
  if set(zip(py_i, py_j)) != set(zip(sketchlib_i, sketchlib_j)):
    raise RuntimeError("Threshold 1D iterate mismatch at offset " + str(offset))

# move boundary 2D
# example is for boundaries (0.1, 0.2); (0.2, 0.2); (0.3, 0.2)
offsets = [0.1, 0.2, 0.3]
y_max = 0.2
i_vec, j_vec, idx_vec = poppunk_refine.thresholdIterate2D(distMat, offsets, y_max)
sketchlib_i = []
sketchlib_j = []
for offset_idx, offset in enumerate(offsets):
  for i, j, idx in zip(i_vec, j_vec, idx_vec):
    if idx > offset_idx:
      break
    elif idx == offset_idx:
      sketchlib_i.append(i)
      sketchlib_j.append(j)

  py_i = []
  py_j = []
  assign = poppunk_refine.assignThreshold(distMat, 2, offset, y_max, 1)
  dist_idx = 0
  for i in range(samples):
    for j in range(i + 1, samples):
      if assign[dist_idx] <= 0:
        py_i.append(i)
        py_j.append(j)
      dist_idx += 1
  if set(zip(py_i, py_j)) != set(zip(sketchlib_i, sketchlib_j)):
    raise RuntimeError("Threshold 2D iterate mismatch at offset " + str(offset))

