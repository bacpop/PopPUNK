/*
 * sketchlib_bindings.cpp
 * Python bindings for PopPUNK C++ functions
 *
 */

// pybind11 headers
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "boundary.hpp"

// Wrapper which makes a ref to the python/numpy array
Eigen::VectorXf assignThreshold(const Eigen::Ref<NumpyMatrix> &distMat,
                                const int slope, const double x_max,
                                const double y_max,
                                const unsigned int num_threads = 1) {
  Eigen::VectorXf assigned =
      assign_threshold(distMat, slope, x_max, y_max, num_threads);
  return (assigned);
}

edge_tuple edgeThreshold(const Eigen::Ref<NumpyMatrix> &distMat,
                         const int slope, const double x_max,
                         const double y_max) {
  edge_tuple edges = edge_iterate(distMat, slope, x_max, y_max);
  return (edges);
}

edge_tuple generateTuples(const std::vector<int> &assignments,
                            const int within_label,
                            bool self,
                            const int num_ref,
                            const int int_offset) {
    edge_tuple edges = generate_tuples(assignments,
                                        within_label,
                                        self,
                                        num_ref,
                                        int_offset);
    return (edges);
}

edge_tuple generateAllTuples(const int num_ref,
                                const int num_queries,
                                bool self = true,
                                const int int_offset = 0) {
    edge_tuple edges = generate_all_tuples(num_ref,
                                            num_queries,
                                            self,
                                            int_offset);
    return (edges);
}

network_coo thresholdIterate1D(const Eigen::Ref<NumpyMatrix> &distMat,
                               const std::vector<double> &offsets,
                               const int slope, const double x0,
                               const double y0, const double x1,
                               const double y1, const int num_threads) {
  if (!std::is_sorted(offsets.begin(), offsets.end())) {
    throw std::runtime_error("Offsets to thresholdIterate1D must be sorted");
  }
  std::tuple<std::vector<long>, std::vector<long>, std::vector<long>> add_idx =
      threshold_iterate_1D(distMat, offsets, slope, x0, y0, x1, y1,
                           num_threads);
  return (add_idx);
}

network_coo thresholdIterate2D(const Eigen::Ref<NumpyMatrix> &distMat,
                               const std::vector<float> &x_max,
                               const float y_max) {
  if (!std::is_sorted(x_max.begin(), x_max.end())) {
    throw std::runtime_error(
        "x_max range to thresholdIterate2D must be sorted");
  }
  std::tuple<std::vector<long>, std::vector<long>, std::vector<long>> add_idx =
      threshold_iterate_2D(distMat, x_max, y_max);
  return (add_idx);
}

PYBIND11_MODULE(poppunk_refine, m) {
  m.doc() = "Network refine helper functions";

  // Exported functions
  m.def("assignThreshold", &assignThreshold,
        py::return_value_policy::reference_internal,
        "Assign samples based on their relation to a 2D boundary",
        py::arg("distMat").noconvert(), py::arg("slope"), py::arg("x_max"),
        py::arg("y_max"), py::arg("num_threads") = 1);

  m.def("edgeThreshold", &edgeThreshold,
        py::return_value_policy::reference_internal,
        "Assign samples based on their relation to a 2D boundary, returning an "
        "edge list",
        py::arg("distMat").noconvert(), py::arg("slope"), py::arg("x_max"),
        py::arg("y_max"));

  m.def("generateTuples", &generateTuples,
        py::return_value_policy::reference_internal,
        "Return edge tuples based on assigned groups",
        py::arg("assignments"), py::arg("within_label"),
        py::arg("self") = true, py::arg("num_ref"),
        py::arg("int_offset"));

  m.def("generateAllTuples", &generateAllTuples,
        py::return_value_policy::reference_internal,
        "Return all edge tuples",
        py::arg("num_ref"),
        py::arg("num_queries"),
        py::arg("self") = true,
        py::arg("int_offset"));
    
  m.def("thresholdIterate1D", &thresholdIterate1D,
        py::return_value_policy::reference_internal,
        "Move a 2D boundary to grow a network by adding edges at each offset",
        py::arg("distMat").noconvert(), py::arg("offsets"), py::arg("slope"),
        py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
        py::arg("num_threads"));

  m.def("thresholdIterate2D", &thresholdIterate2D,
        py::return_value_policy::reference_internal,
        "Move a 2D boundary to grow a network by adding edges at each offset",
        py::arg("distMat").noconvert(), py::arg("x_max"), py::arg("y_max"));
}
