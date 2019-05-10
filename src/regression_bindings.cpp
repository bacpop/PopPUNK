/*
 * regression_bindings.cpp
 * Python bindings for linear_regression
 *
 */

#include "linear_regression.hpp"

void fit_all(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
             py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& klist,
             int num_threads)
{
    // Check input
    if (num_threads < 1)
    {
        num_threads = 1;
    }

    if (raw.shape()[0] != dists.shape()[0])
    {
        throw std::runtime_error("Number of rows in raw and dists must match");
    }
    if (raw.shape()[1] != klist.shape()[0])
    {
        throw std::runtime_error("Number of columns in raw must match length of klist");
    }
    if (dists.ndim() != 2 || dists.shape()[1] != 2)
    {
        throw std::runtime_error("Dists must be a 2D array with two columns");
    }
    if (raw.ndim() != 2)
    {
        throw std::runtime_error("Raw must be a 2D array");
    }
    if (klist.ndim() != 1)
    {
        throw std::runtime_error("klist must be a vector");
    }

    // convert klist
    column_vector k_vec(klist.size());
    auto r = klist.unchecked<1>();
    for (unsigned int i = 0; i < klist.size(); ++i)
    {
        k_vec(i) = r(i);
    }
    //std::memcpy(k_vec.data(), klist.data(), klist.size()*sizeof(double));

    // call C++ function which changes dists in place
    // returns (Na, ta) tuple
    fitKmers(raw, dists, k_vec, num_threads);
}

PYBIND11_MODULE(kmer_regression, m)
{
  m.doc() = "Performs regressions on k-mer lengths";

  m.def("fit_all", &fit_all, "Calculate core and accessory distances",
        py::arg("raw").noconvert(), py::arg("dists").noconvert(), py::arg("klist"), py::arg("num_threads") = 1);
}