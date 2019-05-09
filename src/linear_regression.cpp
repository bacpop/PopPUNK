/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */

#include "linear_regression.hpp"

// Constants
const std::string VERSION = "1.0.0";

// Wrapper around fitKmer block which multithreads calculation
void fitKmers(py::array_t<double, py::array::c_style | py::array::forcecast>& raw,
              py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
              py::array_t<double, py::array::c_style | py::array::forcecast>& klist,
              int num_threads)
{
    // Create threaded queue for distance calculations
    std::vector<std::thread> work_threads(num_threads);
    const unsigned long int calc_per_thread = (unsigned long int)raw.shape()[0] / num_threads;
    const unsigned int num_big_threads = raw.shape()[0] % num_threads;

    // Spawn worker threads
    size_t start = 0;
    for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) // Loop over threads
    {
        // First 'big' threads have an extra job
        thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads)
        {
            thread_jobs++;
        }
        work_threads.push_back(std::thread(fitKmerBlock, raw, dists, klist, start, start + thread_jobs));
        start += thread_jobs + 1;
    }

    // Wait for threads to complete
    for (auto it = work_threads.begin(); it != work_threads.end(); it++)
    {
        it->join();
    }
}

void fitKmerBlock(py::array_t<double, py::array::c_style | py::array::forcecast>& raw,
                  py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
                  column_vector& klist,
                  size_t start,
                  size_t end)
{
    // TODO: do these initialise correctly?
    const column_vector x_lower(2, -dlib::infinity);
    const column_vector x_upper(2, 0);

    auto raw_access = raw.unchecked<double, 2>;
    auto dist_access = dists.mutable_unchecked<double, 2>;
    for (size_t row = start; row <= end; row++)
    {
        const column_vector y_vec(klist.nr());
        for (unsigned int i = 0; i < y_vec.nr(); ++i)
        {
            y_vec(i) = raw_access(row, i);
        }
        LinearLink linear_fit(klist, y_vec);

        column_vector starting_point(2);
        starting_point(0) = -0.01;
        starting_point(1) = 0;

        try
        {
            dlib::find_max_box_constrained(
                           dlib::bfgs_search_strategy(),
                           dlib::objective_delta_stop_strategy(convergence_limit),
                           linear_fit.likelihood,
                           linear_fit.gradient,
                           starting_point,
                           x_lower,
                           x_upper);

            // Store core/accessory in dists
            dist_access(row, 0) = 1 - exp(starting_point(1));
            dist_access(row, 1) = 1 - exp(starting_point(0));
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << "Fitting k-mer gradient failed, matches:" << std::endl;
            for (unsigned int i = 0; i < y_vec.nr(); ++i)
            {
                std::cerr << "\t" << y_vec(i);
            }
            std::cerr << std::endl << "Check for low quality genomes" << std::endl;
            exit(1);
        }
}