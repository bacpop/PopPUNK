/*
 * linear_regression.cpp
 * Regresses k-mer lengths and matches
 *
 */

#include "linear_regression.hpp"

// Constants
const std::string VERSION = "1.0.0";

// Wrapper around fitKmer block which multithreads calculation
void fitKmers(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
              py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
              const column_vector& klist,
              const int num_threads)
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
        unsigned long int thread_jobs = calc_per_thread;
        if (thread_idx < num_big_threads)
        {
            thread_jobs++;
        }
        work_threads.push_back(std::thread(&fitKmerBlock,
                                           std::cref(raw),
                                           std::ref(dists),
                                           std::cref(klist),
                                           start,
                                           start + thread_jobs));
        start += thread_jobs + 1;
    }

    // Wait for threads to complete
    for (auto it = work_threads.begin(); it != work_threads.end(); it++)
    {
        it->join();
    }
}

void fitKmerBlock(const py::array_t<float, py::array::c_style | py::array::forcecast>& raw,
                  py::array_t<double, py::array::c_style | py::array::forcecast>& dists,
                  const column_vector& klist,
                  const size_t start,
                  const size_t end)
{
    const column_vector x_lower(2, -std::numeric_limits<double>::infinity());
    const column_vector x_upper(2, 0);

    auto raw_access = raw.unchecked<2>();
    auto dist_access = dists.mutable_unchecked<2>();
    for (size_t row = start; row <= end; row++)
    {
        column_vector y_vec(klist.nr());
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
                [&linear_fit](const column_vector& a) {
                    return linear_fit.likelihood(a);
                },
                [&linear_fit](const column_vector& b) {
                    return linear_fit.gradient(b);
                },
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
}