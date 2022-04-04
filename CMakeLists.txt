cmake_minimum_required(VERSION 3.16)
project(poppunk_refine)
set(CMAKE_CXX_STANDARD 14)

# Variable definitions
set(TARGET_NAME poppunk_refine)
add_compile_definitions(PYTHON_EXT)

# Add -O0 to remove optimizations when using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
    set(CMAKE_LD_FLAGS "${CMAKE_LDFLAGS} -Wl,--as-needed")
endif()

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

# Add libraries
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

execute_process(COMMAND pybind11-config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(OpenMP) # This links system openmp if present - conda sorts out rpath but take care

# Define python library target
add_library("${TARGET_NAME}" MODULE)

# Compile CPU library
target_sources("${TARGET_NAME}" PRIVATE src/python_bindings.cpp
                                        src/boundary.cpp
                                        src/extend.cpp)

set_target_properties("${TARGET_NAME}" PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module Eigen3::Eigen
                                               z gomp openblas gfortran m dl)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries("${TARGET_NAME}" PRIVATE OpenMP::OpenMP_CXX)
#endif()
