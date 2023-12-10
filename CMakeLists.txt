cmake_minimum_required(VERSION 3.14)
project(hw2 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")

find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# Common library target for benchmarking.
add_library(benchmark OBJECT benchmark.cpp)
#target_compile_features(benchmark PRIVATE cxx_std_11)
target_compile_options(benchmark PRIVATE -fopenmp -march=native)

# add_executable(benchmark-blas dgemm-blas.cpp)
# target_link_libraries(benchmark-blas PRIVATE benchmark ${BLAS_LIBRARIES})
# target_link_options(benchmark-blas PRIVATE -fopenmp)

# target_compile_features(benchmark-blas PRIVATE c_std_11 c_restrict)
# target_compile_options(benchmark-blas PRIVATE -Wall -pedantic -march=native)
# set(BENCHMARK "blas")
# configure_file(job.in job-${BENCHMARK})

add_executable(mpi_2dmesh mpi_2dmesh.cpp)
# include directories
target_include_directories(mpi_2dmesh PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

# link libraries
target_link_libraries(mpi_2dmesh PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

# eof
