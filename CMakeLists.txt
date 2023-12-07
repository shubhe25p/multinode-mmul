cmake_minimum_required(VERSION 3.14)
project(mpi_2dmesh_harness LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

find_package(MPI REQUIRED)

add_executable(mpi_2dmesh_sqdgem mpi_2dmesh_sqdgem.cpp)
# include directories
target_include_directories(mpi_2dmesh_sqdgem PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

# link libraries
target_link_libraries(mpi_2dmesh_sqdgem PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(mpi_2dmesh_cblas mpi_2dmesh_cblas.cpp)
# include directories
target_include_directories(mpi_2dmesh_cblas PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

# link libraries
target_link_libraries(mpi_2dmesh_cblas PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})