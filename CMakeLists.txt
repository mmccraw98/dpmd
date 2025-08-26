# rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA_DEBUG=ON .. && make -j8
# rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA_DEBUG=OFF .. && make -j8
# -DCMAKE_BUILD_TYPE= [Debug, Release]
# -DENABLE_CUDA_DEBUG= [ON, OFF]
# -DENABLE_PBC_DIST= [ON, OFF]

cmake_minimum_required(VERSION 3.18)

# Project setup with CUDA and C languages (C needed for HDF5)
project(dpmd CUDA CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation for device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Debug option for CUDA debugging
option(ENABLE_CUDA_DEBUG "Enable CUDA debug mode with heavy sync and error checking" OFF)

# PBC option for periodic boundary conditions
option(ENABLE_PBC_DIST "Enable periodic boundary conditions in force calculations" ON)

# Find required packages
find_package(HDF5 REQUIRED COMPONENTS C)

# CUDA architecture setting
set(CMAKE_CUDA_ARCHITECTURES "89")
# set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89)  # For compatibility with all gpus on the cluster

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${HDF5_INCLUDE_DIRS})

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set MD_DEBUG flag based on option
if(ENABLE_CUDA_DEBUG)
    add_compile_definitions(MD_DEBUG=1)
    message(STATUS "CUDA Debug mode: ENABLED (heavy sync and error checking)")
else()
    add_compile_definitions(MD_DEBUG=0)
    message(STATUS "CUDA Debug mode: DISABLED (performance mode)")
endif()

# Core library sources
set(CORE_KERNEL_SOURCES
    src/kernels/common_globals.cu
    src/kernels/base_point_particle_kernels.cu
    src/kernels/base_poly_particle_kernels.cu
)

set(CORE_PARTICLE_SOURCES
    src/particles/disk.cu
    src/particles/rigid_bumpy.cu
)

set(CORE_UTILS_SOURCES
    src/utils/device_fields.cu
)

# Create core library
add_library(dpmd_core ${CORE_KERNEL_SOURCES} ${CORE_PARTICLE_SOURCES} ${CORE_UTILS_SOURCES})

# Link libraries to core
target_link_libraries(dpmd_core ${HDF5_LIBRARIES})

# Set properties for core library
set_target_properties(dpmd_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Test executables
set(TEST_SOURCES
    tests/test_base_particle.cu
    tests/test_base_point_particle.cu
    tests/test_base_poly_particle.cu
    tests/test_device_fields.cu
    tests/test_disk.cu
    tests/test_disk_system_naive_parallelism.cu
    tests/test_disk_system_cell_parallelism.cu
    tests/rigid_bumpy_box_sample.cu
    tests/jam_rigid_bumpy.cu
    tests/damped_jam_rigid_bumpy.cu
)

# Create test executables
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} dpmd_core ${HDF5_LIBRARIES})
    set_target_properties(${test_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()

# Optional: Create install targets
install(TARGETS dpmd_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

# Optional: Enable testing
enable_testing()

# Add some key tests to CTest
add_test(NAME test_base_particle COMMAND test_base_particle)
add_test(NAME test_device_fields COMMAND test_device_fields)
add_test(NAME test_disk COMMAND test_disk)
