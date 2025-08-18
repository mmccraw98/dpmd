# cmake -S . -B build -DMD_DEBUG=0 (off) or 1 (on) && cd build && make -j8

cmake_minimum_required(VERSION 3.18)
project(md LANGUAGES C CXX CUDA)

# -----------------------------
# Options
# -----------------------------
option(MD_DEBUG "Enable heavy CUDA debug checks (sync + error after each launch)" OFF)

# -----------------------------
# Language & architectures
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 (Ada)
  # set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89)  # For compatibility with all gpus on the cluster
endif()

# -----------------------------
# Dependencies
# -----------------------------
find_package(HDF5 REQUIRED COMPONENTS C HL)

# -----------------------------
# Header-only utilities (include/utils/*.hpp)
# -----------------------------
add_library(md_utils INTERFACE)
target_include_directories(md_utils INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Prefer imported HDF5 targets if available
if(TARGET HDF5::C AND TARGET HDF5::HL)
  target_link_libraries(md_utils INTERFACE HDF5::C HDF5::HL)
else()
  target_link_libraries(md_utils INTERFACE HDF5::HDF5)
endif()
target_compile_options(md_utils INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

# MD_DEBUG define is visible to anything that consumes md_utils
if(MD_DEBUG)
  target_compile_definitions(md_utils INTERFACE MD_DEBUG=1)
else()
  target_compile_definitions(md_utils INTERFACE MD_DEBUG=0)
endif()

# -----------------------------
# Core library (src/*.cu, src/*.cpp)
# -----------------------------
file(GLOB_RECURSE MD_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
add_library(md_core STATIC ${MD_CORE_SOURCES})
if (MD_CORE_SOURCES STREQUAL "")
  file(WRITE "${CMAKE_BINARY_DIR}/dummy.cu" "__global__ void __dummy(){}")
  target_sources(md_core PRIVATE "${CMAKE_BINARY_DIR}/dummy.cu")
endif()

target_link_libraries(md_core PUBLIC md_utils curand)
target_include_directories(md_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(md_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(md_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# MD_DEBUG only after md_core exists
if(MD_DEBUG)
  message(STATUS "Building with MD_DEBUG=ON (CUDA launch sync+checks)")
  target_compile_definitions(md_core PUBLIC MD_DEBUG=1)
  # target_compile_options(md_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)  # uncomment for full device debug
else()
  message(STATUS "Building with MD_DEBUG=OFF")
  target_compile_definitions(md_core PUBLIC MD_DEBUG=0)
endif()

# Warnings (C++ only)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(md_core PUBLIC -Wall -Wextra -Wno-unknown-pragmas)
endif()

# -----------------------------
# Build type message
# -----------------------------
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "CMAKE_BUILD_TYPE=Debug")
else()
  message(STATUS "Building in Release (default)")
endif()

# -----------------------------
# Tests / examples
# -----------------------------
add_executable(test_device_fields tests/test_device_fields.cu)
set_target_properties(test_device_fields PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_device_fields PRIVATE md_core)

add_executable(test_base_particle tests/test_base_particle.cu)
set_target_properties(test_base_particle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_base_particle PRIVATE md_core)

add_executable(test_base_point_particle tests/test_base_point_particle.cu)
set_target_properties(test_base_point_particle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_base_point_particle PRIVATE md_core)

add_executable(test_disk tests/test_disk.cu)
set_target_properties(test_disk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_disk PRIVATE md_core)

# add_executable(main tests/main.cu)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(main PRIVATE md_core)

# add_executable(box_sample tests/box_sample.cu)
# set_target_properties(box_sample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(box_sample PRIVATE md_core)

# add_executable(rigid_box_sample tests/rigid_box_sample.cu)
# set_target_properties(rigid_box_sample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(rigid_box_sample PRIVATE md_core)