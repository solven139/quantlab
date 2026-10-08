cmake_minimum_required(VERSION 3.18)
project(quantlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Pure C++ library: knows nothing about Python
add_library(quantlab_core STATIC
    cpp/src/black_scholes.cpp
    cpp/src/gbm.cpp
    cpp/src/hedging.cpp
    cpp/src/implied_vol.cpp
    cpp/src/models.cpp
    cpp/src/cos.cpp
    cpp/src/jump_paths.cpp
    cpp/src/heston_paths.cpp
    cpp/src/mc_schemes.cpp
    cpp/src/local_vol.cpp
)
target_include_directories(quantlab_core PUBLIC cpp/include)
set_target_properties(quantlab_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Thin Python binding layer
pybind11_add_module(_core cpp/bindings/module.cpp)
target_link_libraries(_core PRIVATE quantlab_core)

install(TARGETS _core DESTINATION quantlab)
