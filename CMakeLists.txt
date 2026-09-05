cmake_minimum_required(VERSION 3.28)

project(hllm_runtime VERSION 0.1.0 LANGUAGES CXX)

option(HLLM_BUILD_TESTS "Build native tests" ON)
option(HLLM_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(hllm_project_options INTERFACE)
target_compile_features(hllm_project_options INTERFACE cxx_std_20)

if(MSVC)
  target_compile_options(hllm_project_options INTERFACE /W4 /permissive-)
else()
  target_compile_options(
    hllm_project_options
    INTERFACE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror
  )
endif()

if(HLLM_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "HLLM_ENABLE_SANITIZERS is not supported with MSVC")
  endif()
  target_compile_options(hllm_project_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(hllm_project_options INTERFACE -fsanitize=address,undefined)
endif()

add_subdirectory(proto)
add_subdirectory(cpp)

if(HLLM_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
