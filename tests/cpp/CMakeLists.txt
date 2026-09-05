include(FetchContent)

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(
  hllm_runtime_common_tests
  buffer_test.cpp
  half_test.cpp
  llama_test.cpp
  safetensors_test.cpp
  tensor_envelope_test.cpp
  worker_control_test.cpp
)
target_link_libraries(
  hllm_runtime_common_tests
  PRIVATE
    hllm::cpu_backend
    hllm::runtime_common
    hllm::worker_runtime
    GTest::gtest_main
    hllm_project_options
)
gtest_discover_tests(hllm_runtime_common_tests)
