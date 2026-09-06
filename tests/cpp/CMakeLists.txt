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
  backend_contract_test.cpp
  buffer_test.cpp
  half_test.cpp
  llama_test.cpp
  qwen3_test.cpp
  stage_test.cpp
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
    nlohmann_json::nlohmann_json
    GTest::gtest_main
    hllm_project_options
)
gtest_discover_tests(hllm_runtime_common_tests)

target_compile_definitions(hllm_runtime_common_tests PRIVATE
  HLLM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures"
)

find_program(HLLM_UV_EXECUTABLE uv REQUIRED)
add_test(NAME CpuPipelineIntegration
  COMMAND ${CMAKE_COMMAND} -E env HLLM_CPU_WORKER=$<TARGET_FILE:hllm-worker-cpu>
    ${HLLM_UV_EXECUTABLE} run pytest tests/integration -q
)
set_tests_properties(CpuPipelineIntegration PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  TIMEOUT 120
)

if(HLLM_ENABLE_CUDA)
  add_executable(hllm_cuda_tests cuda_stage_test.cpp)
  target_link_libraries(hllm_cuda_tests PRIVATE hllm::cuda_backend hllm::cpu_backend hllm::worker_runtime
    nlohmann_json::nlohmann_json GTest::gtest_main hllm_project_options)
  target_compile_definitions(hllm_cuda_tests PRIVATE HLLM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
  add_test(NAME CudaNumericalParity COMMAND hllm_cuda_tests)
  set_tests_properties(CudaNumericalParity PROPERTIES TIMEOUT 180 LABELS "cuda;hardware")
  add_test(NAME MixedPipelineIntegration
    COMMAND ${CMAKE_COMMAND} -E env HLLM_CPU_WORKER=$<TARGET_FILE:hllm-worker-cpu>
      HLLM_CUDA_WORKER=$<TARGET_FILE:hllm-worker-cuda>
      ${HLLM_UV_EXECUTABLE} run pytest tests/cuda/test_mixed.py -q)
  set_tests_properties(MixedPipelineIntegration PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 240 LABELS "cuda;hardware")
  add_test(NAME CudaBackendSmoke
    COMMAND ${CMAKE_COMMAND} -E env HLLM_CUDA_WORKER=$<TARGET_FILE:hllm-worker-cuda>
      ${HLLM_UV_EXECUTABLE} run pytest tests/cuda/test_worker.py -q)
  set_tests_properties(CudaBackendSmoke PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    TIMEOUT 120
    LABELS "cuda;hardware")
endif()
