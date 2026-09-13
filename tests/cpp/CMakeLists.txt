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
  downstream_channel_test.cpp
  generation_deadline_test.cpp
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
  find_package(CUDAToolkit REQUIRED)
  add_executable(hllm_cuda_transfer_tests cuda_transfer_test.cpp)
  target_include_directories(hllm_cuda_transfer_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src/cuda")
  target_link_libraries(hllm_cuda_transfer_tests PRIVATE CUDA::cudart GTest::gtest_main hllm_project_options)
  add_test(NAME CudaTransfers COMMAND hllm_cuda_transfer_tests)
  set_tests_properties(CudaTransfers PROPERTIES TIMEOUT 60 LABELS "cuda;hardware")
  find_package(Torch CONFIG REQUIRED)
  add_library(hllm_cuda_diagnostics STATIC cuda_diagnostics.cpp)
  target_include_directories(hllm_cuda_diagnostics PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src/cuda")
  target_link_libraries(hllm_cuda_diagnostics PRIVATE ${TORCH_LIBRARIES} CUDA::cudart hllm_project_options)
  add_executable(hllm_cuda_tests cuda_stage_test.cpp cuda_lifecycle_test.cpp)
  target_link_libraries(hllm_cuda_tests PRIVATE hllm_cuda_diagnostics hllm::cuda_backend hllm::cpu_backend hllm::worker_runtime
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
  add_test(NAME MixedFailureQualification
    COMMAND ${CMAKE_COMMAND} -E env HLLM_CPU_WORKER=$<TARGET_FILE:hllm-worker-cpu>
      HLLM_CUDA_WORKER=$<TARGET_FILE:hllm-worker-cuda>
      ${HLLM_UV_EXECUTABLE} run pytest tests/cuda/test_failures.py -q)
  set_tests_properties(MixedFailureQualification PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 600 LABELS "cuda;hardware")
  add_test(NAME CudaBackendSmoke
    COMMAND ${CMAKE_COMMAND} -E env HLLM_CUDA_WORKER=$<TARGET_FILE:hllm-worker-cuda>
      ${HLLM_UV_EXECUTABLE} run pytest tests/cuda/test_worker.py -q)
  set_tests_properties(CudaBackendSmoke PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    TIMEOUT 120
    LABELS "cuda;hardware")
endif()

if(HLLM_ENABLE_MLX)
  add_executable(hllm_mlx_tests mlx_stage_test.cpp mlx_lifecycle_test.cpp)
  target_include_directories(hllm_mlx_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src/mlx")
  target_link_libraries(hllm_mlx_tests PRIVATE mlx)
  target_link_libraries(hllm_mlx_tests PRIVATE hllm::mlx_backend hllm::cpu_backend hllm::worker_runtime
    nlohmann_json::nlohmann_json GTest::gtest_main hllm_project_options)
  target_compile_definitions(hllm_mlx_tests PRIVATE HLLM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures")
  add_test(NAME MlxNumericalParity COMMAND hllm_mlx_tests)
  set_tests_properties(MlxNumericalParity PROPERTIES TIMEOUT 180 LABELS "mlx;hardware")
  add_test(NAME MlxPipelineIntegration
    COMMAND ${CMAKE_COMMAND} -E env HLLM_CPU_WORKER=$<TARGET_FILE:hllm-worker-cpu>
      HLLM_MLX_WORKER=$<TARGET_FILE:hllm-worker-mlx>
      ${HLLM_UV_EXECUTABLE} run pytest tests/mlx -q)
  set_tests_properties(MlxPipelineIntegration PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 600 LABELS "mlx;hardware")
endif()

# Manual full-checkpoint probes use the diagnostic stage API and an external oracle.
foreach(backend IN ITEMS mlx cuda)
  string(TOUPPER "${backend}" backend_upper)
  if(HLLM_ENABLE_${backend_upper})
    add_executable(hllm-checkpoint-${backend} checkpoint_probe.cpp)
    target_link_libraries(hllm-checkpoint-${backend} PRIVATE hllm::${backend}_backend
      nlohmann_json::nlohmann_json hllm_project_options)
    if(backend STREQUAL "mlx")
      target_compile_definitions(hllm-checkpoint-${backend} PRIVATE HLLM_PROBE_MLX=1)
    endif()
  endif()
endforeach()

# Separate opt-in diagnostic; never changes the serving worker binary.
foreach(backend IN ITEMS mlx cuda)
  if(TARGET hllm-checkpoint-${backend})
    add_executable(hllm-boundary-trace-${backend} boundary_trace.cpp)
    target_link_libraries(hllm-boundary-trace-${backend} PRIVATE hllm::${backend}_backend
      nlohmann_json::nlohmann_json hllm_project_options)
    if(backend STREQUAL "mlx")
      target_compile_definitions(hllm-boundary-trace-${backend} PRIVATE HLLM_PROBE_MLX=1)
    endif()
  endif()
endforeach()

foreach(backend IN ITEMS cpu mlx cuda)
  if(TARGET hllm-profile-memory-${backend})
    add_test(NAME MemoryProfile-${backend}
      COMMAND ${CMAKE_COMMAND} -E env
        HLLM_MEMORY_PROFILER=$<TARGET_FILE:hllm-profile-memory-${backend}>
        HLLM_PROFILE_BACKEND=${backend}
        ${HLLM_UV_EXECUTABLE} run pytest tests/profiling -q)
    set_tests_properties(MemoryProfile-${backend} PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 180)
  endif()
endforeach()

foreach(backend IN ITEMS cpu mlx cuda)
  if(TARGET hllm-profile-compute-${backend})
    add_test(NAME ComputeProfile-${backend}
      COMMAND ${CMAKE_COMMAND} -E env
        HLLM_COMPUTE_PROFILER=$<TARGET_FILE:hllm-profile-compute-${backend}>
        HLLM_PROFILE_BACKEND=${backend}
        ${HLLM_UV_EXECUTABLE} run pytest tests/profiling/test_compute_process.py -q)
    set_tests_properties(ComputeProfile-${backend} PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 180)
  endif()
endforeach()

add_test(NAME LinkProfile
  COMMAND ${CMAKE_COMMAND} -E env HLLM_LINK_PROFILER=$<TARGET_FILE:hllm-profile-link>
    ${HLLM_UV_EXECUTABLE} run pytest tests/profiling/test_link_process.py -q)
set_tests_properties(LinkProfile PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 90)

add_test(NAME MeasuredPlannerIntegration
  COMMAND ${CMAKE_COMMAND} -E env
    HLLM_CPU_WORKER=$<TARGET_FILE:hllm-worker-cpu>
    HLLM_MEMORY_PROFILER=$<TARGET_FILE:hllm-profile-memory-cpu>
    ${HLLM_UV_EXECUTABLE} run pytest tests/profiling/test_measured_process.py -q)
set_tests_properties(MeasuredPlannerIntegration PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 90)
