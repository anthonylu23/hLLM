include(FetchContent)

find_package(nlohmann_json CONFIG QUIET GLOBAL)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
  )
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(
  hllm_runtime_common
  src/buffer.cpp
  src/half.cpp
  src/safetensors.cpp
  src/tensor_envelope.cpp
)

target_include_directories(
  hllm_runtime_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  hllm_runtime_common
  PUBLIC hllm_project_options
  PRIVATE nlohmann_json::nlohmann_json
)

add_library(hllm::runtime_common ALIAS hllm_runtime_common)

add_library(
  hllm_cpu_backend
  src/cpu/llama.cpp
  src/cpu/tensor.cpp
  src/cpu/stage.cpp
)
target_include_directories(
  hllm_cpu_backend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(hllm_cpu_backend PUBLIC hllm::runtime_common hllm::proto hllm_project_options)

add_library(hllm::cpu_backend ALIAS hllm_cpu_backend)

add_library(hllm_worker_runtime src/worker/control_service.cpp src/worker/execution_service.cpp)
target_include_directories(
  hllm_worker_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  hllm_worker_runtime
  PUBLIC hllm::proto hllm::runtime_common hllm_project_options
)

add_library(hllm::worker_runtime ALIAS hllm_worker_runtime)

add_executable(hllm-worker-cpu src/worker/main.cpp)
target_link_libraries(
  hllm-worker-cpu
  PRIVATE hllm::cpu_backend hllm::worker_runtime hllm_project_options
)

if(HLLM_ENABLE_CUDA)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "HLLM_ENABLE_CUDA currently supports Linux only")
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_package(Torch CONFIG REQUIRED)
  if(NOT TARGET torch_cuda)
    message(FATAL_ERROR "HLLM_ENABLE_CUDA requires a CUDA-enabled LibTorch distribution")
  endif()
  # The common runtime and system gRPC/Protobuf use the modern GNU C++ ABI.
  # Mixing pre-C++11 LibTorch builds across this boundary is unsupported.
  if(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
    message(FATAL_ERROR "Use a cxx11-ABI LibTorch build with ABI-compatible gRPC/Protobuf")
  endif()
  add_library(hllm_cuda_backend src/cuda/backend.cpp)
  target_include_directories(hllm_cuda_backend PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  separate_arguments(HLLM_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  # LibTorch can ship its own Protobuf headers. Keep its include directories
  # out of RPC-facing translation units; exchange only our plain C++ types.
  add_library(hllm_cuda_device src/cuda/device.cpp)
  target_compile_options(hllm_cuda_device PRIVATE ${HLLM_TORCH_CXX_FLAGS})
  target_link_libraries(hllm_cuda_device PRIVATE hllm_project_options ${TORCH_LIBRARIES} CUDA::cudart)
  target_link_libraries(hllm_cuda_backend
    PUBLIC hllm::runtime_common hllm::proto hllm_project_options
    PRIVATE hllm_cuda_device)
  add_library(hllm::cuda_backend ALIAS hllm_cuda_backend)
  add_executable(hllm-worker-cuda src/worker/main.cpp)
  target_compile_definitions(hllm-worker-cuda PRIVATE HLLM_WORKER_CUDA=1)
  target_link_libraries(hllm-worker-cuda PRIVATE hllm::cuda_backend hllm::worker_runtime)
endif()
