include(FetchContent)

find_package(nlohmann_json CONFIG QUIET)
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
  PUBLIC hllm::proto hllm::runtime_common hllm::cpu_backend hllm_project_options
)

add_library(hllm::worker_runtime ALIAS hllm_worker_runtime)

add_executable(hllm-worker-cpu src/worker/main.cpp)
target_link_libraries(
  hllm-worker-cpu
  PRIVATE hllm::cpu_backend hllm::worker_runtime hllm_project_options
)
