find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(HLLM_PROTO_FILES
    common.proto
    control.proto
    execution.proto
    model.proto
    placement.proto
    profile.proto
    telemetry.proto
)

set(HLLM_PROTO_SOURCES)
set(HLLM_PROTO_HEADERS)
foreach(PROTO_FILE IN LISTS HLLM_PROTO_FILES)
  get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)
  list(
    APPEND HLLM_PROTO_SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.cc"
  )
  list(
    APPEND HLLM_PROTO_HEADERS
    "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.h"
  )
  add_custom_command(
    OUTPUT
      "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.cc"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.h"
    COMMAND protobuf::protoc
    ARGS
      "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}"
      "--cpp_out=${CMAKE_CURRENT_BINARY_DIR}"
      "--grpc_out=${CMAKE_CURRENT_BINARY_DIR}"
      "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}" protobuf::protoc gRPC::grpc_cpp_plugin
    VERBATIM
  )
endforeach()

add_library(hllm_proto ${HLLM_PROTO_SOURCES} ${HLLM_PROTO_HEADERS})
target_include_directories(hllm_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(
  hllm_proto
  PUBLIC protobuf::libprotobuf gRPC::grpc++
)
target_compile_features(hllm_proto PUBLIC cxx_std_20)

add_library(hllm::proto ALIAS hllm_proto)
