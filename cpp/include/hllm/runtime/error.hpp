#pragma once

#include <stdexcept>
#include <string>

namespace hllm::runtime {

// Failure categories the worker services translate into wire error codes and
// gRPC statuses. Anything thrown without a category is reported as internal.
enum class ErrorCode {
  kInvalidRequest,      // the caller's request violates the protocol or model limits
  kIncompatibleWorker,  // this worker cannot load or execute the described stage
  kResourceExhausted,   // a memory budget or transport limit would be exceeded
  kDeadlineExceeded,    // the request's deadline has already passed
  kInternal,            // an invariant the worker relies on was violated
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

  static Error invalid_request(const std::string& message) {
    return {ErrorCode::kInvalidRequest, message};
  }
  static Error incompatible_worker(const std::string& message) {
    return {ErrorCode::kIncompatibleWorker, message};
  }
  static Error resource_exhausted(const std::string& message) {
    return {ErrorCode::kResourceExhausted, message};
  }
  static Error deadline_exceeded(const std::string& message) {
    return {ErrorCode::kDeadlineExceeded, message};
  }
  static Error internal(const std::string& message) { return {ErrorCode::kInternal, message}; }

 private:
  ErrorCode code_;
};

}  // namespace hllm::runtime
