#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>

#include "hllm/runtime/stage_backend.hpp"

namespace hllm::worker {

struct SchedulerMetrics {
  std::size_t queued_prefills{};
  std::size_t queued_decodes{};
  std::size_t executing{};
  std::size_t completed_steps{};
  double queue_wait_ms{};
  std::size_t decode_batches{};
  std::size_t largest_decode_batch{};
};

// RPC callers own tickets and sequence buffers until compute completes. One
// caller leads each dispatch; other batch members wait for their own result.
class StageScheduler final {
 public:
  void configure(std::size_t maximum_decode_batch);
  runtime::StageOutput execute(const runtime::StageBackend& backend, runtime::StageInput input,
                               std::size_t position, runtime::SequenceState& sequence,
                               const std::atomic_bool& cancelled,
                               std::chrono::system_clock::time_point deadline);
  SchedulerMetrics metrics() const;

 private:
  struct Ticket {
    runtime::DecodeBatchItem item;
    std::chrono::system_clock::time_point deadline;
    std::chrono::steady_clock::time_point queued;
    bool claimed{false}, done{false};
    std::optional<runtime::StageOutput> result;
    std::exception_ptr error;
  };
  Ticket* next() const;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Ticket*> prefills_, decodes_;
  std::size_t maximum_decode_batch_{1};
  std::size_t executing_{0};
  std::size_t decode_streak_{0};
  std::size_t completed_steps_{0};
  double queue_wait_ms_{0};
  std::size_t decode_batches_{0}, largest_decode_batch_{0};
};
}  // namespace hllm::worker
