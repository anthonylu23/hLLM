#include "hllm/worker/scheduler.hpp"

#include <algorithm>

#include "hllm/runtime/error.hpp"

namespace hllm::worker {
void StageScheduler::configure(std::size_t maximum_decode_batch) {
  if (maximum_decode_batch == 0 || maximum_decode_batch > 8) {
    throw runtime::Error::invalid_request("decode batch size must be between 1 and 8");
  }
  std::scoped_lock lock(mutex_);
  if (executing_ || !prefills_.empty() || !decodes_.empty()) {
    throw runtime::Error::internal("cannot reconfigure an active scheduler");
  }
  maximum_decode_batch_ = maximum_decode_batch;
}
StageScheduler::Ticket* StageScheduler::next() const {
  if (!prefills_.empty() && (decodes_.empty() || decode_streak_ >= 8U)) return prefills_.front();
  if (!decodes_.empty()) return decodes_.front();
  return nullptr;
}

runtime::StageOutput StageScheduler::execute(const runtime::StageBackend& backend,
                                             runtime::StageInput input, std::size_t position,
                                             runtime::SequenceState& sequence,
                                             const std::atomic_bool& cancelled,
                                             std::chrono::system_clock::time_point deadline) {
  Ticket ticket{{std::move(input), position, &sequence, &cancelled},
                deadline,
                std::chrono::steady_clock::now(),
                false,
                false,
                {},
                {}};
  std::unique_lock lock(mutex_);
  const bool prefill = position == 0 || sequence.prefilling;
  auto& queue = prefill ? prefills_ : decodes_;
  queue.push_back(&ticket);
  while (!ticket.done) {
    if (!ticket.claimed && (cancelled.load() || std::chrono::system_clock::now() >= deadline)) {
      queue.erase(std::find(queue.begin(), queue.end(), &ticket));
      ready_.notify_all();
      if (std::chrono::system_clock::now() >= deadline) {
        throw runtime::Error::deadline_exceeded("deadline expired in native queue");
      }
      throw runtime::Error::cancelled("request cancelled in native queue");
    }
    if (ticket.claimed || executing_ || next() != &ticket) {
      ready_.wait_for(lock, std::chrono::milliseconds(5));
      continue;
    }
    std::vector<Ticket*> dispatch;
    auto limit = prefill || !backend.supports_decode_batch() ? 1U : maximum_decode_batch_;
    if (!prefill && !prefills_.empty()) limit = std::min(limit, 8U - decode_streak_);
    while (!queue.empty() && dispatch.size() < limit) {
      auto* member = queue.front();
      queue.pop_front();
      member->claimed = true;
      if (member->item.cancelled->load() || std::chrono::system_clock::now() >= member->deadline) {
        member->error = std::make_exception_ptr(runtime::Error::cancelled("queued request ended"));
        member->done = true;
        continue;
      }
      dispatch.push_back(member);
      queue_wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  member->queued)
                            .count();
    }
    if (dispatch.empty()) {
      ready_.notify_all();
      continue;
    }
    executing_ = dispatch.size();
    decode_streak_ = prefill ? 0U : std::min(decode_streak_ + dispatch.size(), std::size_t{8});
    lock.unlock();
    std::vector<runtime::StageOutput> outputs;
    std::exception_ptr error;
    try {
      if (dispatch.size() == 1) {
        auto& item = dispatch.front()->item;
        outputs.push_back(
            backend.execute(std::move(item.input), item.position, *item.sequence, *item.cancelled));
      } else {
        std::vector<runtime::DecodeBatchItem> items;
        for (auto* member : dispatch) items.push_back(std::move(member->item));
        outputs = backend.execute_decode_batch(std::move(items));
        if (outputs.size() != dispatch.size())
          throw runtime::Error::internal("batch output count mismatch");
      }
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    for (std::size_t i = 0; i < dispatch.size(); ++i) {
      auto* member = dispatch[i];
      member->error = error;
      if (!error) member->result = std::move(outputs[i]);
      member->done = true;
    }
    completed_steps_ += dispatch.size();
    if (dispatch.size() > 1) {
      ++decode_batches_;
      largest_decode_batch_ = std::max(largest_decode_batch_, dispatch.size());
    }
    executing_ = 0;
    ready_.notify_all();
  }
  if (ticket.error) std::rethrow_exception(ticket.error);
  return std::move(*ticket.result);
}
SchedulerMetrics StageScheduler::metrics() const {
  std::scoped_lock lock(mutex_);
  return {prefills_.size(), decodes_.size(), executing_,           completed_steps_,
          queue_wait_ms_,   decode_batches_, largest_decode_batch_};
}
}  // namespace hllm::worker
