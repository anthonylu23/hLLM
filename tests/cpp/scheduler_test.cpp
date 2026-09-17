#include "hllm/worker/scheduler.hpp"

#include <gtest/gtest.h>

#include <future>
#include <thread>

#include "hllm/runtime/error.hpp"

namespace hllm::worker {
namespace {
using namespace std::chrono_literals;

class GateStage : public runtime::StageBackend {
 public:
  mutable std::mutex mutex;
  mutable std::vector<std::uint64_t> order;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  bool supports_decode_batch() const override { return true; }
  runtime::MemoryAmounts weight_memory() const override { return {}; }
  std::size_t hidden_size() const override { return 1U; }
  std::size_t vocabulary_size() const override { return 1000U; }
  std::size_t maximum_tokens() const override { return 1024U; }
  runtime::SequenceMemory sequence_memory(std::size_t) const override { return {}; }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t) const override {
    return std::make_unique<runtime::SequenceState>();
  }
  runtime::StageOutput execute(runtime::StageInput input, std::size_t, runtime::SequenceState&,
                               const std::atomic_bool&) const override {
    const auto id = std::get<runtime::TokenInput>(input).ids.at(0);
    if (id == 999U) gate.wait();
    {
      std::scoped_lock lock(mutex);
      order.push_back(id);
    }
    return runtime::SampledToken{id};
  }
};

template <class Predicate>
bool wait_for(Predicate predicate) {
  const auto until = std::chrono::steady_clock::now() + 2s;
  while (!predicate() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
  return predicate();
}

class StageSchedulerTest : public ::testing::TestWithParam<std::size_t> {};
TEST_P(StageSchedulerTest, BoundsDecodePriorityAndDropsCancelledQueuedWork) {
  StageScheduler scheduler;
  scheduler.configure(GetParam());
  GateStage backend;
  std::atomic_bool cancelled{false}, dropped{false};
  const auto execute = [&](std::uint64_t id, std::size_t position, const std::atomic_bool& stop) {
    runtime::SequenceState sequence;
    sequence.prefilling = id == 100U;
    return scheduler.execute(backend, runtime::TokenInput{{id}}, position, sequence, stop,
                             std::chrono::system_clock::now() + 5s);
  };
  auto first = std::async(std::launch::async, [&] { return execute(999U, 0U, cancelled); });
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().executing == 1U; }));
  std::vector<std::future<runtime::StageOutput>> decodes;
  for (std::size_t i = 0; i < 10U; ++i) {
    decodes.push_back(std::async(std::launch::async, [&, i] { return execute(i, 1U, cancelled); }));
    EXPECT_TRUE(wait_for([&] { return scheduler.metrics().queued_decodes == i + 1U; }));
  }
  auto prefill = std::async(std::launch::async, [&] { return execute(100U, 3U, cancelled); });
  auto cancelled_job = std::async(std::launch::async, [&] {
    EXPECT_THROW(static_cast<void>(execute(200U, 0U, dropped)), std::runtime_error);
  });
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().queued_prefills == 2U; }));
  dropped = true;
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().queued_prefills == 1U; }));
  backend.release.set_value();
  static_cast<void>(first.get());
  for (auto& decode : decodes) static_cast<void>(decode.get());
  static_cast<void>(prefill.get());
  cancelled_job.get();
  EXPECT_EQ(backend.order,
            (std::vector<std::uint64_t>{999U, 0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 100U, 8U, 9U}));
  const auto metrics = scheduler.metrics();
  EXPECT_EQ(metrics.queued_prefills + metrics.queued_decodes + metrics.executing, 0U);
  EXPECT_EQ(metrics.completed_steps, 12U);
  if (GetParam() > 1) {
    EXPECT_GT(metrics.decode_batches, 0U);
    EXPECT_EQ(metrics.largest_decode_batch, GetParam());
  }
}
INSTANTIATE_TEST_SUITE_P(DecodeBatchSizes, StageSchedulerTest, ::testing::Values(1U, 4U));

TEST(StageSchedulerDeadlineTest, ExpiryRetiresQueuedWorkWithoutEnteringBackend) {
  StageScheduler scheduler;
  GateStage backend;
  std::atomic_bool stop{false};
  runtime::SequenceState active, queued;
  auto first = std::async(std::launch::async, [&] {
    return scheduler.execute(backend, runtime::TokenInput{{999}}, 0, active, stop,
                             std::chrono::system_clock::now() + 5s);
  });
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().executing == 1; }));
  try {
    static_cast<void>(scheduler.execute(backend, runtime::TokenInput{{1}}, 1, queued, stop,
                                        std::chrono::system_clock::now() + 20ms));
    ADD_FAILURE() << "expired request entered the backend";
  } catch (const runtime::Error& error) {
    EXPECT_EQ(error.code(), runtime::ErrorCode::kDeadlineExceeded);
  }
  EXPECT_EQ(scheduler.metrics().queued_decodes, 0U);
  backend.release.set_value();
  static_cast<void>(first.get());
  EXPECT_EQ(backend.order, (std::vector<std::uint64_t>{999}));
}

class BatchGateStage final : public GateStage {
 public:
  mutable std::promise<void> entered;
  std::promise<void> finish;
  std::shared_future<void> batch_gate = finish.get_future().share();
  bool fail = false;
  std::vector<runtime::StageOutput> execute_decode_batch(
      std::vector<runtime::DecodeBatchItem> items) const override {
    entered.set_value();
    if (batch_gate.wait_for(5s) != std::future_status::ready)
      throw std::runtime_error("test batch gate timed out");
    if (fail) throw std::runtime_error("injected dispatch failure");
    std::vector<runtime::StageOutput> result;
    for (auto& item : items) {
      // Like accelerator dispatch, completion owns every member until compute ends.
      item.sequence->generated_index += 1;
      result.push_back(runtime::SampledToken{std::get<runtime::TokenInput>(item.input).ids[0]});
    }
    return result;
  }
};

class InFlightBatchTest : public ::testing::TestWithParam<bool> {};
TEST_P(InFlightBatchTest, RetainsCancelledMemberUntilDispatchEndsAndRecoversAfterFailure) {
  StageScheduler scheduler;
  scheduler.configure(4);
  BatchGateStage backend;
  backend.fail = GetParam();
  std::atomic_bool running{false}, cancelled{false};
  runtime::SequenceState prefill, a, b;
  const auto execute = [&](std::uint64_t id, std::size_t position, runtime::SequenceState& sequence,
                           const std::atomic_bool& stop) {
    return scheduler.execute(backend, runtime::TokenInput{{id}}, position, sequence, stop,
                             std::chrono::system_clock::now() + 5s);
  };
  auto first = std::async(std::launch::async, [&] { return execute(999, 0, prefill, running); });
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().executing == 1; }));
  auto member = std::async(std::launch::async, [&] { return execute(1, 1, a, cancelled); });
  auto peer = std::async(std::launch::async, [&] { return execute(2, 1, b, running); });
  EXPECT_TRUE(wait_for([&] { return scheduler.metrics().queued_decodes == 2; }));
  backend.release.set_value();
  EXPECT_EQ(backend.entered.get_future().wait_for(2s), std::future_status::ready);
  cancelled = true;
  EXPECT_EQ(member.wait_for(30ms), std::future_status::timeout);
  EXPECT_EQ(peer.wait_for(30ms), std::future_status::timeout);
  EXPECT_EQ(scheduler.metrics().executing, 2U);
  backend.finish.set_value();
  static_cast<void>(first.get());
  if (GetParam()) {
    EXPECT_THROW(static_cast<void>(member.get()), std::runtime_error);
    EXPECT_THROW(static_cast<void>(peer.get()), std::runtime_error);
  } else {
    EXPECT_EQ(std::get<runtime::SampledToken>(member.get()).id, 1U);
    EXPECT_EQ(std::get<runtime::SampledToken>(peer.get()).id, 2U);
    EXPECT_EQ(a.generated_index, 1U);
    EXPECT_EQ(b.generated_index, 1U);
  }
  runtime::SequenceState recovery;
  EXPECT_EQ(std::get<runtime::SampledToken>(execute(3, 1, recovery, running)).id, 3U);
  const auto metrics = scheduler.metrics();
  EXPECT_EQ(metrics.executing + metrics.queued_decodes + metrics.queued_prefills, 0U);
}
INSTANTIATE_TEST_SUITE_P(DispatchFailure, InFlightBatchTest, ::testing::Bool());
}  // namespace
}  // namespace hllm::worker
