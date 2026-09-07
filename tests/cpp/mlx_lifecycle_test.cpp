#include <gtest/gtest.h>

#include <mlx/allocator.h>

#include <chrono>
#include <future>
#include <limits>

#include "device.hpp"
#include "hllm/mlx/backend.hpp"
#include "hllm/worker/control_service.hpp"
#include "model_fixture.hpp"

namespace hllm::mlx {
namespace {
TEST(MlxLifecycleTest, FreshHandlerThreadsReuseTheExecutionStream) {
  const test::ModelFixture fixture;
  const auto exercise = [&] {
    auto factory = make_backend_factory();
    auto stage = factory->load(fixture.load(0U, false), fixture.root,
                               {0U, 0U, 0U, 128U * 1024U * 1024U});
    auto state = stage->allocate_sequence(4U);
    const std::atomic_bool cancelled{false};
    static_cast<void>(stage->execute(runtime::TokenInput{{1U, 4U, 2U}}, 0U, *state, cancelled));
  };
  exercise();  // Warm framework initialization before counting persistent streams.
  const auto baseline = mx::get_streams().size();
  for (int thread = 0; thread < 8; ++thread) {
    auto call = std::async(std::launch::async, exercise);
    EXPECT_NO_THROW(call.get());
    EXPECT_EQ(mx::get_streams().size(), baseline);
  }
}

TEST(MlxLifecycleTest, RealMetalBufferLimitKeepsResourceExhaustionContract) {
  auto factory = make_backend_factory();
  std::scoped_lock lock(device_mutex());
  const auto baseline = mx::get_active_memory();
  // MetalAllocator rejects this above maxBufferLength before allocating or
  // submitting GPU work. Use its real API: zeros() can just broadcast a scalar.
  EXPECT_THROW(completed(execution_stream(), [] {
                 auto buffer = mx::allocator::malloc(std::numeric_limits<std::size_t>::max());
                 mx::allocator::free(buffer);
                 return true;
               }), std::bad_alloc);
  EXPECT_EQ(mx::get_active_memory(), baseline);
  EXPECT_EQ(completed(execution_stream(),
                      [] { return mx::sum(mx::ones({2, 2}, mx::float32)).item<float>(); }),
            4.0F);
}

TEST(MlxLifecycleTest, BusyMetricsReturnWithoutWaitingAndClearPreviousSample) {
  const test::ModelFixture fixture;
  worker::ControlService service(
      {"cpu-a", "localhost", fixture.root, 0U, 0U, 0U, 128U * 1024U * 1024U},
      make_backend_factory());
  v1::WorkerMetrics response;
  ASSERT_TRUE(service.GetMetrics(nullptr, nullptr, &response).ok());
  ASSERT_TRUE(response.has_allocator());
  std::future<grpc::Status> poll;
  bool ready = false;
  {
    std::scoped_lock lock(device_mutex());
    poll = std::async(std::launch::async,
                      [&] { return service.GetMetrics(nullptr, nullptr, &response); });
    ready = poll.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
  }  // Release before joining even when the old blocking implementation fails.
  EXPECT_TRUE(ready);
  EXPECT_TRUE(poll.get().ok());
  EXPECT_EQ(response.worker_id(), "cpu-a");
  EXPECT_FALSE(response.has_allocator());
  ASSERT_TRUE(service.GetMetrics(nullptr, nullptr, &response).ok());
  EXPECT_TRUE(response.has_allocator());
}

TEST(MlxLifecycleTest, AllocationErrorsKeepResourceExhaustionContract) {
  auto factory = make_backend_factory();
  std::scoped_lock lock(device_mutex());
  for (const auto message :
       {"[malloc] Unable to allocate 1024 bytes.", "[metal::malloc] Resource limit (10) exceeded.",
        "[metal::malloc] Attempting to allocate 100 bytes which is greater than the maximum "
        "allowed buffer size"}) {
    auto pending = mx::array(0.0F);
    EXPECT_THROW(completed(execution_stream(),
                           [&pending, message]() -> bool {
                             pending = mx::sum(mx::ones({32, 32}, mx::float32));
                             mx::async_eval(pending);
                             throw std::runtime_error(message);
                           }),
                 std::bad_alloc);
    EXPECT_EQ(pending.item<float>(), 1024.0F);
  }
  EXPECT_THROW(completed(execution_stream(),
                         []() -> bool { throw std::runtime_error("device execution failed"); }),
               std::runtime_error);
  EXPECT_THROW(completed(execution_stream(),
                         []() -> bool { throw std::invalid_argument("invalid tensor"); }),
               std::invalid_argument);
  // The synchronized stream remains usable after recoverable failures.
  EXPECT_EQ(completed(execution_stream(),
                      [] { return mx::sum(mx::ones({2, 2}, mx::float32)).item<float>(); }),
            4.0F);
}
}  // namespace
}  // namespace hllm::mlx
