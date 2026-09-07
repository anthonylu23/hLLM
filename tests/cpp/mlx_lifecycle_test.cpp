#include <gtest/gtest.h>

#include "device.hpp"
#include "hllm/mlx/backend.hpp"

namespace hllm::mlx {
namespace {
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
