// Isolated executable: allocation faults must not affect the other test suites.
#include "hllm/worker/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

namespace {
using namespace std::chrono_literals;
thread_local bool inject = false;
std::atomic_int countdown{-1};
std::atomic_bool pointer_pair_only{false};
std::atomic_int faults{0};

void require(bool value, const char* message) {
  if (!value) {
    std::fprintf(stderr, "%s\n", message);
    // A broken scheduler can leave joinable callers waiting forever.
    std::_Exit(1);
  }
}
template <class Predicate>
void wait_for(Predicate predicate) {
  const auto until = std::chrono::steady_clock::now() + 2s;
  while (!predicate() && std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(1ms);
  require(predicate(), "scheduler failed to retire callers or reach the test gate");
}
class GateBackend final : public hllm::runtime::StageBackend {
 public:
  mutable std::atomic_bool entered{false}, proceed{false};
  hllm::runtime::MemoryAmounts weight_memory() const override { return {}; }
  std::size_t hidden_size() const override { return 1; }
  std::size_t vocabulary_size() const override { return 1; }
  std::size_t maximum_tokens() const override { return 10; }
  hllm::runtime::SequenceMemory sequence_memory(std::size_t) const override { return {}; }
  std::unique_ptr<hllm::runtime::SequenceState> allocate_sequence(std::size_t) const override {
    return std::make_unique<hllm::runtime::SequenceState>();
  }
  bool supports_decode_batch() const override { return true; }
  hllm::runtime::StageOutput execute(hllm::runtime::StageInput, std::size_t position,
                                    hllm::runtime::SequenceState&,
                                    const std::atomic_bool&) const override {
    if (position == 0) {
      entered = true;
      while (!proceed) std::this_thread::sleep_for(1ms);
    }
    return hllm::runtime::SampledToken{};
  }
};

void exercise(int fail_at, bool pair_only) {
  hllm::worker::StageScheduler scheduler;
  scheduler.configure(4);
  GateBackend backend;
  hllm::runtime::SequenceState states[3];
  std::atomic_bool cancelled{false};
  std::atomic_int completed{0}, errors{0};
  const auto run = [&](int i) {
    inject = i != 0;
    try {
      static_cast<void>(scheduler.execute(
          backend, hllm::runtime::TokenInput{}, i == 0 ? 0 : 1, states[i], cancelled,
          std::chrono::system_clock::now() + 5s));
    } catch (const std::bad_alloc&) {
      ++errors;
    } catch (...) {
      require(false, "unexpected scheduler error");
    }
    inject = false;
    ++completed;
  };
  std::thread first(run, 0);
  wait_for([&] { return backend.entered.load(); });
  std::thread member(run, 1), peer(run, 2);
  wait_for([&] { return scheduler.metrics().queued_decodes == 2; });
  faults = 0;
  pointer_pair_only = pair_only;
  countdown = fail_at;
  backend.proceed = true;
  wait_for([&] { return completed == 3; });
  countdown = -1;
  first.join();
  member.join();
  peer.join();
  require(errors == (faults == 0 ? 0 : 2), "dispatch failure did not reach both claimed members");
  if (!pair_only && fail_at == 0) require(faults == 1, "allocation injection did not fire");
  const auto metrics = scheduler.metrics();
  require(metrics.queued_decodes + metrics.queued_prefills + metrics.executing == 0,
          "scheduler retained work after an allocation failure");
  static_cast<void>(scheduler.execute(backend, hllm::runtime::TokenInput{}, 1, states[0], cancelled,
                                      std::chrono::system_clock::now() + 1s));
}
}  // namespace

void* operator new(std::size_t bytes) {
  if (inject && (!pointer_pair_only || bytes == 2 * sizeof(void*))) {
    auto remaining = countdown.load();
    while (remaining >= 0) {
      if (countdown.compare_exchange_weak(remaining, remaining - 1)) {
        if (remaining == 0) {
          ++faults;
          throw std::bad_alloc();
        }
        break;
      }
    }
  }
  if (void* result = std::malloc(bytes == 0 ? 1 : bytes)) return result;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
  // Original regression: allocation for the second dispatch pointer after claim.
  exercise(0, true);
  // Cover allocations while preparing inputs, computing, and collecting outputs.
  for (int index = 0; index < 6; ++index) exercise(index, false);
  std::puts("allocation failures retired all callers; subsequent dispatches succeeded");
}
