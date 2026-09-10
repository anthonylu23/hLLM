#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hllm::runtime {
struct ComponentTiming {
  std::string component;
  std::optional<std::size_t> layer_index;
  double milliseconds;
};
struct ExecutionTiming {
  std::vector<ComponentTiming> components;
};
using ProfileClock = std::chrono::steady_clock;
inline double elapsed_ms(ProfileClock::time_point start) {
  return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}
// Null collectors do not read clocks, synchronize devices or allocate records.
// Call only after materializing lazy outputs on backends that defer evaluation.
template <class Synchronize>
class PhaseTimer {
 public:
  PhaseTimer(ExecutionTiming* timing, Synchronize synchronize)
      : timing_(timing), synchronize_(std::move(synchronize)) { restart(); }
  void restart() {
    if (timing_) { synchronize_(); start_ = ProfileClock::now(); }
  }
  void mark(const char* component, std::optional<std::size_t> layer = std::nullopt) {
    if (!timing_) return;
    synchronize_();
    const auto milliseconds = elapsed_ms(start_);
    timing_->components.push_back({component, layer, milliseconds});
    start_ = ProfileClock::now();
  }
 private:
  ExecutionTiming* timing_;
  Synchronize synchronize_;
  ProfileClock::time_point start_{};
};
}  // namespace hllm::runtime
