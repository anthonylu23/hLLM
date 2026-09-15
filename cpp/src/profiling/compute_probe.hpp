#pragma once

#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <string>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::profiling {
using nlohmann::json;
namespace rt = hllm::runtime;
inline bool identical(const rt::StageOutput& a, const rt::StageOutput& b) {
  if (a.index() != b.index()) return false;
  if (const auto* token = std::get_if<rt::SampledToken>(&a))
    return token->id == std::get<rt::SampledToken>(b).id;
  const auto& x = std::get<rt::BoundaryActivation>(a);
  const auto& y = std::get<rt::BoundaryActivation>(b);
  return x.tokens == y.tokens && x.width == y.width && x.payload == y.payload;
}
inline void compute_probe(rt::StageBackend& stage, const rt::MemoryAmounts& capacity,
                          bool first, std::size_t tokens, std::size_t prompt,
                          std::size_t steps, std::size_t cycles,
                          const std::function<void(const json&)>& emit) {
  const auto reservation = stage.sequence_memory(tokens);
  rt::require_memory(rt::add_memory(stage.weight_memory(),
      rt::add_memory(reservation.cache, reservation.workspace)), capacity);
  std::atomic_bool cancelled{false};
  for (std::size_t cycle = 0U; cycle < cycles; ++cycle) {
    // Alternate ordering to expose warm-cache/frequency bias in paired comparisons.
    const std::array<bool, 2> order = cycle % 2U ? std::array{true, false} : std::array{false, true};
    std::vector<rt::StageOutput> reference;
    for (const auto instrumented : order) {
      std::vector<json> records;
      const auto record = [&](const char* component, std::optional<std::size_t> layer,
                              std::size_t step, std::size_t position, double ms) {
        if (!std::isfinite(ms) || ms < 0.0) throw std::runtime_error("invalid native timing");
        records.push_back({{"event", "timing"}, {"cycle", cycle}, {"step", step},
          {"context_tokens", position}, {"phase", step == 0U ? "prefill" : "decode"},
          {"component", component}, {"layer_index", layer ? json(*layer) : json(nullptr)},
          {"timing_mode", instrumented ? "component-synchronized" : "whole-stage"},
          {"milliseconds", ms}});
      };
      auto start = rt::ProfileClock::now();
      auto state = stage.allocate_sequence(tokens);
      record("sequence_allocation", std::nullopt, 0U, 0U, rt::elapsed_ms(start));
      for (std::size_t step = 0U; step < steps; ++step) {
        const auto count = step == 0U ? prompt : 1U;
        const auto position = step == 0U ? 0U : prompt + step - 1U;
        rt::StageInput input;
        if (first) input = rt::TokenInput{std::vector<std::uint64_t>(count, 0U)};
        else input = rt::BoundaryActivation{count, stage.hidden_size(), std::vector<std::byte>(
            rt::checked_multiply(rt::checked_multiply(count, stage.hidden_size()), 2U))};
        rt::ExecutionTiming timing;
        start = rt::ProfileClock::now();
        auto output = instrumented
            ? stage.execute_profiled(std::move(input), position, *state, cancelled, timing)
            : stage.execute(std::move(input), position, *state, cancelled);
        const auto total = rt::elapsed_ms(start);
        record("stage", std::nullopt, step, position, total);
        double attributed = 0.0;
        for (const auto& item : timing.components) {
          record(item.component.c_str(), item.layer_index, step, position, item.milliseconds);
          attributed += item.milliseconds;
        }
        if (instrumented) {
          if (attributed > total + 0.001) throw std::runtime_error("timing regions overlap");
          record("runtime_overhead", std::nullopt, step, position, std::max(0.0, total - attributed));
        }
        if (reference.size() == steps) {
          if (!identical(output, reference.at(step)))
            throw std::runtime_error("profiling changed stage output");
        } else reference.push_back(std::move(output));
      }
      state.reset();
      // No JSON, disk I/O or memory RPCs in the timed execute calls.
      for (const auto& item : records) emit(item);
    }
    emit({{"event", "pair_complete"}, {"cycle", cycle}, {"outputs_identical", true}});
  }
}
}  // namespace hllm::profiling
