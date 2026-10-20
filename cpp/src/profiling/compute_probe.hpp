#pragma once

#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/stage_backend.hpp"

namespace hllm::profiling {
using nlohmann::json;
namespace rt = hllm::runtime;
inline bool identical(const rt::StageOutput& a, const rt::StageOutput& b) {
  if (a.index() != b.index()) return false;
  if (std::holds_alternative<rt::PrefillProgress>(a)) return true;
  if (const auto* token = std::get_if<rt::SampledToken>(&a))
    return token->id == std::get<rt::SampledToken>(b).id;
  const auto& x = std::get<rt::BoundaryActivation>(a);
  const auto& y = std::get<rt::BoundaryActivation>(b);
  return x.tokens == y.tokens && x.width == y.width && x.payload == y.payload;
}
inline void compute_probe(rt::StageBackend& stage, const rt::MemoryAmounts& capacity,
                          bool first, std::size_t tokens, std::size_t prompt,
                          std::size_t steps, std::size_t cycles, std::size_t chunk,
                          const std::function<void(const json&)>& emit) {
  // Serving's prefill: `chunk` tokens per input (0 is the whole prompt), each a
  // separate execution with the sequence flags serving sets.
  const auto reservation = stage.sequence_memory(tokens, chunk);
  rt::require_memory(rt::add_memory(stage.weight_memory(),
      rt::add_memory(reservation.cache, reservation.workspace)), capacity);
  std::atomic_bool cancelled{false};
  for (std::size_t cycle = 0U; cycle < cycles; ++cycle) {
    // Alternate ordering to expose warm-cache/frequency bias in paired comparisons.
    const std::array<bool, 2> order = cycle % 2U ? std::array{true, false} : std::array{false, true};
    std::vector<rt::StageOutput> reference;
    for (const auto instrumented : order) {
      std::vector<json> records;
      // Chunk indices appear only on chunked prefill records, so whole-prompt records
      // keep their historical shape.
      const auto record = [&](const char* component, std::optional<std::size_t> layer,
                              std::size_t step, std::size_t position, double ms,
                              std::optional<std::size_t> chunk_index = std::nullopt) {
        if (!std::isfinite(ms) || ms < 0.0) throw std::runtime_error("invalid native timing");
        json item{{"event", "timing"}, {"cycle", cycle}, {"step", step},
          {"context_tokens", position}, {"phase", step == 0U ? "prefill" : "decode"},
          {"component", component}, {"layer_index", layer ? json(*layer) : json(nullptr)},
          {"timing_mode", instrumented ? "component-synchronized" : "whole-stage"},
          {"milliseconds", ms}};
        if (chunk_index) item["chunk_index"] = *chunk_index;
        records.push_back(std::move(item));
      };
      auto start = rt::ProfileClock::now();
      auto state = stage.allocate_sequence(tokens);
      const std::optional<std::size_t> first_chunk =
          chunk ? std::optional<std::size_t>(0U) : std::nullopt;
      record("sequence_allocation", std::nullopt, 0U, 0U, rt::elapsed_ms(start), first_chunk);
      // One execution per prefill chunk, then one per decode step.
      std::vector<std::pair<std::size_t, std::size_t>> executions;  // (step, position)
      for (std::size_t position = 0U; position < prompt;
           position += std::min(chunk ? chunk : prompt, prompt - position)) {
        executions.emplace_back(0U, position);
      }
      for (std::size_t step = 1U; step < steps; ++step)
        executions.emplace_back(step, prompt + step - 1U);
      for (std::size_t index = 0U; index < executions.size(); ++index) {
        const auto [step, position] = executions[index];
        const auto count = step == 0U ? std::min(chunk ? chunk : prompt, prompt - position) : 1U;
        const std::optional<std::size_t> chunk_index =
            chunk && step == 0U ? std::optional<std::size_t>(index) : std::nullopt;
        state->prefilling = step == 0U;
        state->prefill_only = step == 0U && position + count < prompt;
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
        record("stage", std::nullopt, step, position, total, chunk_index);
        double attributed = 0.0;
        for (const auto& item : timing.components) {
          record(item.component.c_str(), item.layer_index, step, position, item.milliseconds,
                 chunk_index);
          attributed += item.milliseconds;
        }
        if (instrumented) {
          if (attributed > total + 0.001) throw std::runtime_error("timing regions overlap");
          record("runtime_overhead", std::nullopt, step, position,
                 std::max(0.0, total - attributed), chunk_index);
        }
        if (reference.size() == executions.size()) {
          if (!identical(output, reference.at(index)))
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
