// Opt-in full-checkpoint qualification; never registered as an ordinary test.
#include <google/protobuf/json/json.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "hllm/runtime/checked_size.hpp"

#ifdef HLLM_PROBE_MLX
#include "hllm/mlx/backend.hpp"
#include "hllm/mlx/stage.hpp"
namespace backend = hllm::mlx;
#else
#include "hllm/cuda/backend.hpp"
#include "hllm/cuda/stage.hpp"
namespace backend = hllm::cuda;
#endif

using nlohmann::json;
namespace {
json read(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("cannot open probe input");
  return json::parse(input);
}
json difference(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::invalid_argument("oracle shape mismatch");
  }
  double maximum = 0.0, squared = 0.0, reference_squared = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      throw std::runtime_error("non-finite diagnostic output");
    }
    const auto error = static_cast<double>(actual[i]) - expected[i];
    maximum = std::max(maximum, std::abs(error));
    squared += error * error;
    reference_squared += static_cast<double>(expected[i]) * expected[i];
  }
  return {{"max_absolute", maximum},
          {"relative_l2", std::sqrt(squared / std::max(reference_squared, 1e-30))}};
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument("usage: checkpoint-probe MODEL_ROOT LOAD_SPEC ORACLE OUTPUT");
    const auto spec = read(argv[2]);
    hllm::v1::LoadStageRequest request;
    auto status =
        google::protobuf::json::JsonStringToMessage(spec.at("load_request").dump(), &request);
    if (!status.ok()) throw std::invalid_argument(status.ToString());
    const auto oracle = read(argv[3]);
    const auto& budget = spec.at("budget");
    hllm::runtime::MemoryAmounts capacity{budget.value("host", std::size_t{0}),
                                          budget.value("device", std::size_t{0}), 0U,
                                          budget.value("unified", std::size_t{0})};
#ifdef HLLM_PROBE_MLX
    auto factory = backend::make_backend_factory(capacity.unified_bytes);
#else
    auto factory = backend::make_backend_factory();
#endif
    auto stage = factory->load(request, argv[1], capacity);
    auto& traced = dynamic_cast<backend::ReferenceStage&>(*stage);
    const bool half = request.plan().execution_dtype() == hllm::v1::DATA_TYPE_F16;
    // Declared before measurement; applies to this explicit full-checkpoint probe.
    const double absolute_limit = half ? 0.25 : 0.01;
    const double relative_limit = half ? 0.01 : 0.001;
    json result{{"dtype", half ? "F16" : "F32"},
                {"absolute_logit_limit", absolute_limit},
                {"relative_logit_limit", relative_limit},
                {"cases", json::array()}};
    bool passed = true;
    std::atomic_bool cancelled{false};
    for (const auto& item : oracle.at("cases")) {
      std::size_t capacity_tokens = item.at("token_ids").size();
      for (const auto& step : item.at("steps")) {
        capacity_tokens = std::max(capacity_tokens,
            hllm::runtime::checked_add(step.at("position").get<std::size_t>(),
                                      step.at("input_ids").size()));
      }
      const auto memory = stage->sequence_memory(capacity_tokens);
      hllm::runtime::require_memory(
          hllm::runtime::add_memory(stage->weight_memory(),
                                    hllm::runtime::add_memory(memory.cache, memory.workspace)),
          capacity);
      auto state = stage->allocate_sequence(capacity_tokens);
      // Optional teacher-forced history for investigating later greedy divergence.
      for (const auto& step : item.value("warmup", json::array())) {
        static_cast<void>(stage->execute(
            hllm::runtime::TokenInput{step.at("input_ids").get<std::vector<std::uint64_t>>()},
            step.at("position").get<std::size_t>(), *state, cancelled));
      }
      json measured{{"prompt", item.at("prompt")}, {"steps", json::array()}};
      for (const auto& step : item.at("steps")) {
        backend::ExecutionTrace trace;
        auto output = traced.execute_traced(
            hllm::runtime::TokenInput{step.at("input_ids").get<std::vector<std::uint64_t>>()},
            step.at("position").get<std::size_t>(), *state, cancelled, trace);
        auto errors = difference(trace.last_logits, step.at("logits").get<std::vector<float>>());
        const auto token = std::get<hllm::runtime::SampledToken>(output).id;
        errors["argmax"] = token;
        errors["reference_argmax"] = step.at("argmax");
        errors["reference_margin"] = step.at("top2_margin");
        errors["position"] = step.at("position");
        double layer_relative = 0.0;
        if (trace.layers.size() != step.at("layers").size())
          throw std::runtime_error("layer count mismatch");
        for (std::size_t i = 0; i < trace.layers.size(); ++i) {
          const auto& layer = trace.layers[i];
          auto begin = layer.end() - static_cast<std::ptrdiff_t>(stage->hidden_size());
          const auto error =
              difference({begin, layer.end()}, step.at("layers").at(i).get<std::vector<float>>());
          layer_relative = std::max(layer_relative, error.at("relative_l2").get<double>());
        }
        errors["maximum_layer_relative_l2"] = layer_relative;
        errors["passed"] = errors.at("max_absolute").get<double>() <= absolute_limit &&
                           errors.at("relative_l2").get<double>() <= relative_limit &&
                           layer_relative <= (half ? 0.015 : 0.001) &&
                           token == step.at("argmax").get<std::uint64_t>();
        passed = passed && errors.at("passed").get<bool>();
        measured["steps"].push_back(std::move(errors));
      }
      result["cases"].push_back(std::move(measured));
    }
    const auto weights = stage->weight_memory();
    result["resident_weights"] = {{"host", weights.host_bytes},
                                  {"device", weights.device_bytes},
                                  {"unified", weights.unified_bytes}};
    stage.reset();
    if (auto metrics = factory->allocator_metrics()) {
      result["allocator_after_unload"] = {{"active", metrics->active_bytes},
                                          {"cached", metrics->cached_bytes},
                                          {"peak", metrics->peak_bytes}};
    }
    result["passed"] = passed;
    std::ofstream output(argv[4]);
    output << result.dump(2) << '\n';
    if (!output) throw std::runtime_error("failed to write probe result");
    std::cout << result.dump() << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
