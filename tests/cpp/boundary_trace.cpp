// Opt-in full-checkpoint qualification; never registered as an ordinary test.
#include <google/protobuf/json/json.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <variant>
#include <vector>
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
}  // namespace
// Export or replay the real F16 boundary for an exact teacher-forced history.
int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::invalid_argument(
        "usage: boundary-trace MODEL_ROOT LOAD_SPEC HISTORY OUTPUT");
    if (std::ifstream(argv[4]).good()) throw std::invalid_argument("output already exists");
    const auto spec = read(argv[2]);
    hllm::v1::LoadStageRequest request;
    const auto status = google::protobuf::json::JsonStringToMessage(
        spec.at("load_request").dump(), &request);
    if (!status.ok()) throw std::invalid_argument(status.ToString());
    const auto history = read(argv[3]);
    const auto& budget = spec.at("budget");
    hllm::runtime::MemoryAmounts capacity{budget.value("host", std::size_t{0}),
        budget.value("device", std::size_t{0}), 0U, budget.value("unified", std::size_t{0})};
#ifdef HLLM_PROBE_MLX
    auto factory = backend::make_backend_factory(capacity.unified_bytes);
#else
    auto factory = backend::make_backend_factory();
#endif
    auto stage = factory->load(request, argv[1], capacity);
    auto& traced = dynamic_cast<backend::ReferenceStage&>(*stage);
    const auto tokens = spec.at("capacity_tokens").get<std::size_t>();
    const auto memory = stage->sequence_memory(tokens);
    hllm::runtime::require_memory(hllm::runtime::add_memory(stage->weight_memory(),
        hllm::runtime::add_memory(memory.cache, memory.workspace)), capacity);
    auto state = stage->allocate_sequence(tokens);
    std::atomic_bool cancelled{false};
    json result{{"steps", json::array()}};
    const auto& steps = history.at("steps");
    if (steps.empty()) throw std::invalid_argument("empty history");
    for (std::size_t index = 0; index < steps.size(); ++index) {
      const auto& step = steps.at(index);
      hllm::runtime::StageInput input;
      if (step.contains("input_ids")) {
        input = hllm::runtime::TokenInput{step.at("input_ids").get<std::vector<std::uint64_t>>()};
      } else {
        auto bytes = step.at("payload").get<std::vector<unsigned char>>();
        hllm::runtime::BoundaryActivation boundary{step.at("tokens").get<std::size_t>(),
            step.at("width").get<std::size_t>(), {}};
        for (const auto byte : bytes) boundary.payload.push_back(static_cast<std::byte>(byte));
        input = std::move(boundary);
      }
      const auto position = step.at("position").get<std::size_t>();
      backend::ExecutionTrace trace;
      auto output = index + 1U == steps.size()
          ? traced.execute_traced(std::move(input), position, *state, cancelled, trace)
          : stage->execute(std::move(input), position, *state, cancelled);
      json row{{"position", position}};
      if (const auto* boundary = std::get_if<hllm::runtime::BoundaryActivation>(&output)) {
        std::vector<unsigned char> bytes;
        for (const auto byte : boundary->payload) bytes.push_back(std::to_integer<unsigned char>(byte));
        row["tokens"] = boundary->tokens;
        row["width"] = boundary->width;
        row["payload"] = bytes;
      } else {
        row["argmax"] = std::get<hllm::runtime::SampledToken>(output).id;
      }
      result["steps"].push_back(std::move(row));
      if (index + 1U == steps.size()) {
        result["last_logits"] = trace.last_logits;
        result["last_layer_rows"] = json::array();
        for (const auto& layer : trace.layers) {
          const auto begin = layer.end() - static_cast<std::ptrdiff_t>(stage->hidden_size());
          result["last_layer_rows"].push_back(std::vector<float>(begin, layer.end()));
        }
      }
    }
    std::ofstream output(argv[4]);
    output << result.dump() << '\n';
    if (!output) throw std::runtime_error("cannot write trace");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
