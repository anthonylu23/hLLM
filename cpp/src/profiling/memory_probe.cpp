// An opt-in, single-threaded process. Never reset peaks in a serving worker.
#include <google/protobuf/util/json_util.h>
#include <nlohmann/json.hpp>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifdef HLLM_WORKER_CUDA
#include "hllm/cuda/backend.hpp"
namespace backend = hllm::cuda;
#elif defined(HLLM_WORKER_MLX)
#include "hllm/mlx/backend.hpp"
namespace backend = hllm::mlx;
#else
#include "hllm/cpu/stage.hpp"
namespace backend = hllm::cpu;
#endif
#include "hllm/runtime/checked_size.hpp"
#include "compute_probe.hpp"

using nlohmann::json;
namespace rt = hllm::runtime;
namespace {
json amounts(const rt::MemoryAmounts& a) {
  return {{"host", a.host_bytes}, {"device", a.device_bytes},
          {"pinned", a.pinned_host_bytes}, {"unified", a.unified_bytes}};
}
std::optional<std::size_t> rss() {
#ifdef __APPLE__
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
    return static_cast<std::size_t>(info.resident_size);
#else
  std::ifstream stat("/proc/self/statm");
  std::size_t total = 0U, resident = 0U;
  const auto page = sysconf(_SC_PAGESIZE);
  if (stat >> total >> resident && page > 0)
    return rt::checked_multiply(resident, static_cast<std::size_t>(page));
#endif
  return std::nullopt;
}
std::optional<std::size_t> rss_peak() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return std::nullopt;
#ifdef __APPLE__
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return rt::checked_multiply(static_cast<std::size_t>(usage.ru_maxrss), 1024U);
#endif
}
json optional_bytes(std::optional<std::size_t> value) {
  return value ? json(*value) : json(nullptr);
}
std::size_t number(const json& object, const char* key, std::size_t maximum) {
  const auto& value = object.at(key);
  if (!value.is_number_unsigned() || value.get<std::uint64_t>() > maximum)
    throw std::invalid_argument(std::string("invalid unsigned value: ") + key);
  return value.get<std::size_t>();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: hllm-profile-{memory,compute}-{cpu,mlx,cuda} MODEL_ROOT SPEC_JSON OUTPUT_JSONL\n";
    return 2;
  }
  if (std::filesystem::exists(argv[3])) {
    std::cerr << "refusing to overwrite a profile result\n";
    return 2;
  }
  std::ofstream output(argv[3]);
  if (!output) return 2;
  const auto emit = [&](const json& item) {
    output << item.dump() << '\n';
    output.flush();
    if (!output) throw std::runtime_error("cannot write profile sample");
  };
  std::unique_ptr<rt::BackendFactory> factory;
  std::unique_ptr<rt::StageBackend> stage;
  std::unique_ptr<rt::SequenceState> sequence;
  try {
    std::ifstream input(argv[2]);
    const auto spec = json::parse(input);
    hllm::v1::LoadStageRequest request;
    const auto status = google::protobuf::util::JsonStringToMessage(
        spec.at("load_request").dump(), &request);
    if (!status.ok()) throw std::invalid_argument(status.ToString());
    if (request.stage_index() >= static_cast<std::uint32_t>(request.plan().stages_size()))
      throw std::invalid_argument("stage index outside plan");
    const auto capacity_tokens = number(spec, "capacity_tokens", request.manifest().config().maximum_sequence_length());
    const auto prompt = number(spec, "prompt_tokens", capacity_tokens);
    const auto steps = number(spec, "output_tokens", capacity_tokens);
    const auto cycles = number(spec, "cycles", 32U);
    if (!prompt || !steps || !cycles || rt::checked_add(prompt, steps) > capacity_tokens)
      throw std::invalid_argument("invalid workload dimensions");
    const auto& budget = spec.at("budget");
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const rt::MemoryAmounts capacity{number(budget, "host", maximum), number(budget, "device", maximum),
                                     number(budget, "pinned", maximum), number(budget, "unified", maximum)};
    const bool pinned = spec.at("transport_mode") == "pinned";
#ifdef HLLM_WORKER_CUDA
    factory = backend::make_backend_factory(static_cast<int>(number(spec, "device_id", 1024U)), pinned);
#elif defined(HLLM_WORKER_MLX)
    if (pinned) throw std::invalid_argument("MLX has no pinned transport mode");
    factory = backend::make_backend_factory(capacity.unified_bytes);
#else
    if (pinned) throw std::invalid_argument("CPU has no pinned transport mode");
    factory = backend::make_backend_factory();
#endif
    const auto info = factory->profiling_device_info();
    emit({{"event", "environment"}, {"device_identity", info.identity},
          {"device_name", info.name}, {"backend_version", info.backend_version},
          {"driver_version", info.driver_version}, {"allocator", info.allocator},
          {"backend", hllm::v1::Backend_Name(factory->capabilities().kind)},
          {"compiler", __VERSION__}});
    const auto& assignment = request.plan().stages(static_cast<int>(request.stage_index()));
    if (spec.value("mode", "memory") == "compute") {
      stage = factory->load(request, argv[1], capacity);
      hllm::profiling::compute_probe(*stage, capacity, assignment.owns_token_embedding(),
          capacity_tokens, prompt, steps, cycles, emit);
      stage.reset();
      emit({{"event", "complete"}});
      return 0;
    }
    std::atomic_bool cancelled{false};
    for (std::size_t cycle = 0U; cycle < cycles; ++cycle) {
      rt::SequenceMemory reserved{};
      std::size_t completed_steps = 0U;
      bool reset = false;
      std::size_t phase_start_active = 0U;
      const auto reset_phase = [&] {
        reset = factory->profiling_reset_peak();
        const auto a = factory->allocator_metrics();
        phase_start_active = a ? a->active_bytes : 0U;
      };
      const auto snapshot = [&](const char* phase) {
        json allocator = nullptr;
        if (const auto a = factory->allocator_metrics())
          allocator = {{"active_bytes", a->active_bytes}, {"cached_bytes", a->cached_bytes},
                       {"peak_bytes", reset ? std::max({a->peak_bytes, phase_start_active, a->active_bytes}) : a->peak_bytes},
                       {"peak_scope", reset ? "phase" : "unavailable"}};
        emit({{"event", "sample"}, {"cycle", cycle}, {"phase", phase},
              {"weights", amounts(stage ? stage->weight_memory() : rt::MemoryAmounts{})},
              {"cache", amounts(reserved.cache)}, {"workspace", amounts(reserved.workspace)},
              {"allocator", allocator}, {"rss_bytes", optional_bytes(rss())},
              {"rss_lifetime_peak_bytes", optional_bytes(rss_peak())},
              {"device_available_bytes", optional_bytes(factory->profiling_device_info().available_bytes)},
              {"completed_steps", completed_steps}});
      };
      snapshot("baseline");
      reset_phase();
      stage = factory->load(request, argv[1], capacity);
      snapshot("load");
      reset_phase();
      reserved = stage->sequence_memory(capacity_tokens);
      rt::require_memory(rt::add_memory(stage->weight_memory(),
          rt::add_memory(reserved.cache, reserved.workspace)), capacity);
      sequence = stage->allocate_sequence(capacity_tokens);
      snapshot("allocate");
      const auto execute = [&](std::size_t tokens, std::size_t position) {
        rt::StageInput input_value;
        if (assignment.owns_token_embedding()) {
          // Fixed, valid token IDs; only the shape/memory behavior is measured.
          input_value = rt::TokenInput{std::vector<std::uint64_t>(tokens, 0U)};
        } else {
          const auto bytes = rt::checked_multiply(rt::checked_multiply(tokens, stage->hidden_size()), 2U);
          input_value = rt::BoundaryActivation{tokens, stage->hidden_size(), std::vector<std::byte>(bytes)};
        }
        static_cast<void>(stage->execute(std::move(input_value), position, *sequence, cancelled));
        ++completed_steps;
      };
      reset_phase();
      execute(prompt, 0U);
      snapshot("prefill");
      reset_phase();
      for (std::size_t step = 1U; step < steps; ++step) execute(1U, prompt + step - 1U);
      snapshot("decode");
      reset_phase();
      sequence.reset();
      reserved = {};
      snapshot("request_cleanup");
      reset_phase();
      stage.reset();
      snapshot("unload");
    }
    emit({{"event", "complete"}});
    return 0;
  } catch (const std::exception& error) {
    sequence.reset();
    stage.reset();
    emit({{"event", "error"}, {"error", error.what()}, {"cleanup", "RAII-release"}});
    std::cerr << error.what() << '\n';
    return 1;
  }
}
