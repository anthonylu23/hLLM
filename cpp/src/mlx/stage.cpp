#include "hllm/mlx/stage.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

#include "device.hpp"
#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"

namespace hllm::mlx {
namespace {
constexpr auto add = runtime::checked_add;
constexpr auto mul = runtime::checked_multiply;
int dimension(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw runtime::Error::resource_exhausted("tensor dimension exceeds MLX capacity");
  }
  return static_cast<int>(value);
}
void running(const std::atomic_bool& cancelled) {
  if (cancelled.load()) throw std::runtime_error("request cancelled");
}
void finite(const mx::array& value) {
  if (!mx::all(mx::isfinite(value)).item<bool>()) {
    throw std::runtime_error("non-finite MLX tensor");
  }
}
std::vector<float> snapshot(const mx::array& value) {
  auto host = mx::contiguous(mx::astype(value, mx::float32));
  host.eval();
  return {host.data<float>(), host.data<float>() + host.size()};
}
mx::array slice_axis(const mx::array& value, int axis, int begin, int end) {
  mx::Shape start(value.ndim(), 0);
  auto stop = value.shape();
  start.at(static_cast<std::size_t>(axis)) = begin;
  stop.at(static_cast<std::size_t>(axis)) = end;
  return mx::slice(value, start, stop);
}
struct Cache {
  mx::array keys;
  mx::array values;
};
struct MlxSequence final : runtime::SequenceState {
  const void* owner{};
  std::size_t capacity{}, length{};
  bool failed{false};
  std::vector<Cache> caches;
};

class MlxStage final : public ReferenceStage {
 public:
  MlxStage(const model::DenseSource& source, const runtime::MemoryAmounts& capacity)
      : config_(source.config),
        vocab_(source.vocabulary_size),
        start_(source.layer_start),
        end_(source.layer_end),
        first_(source.first),
        final_(source.final),
        tied_(source.tied_head),
        dtype_(source.execution_dtype == runtime::DataType::kF16 ? mx::float16 : mx::float32),
        stream_(execution_stream()) {
    if (std::endian::native != std::endian::little) {
      throw runtime::Error::incompatible_worker("MLX boundary transfer requires a little-endian host");
    }
    bytes_ = mul(source.float32_weight_bytes / sizeof(float), element_bytes());
    // Source bytes, decoded floats, MLX copied F32 input and conversion can
    // coexist. All are charged to the same physical-memory domain.
    runtime::require_memory({0U, 0U, 0U,
                             add(bytes_, add(add(add(source.largest_payload_bytes, source.verification_workspace_bytes),
                                                 mul(source.largest_float32_tensor_bytes, 3U)),
                                             65536U))},
                            capacity);
    for (const auto& [name, tensor] : source.tensors) {
      static_cast<void>(name);
      for (auto size : tensor.shape) static_cast<void>(dimension(size));
    }
    completed(stream_, [&] {
      for (const auto& [name, tensor] : source.tensors) {
        auto host = source.read_float32(name);
        if (dtype_ == mx::float16) {
          for (auto value : host) {
            if ((runtime::float_to_float16(value) & 0x7c00U) == 0x7c00U) {
              throw runtime::Error::incompatible_worker(
                  "weight is not representable as finite F16: " + name);
            }
          }
        }
        mx::Shape shape;
        for (auto size : tensor.shape) shape.push_back(dimension(size));
        // Iterator construction copies, so checkpoint buffers never outlive
        // their owner through a lazy graph. Evaluate one weight at a time.
        auto value = mx::astype(mx::array(host.begin(), shape), dtype_);
        value.eval();
        weights_.emplace(name, std::move(value));
      }
      return true;
    });
  }
  runtime::MemoryAmounts weight_memory() const override { return {0U, 0U, 0U, bytes_}; }
  std::size_t hidden_size() const override { return config_.hidden_size; }
  std::size_t vocabulary_size() const override { return vocab_; }
  std::size_t maximum_tokens() const override { return config_.maximum_sequence_length; }
  runtime::SequenceMemory sequence_memory(std::size_t tokens) const override {
    if (tokens == 0U || tokens > maximum_tokens()) {
      throw runtime::Error::invalid_request("request exceeds model context capacity");
    }
    static_cast<void>(dimension(tokens));
    auto cache =
        mul(mul(mul(mul(tokens, end_ - start_), config_.key_value_heads), config_.head_dimension),
            2U * element_bytes());
    auto projections = mul(config_.attention_heads, config_.head_dimension);
    auto width = add(add(mul(hidden_size(), 32U), mul(projections, 32U)),
                     mul(config_.intermediate_size, 16U));
    auto linear = mul(mul(tokens, width), sizeof(float));
    auto attention = mul(mul(mul(tokens, tokens), config_.attention_heads), 4U * sizeof(float));
    auto transport = add(mul(mul(tokens, hidden_size()), 12U), mul(tokens, 16U));
    // slice_update may copy the complete cache; do not assume in-place reuse.
    auto workspace = add(add(add(add(linear, attention), cache), transport),
                         add(mul(vocab_, 32U), 8U * 1024U * 1024U));
    return {{0U, 0U, 0U, cache}, {0U, 0U, 0U, workspace}};
  }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t tokens) const override {
    static_cast<void>(sequence_memory(tokens));
    std::scoped_lock lock(device_mutex());
    return completed(stream_, [&]() -> std::unique_ptr<runtime::SequenceState> {
      auto state = std::make_unique<MlxSequence>();
      state->owner = this;
      state->capacity = tokens;
      for (auto layer = start_; layer < end_; ++layer) {
        mx::Shape shape{dimension(config_.key_value_heads), dimension(tokens),
                        dimension(config_.head_dimension)};
        auto keys = mx::zeros(shape, dtype_);
        auto values = mx::zeros(shape, dtype_);
        mx::eval(keys, values);
        state->caches.push_back({std::move(keys), std::move(values)});
      }
      return state;
    });
  }
  runtime::StageOutput execute(runtime::StageInput input, std::size_t position,
                               runtime::SequenceState& state,
                               const std::atomic_bool& cancelled) const override {
    return run(std::move(input), position, state, cancelled, nullptr);
  }
  runtime::StageOutput execute_profiled(runtime::StageInput input, std::size_t position,
      runtime::SequenceState& state, const std::atomic_bool& cancelled,
      runtime::ExecutionTiming& timing) const override {
    timing = {};
    return run(std::move(input), position, state, cancelled, nullptr, &timing);
  }
  runtime::StageOutput execute_traced(runtime::StageInput input, std::size_t position,
                                      runtime::SequenceState& state,
                                      const std::atomic_bool& cancelled,
                                      ExecutionTrace& trace) const override {
    trace = {};
    return run(std::move(input), position, state, cancelled, &trace);
  }

 private:
  std::size_t element_bytes() const { return dtype_ == mx::float16 ? 2U : 4U; }
  const mx::array& weight(const std::string& name) const { return weights_.at(name); }
  mx::array norm(const mx::array& input, const mx::array& scale) const {
    auto value = mx::astype(input, mx::float32);
    return mx::astype(value * mx::rsqrt(mx::mean(mx::square(value), -1, true) +
                                        mx::array(config_.rms_norm_epsilon)),
                      dtype_) *
           scale;
  }
  mx::array rope(const mx::array& input, std::size_t position) const {
    auto half = dimension(config_.head_dimension / 2U);
    auto exponent = mx::arange(half, mx::float32) *
                    mx::array(2.0F / static_cast<float>(config_.head_dimension));
    auto frequency = mx::power(mx::array(config_.rope_theta), -exponent);
    auto positions =
        mx::arange(dimension(position), dimension(position) + input.shape(0), mx::float32);
    auto angle = mx::expand_dims(mx::expand_dims(positions, 1) * mx::expand_dims(frequency, 0), 1);
    auto cosine = mx::cos(angle);
    auto sine = mx::sin(angle);
    auto a = mx::astype(slice_axis(input, 2, 0, half), mx::float32);
    auto b = mx::astype(slice_axis(input, 2, half, 2 * half), mx::float32);
    return mx::astype(mx::concatenate({a * cosine - b * sine, b * cosine + a * sine}, -1), dtype_);
  }
  mx::array layer(mx::array hidden, std::size_t index, std::size_t position, Cache& cache) const {
    auto prefix = "model.layers." + std::to_string(index) + ".";
    auto project = [&](const mx::array& input, const std::string& name) {
      return mx::matmul(input, mx::transpose(weight(prefix + name + ".weight")));
    };
    auto count = hidden.shape(0);
    auto normalized = norm(hidden, weight(prefix + "input_layernorm.weight"));
    auto queries =
        mx::reshape(project(normalized, "self_attn.q_proj"),
                    {count, dimension(config_.attention_heads), dimension(config_.head_dimension)});
    auto keys =
        mx::reshape(project(normalized, "self_attn.k_proj"),
                    {count, dimension(config_.key_value_heads), dimension(config_.head_dimension)});
    auto values =
        mx::reshape(project(normalized, "self_attn.v_proj"),
                    {count, dimension(config_.key_value_heads), dimension(config_.head_dimension)});
    if (config_.query_key_norm) {
      queries = norm(queries, weight(prefix + "self_attn.q_norm.weight"));
      keys = norm(keys, weight(prefix + "self_attn.k_norm.weight"));
    }
    queries = mx::transpose(rope(queries, position), {1, 0, 2});
    keys = mx::transpose(rope(keys, position), {1, 0, 2});
    auto past = dimension(position);
    auto length = past + count;
    mx::Shape begin{0, past, 0};
    mx::Shape stop{dimension(config_.key_value_heads), length, dimension(config_.head_dimension)};
    cache.keys = mx::slice_update(cache.keys, keys, begin, stop);
    cache.values = mx::slice_update(cache.values, mx::transpose(values, {1, 0, 2}), begin, stop);
    auto repeat = dimension(config_.attention_heads / config_.key_value_heads);
    auto all_keys = mx::repeat(slice_axis(cache.keys, 1, 0, length), repeat, 0);
    auto all_values = mx::repeat(slice_axis(cache.values, 1, 0, length), repeat, 0);
    auto scores = mx::matmul(mx::astype(queries, mx::float32),
                             mx::transpose(mx::astype(all_keys, mx::float32), {0, 2, 1})) /
                  mx::array(std::sqrt(static_cast<float>(config_.head_dimension)));
    auto query_positions = mx::expand_dims(mx::arange(past, length, mx::int32), 1);
    auto key_positions = mx::expand_dims(mx::arange(length, mx::int32), 0);
    scores = mx::where(key_positions > query_positions,
                       mx::array(-std::numeric_limits<float>::infinity()), scores);
    auto attended =
        mx::reshape(mx::transpose(mx::astype(mx::matmul(mx::softmax(scores, -1, true),
                                                        mx::astype(all_values, mx::float32)),
                                             dtype_),
                                  {1, 0, 2}),
                    {count, dimension(mul(config_.attention_heads, config_.head_dimension))});
    hidden = hidden + project(attended, "self_attn.o_proj");
    auto post = norm(hidden, weight(prefix + "post_attention_layernorm.weight"));
    auto gate = project(post, "mlp.gate_proj");
    auto gated = (gate * mx::sigmoid(gate)) * project(post, "mlp.up_proj");
    hidden = hidden + project(gated, "mlp.down_proj");
    // Bound lazy graph lifetime to a layer, including cache updates. This also
    // provides a cancellation checkpoint after actual device completion.
    mx::eval(hidden, cache.keys, cache.values);
    return hidden;
  }
  runtime::StageOutput run(runtime::StageInput input, std::size_t position,
                           runtime::SequenceState& opaque, const std::atomic_bool& cancelled,
                           ExecutionTrace* trace, runtime::ExecutionTiming* timing = nullptr) const {
    std::scoped_lock lock(device_mutex());
    auto* state = dynamic_cast<MlxSequence*>(&opaque);
    if (!state || state->owner != this || state->failed || position != state->length) {
      throw runtime::Error::invalid_request("MLX sequence state or position is invalid");
    }
    auto* tokens = std::get_if<runtime::TokenInput>(&input);
    auto count = tokens ? tokens->ids.size() : std::get<runtime::BoundaryActivation>(input).tokens;
    if ((tokens != nullptr) != first_ || count == 0U || position > state->capacity ||
        count > state->capacity - position) {
      throw runtime::Error::invalid_request("invalid MLX stage input or context capacity");
    }
    running(cancelled);
    try {
      return completed(stream_, [&]() -> runtime::StageOutput {
        runtime::PhaseTimer timer(timing, [&] { mx::synchronize(stream_); });
        auto hidden = [&]() -> mx::array {
          if (tokens) {
            std::vector<std::int32_t> ids;
            for (auto id : tokens->ids) {
              if (id >= vocab_) throw runtime::Error::invalid_request("token ID exceeds vocabulary");
              ids.push_back(dimension(id));
            }
            return mx::take(weight("model.embed_tokens.weight"),
                            mx::array(ids.begin(), {dimension(count)}), 0);
          }
          const auto& boundary = std::get<runtime::BoundaryActivation>(input);
          if (boundary.width != hidden_size() ||
              boundary.payload.size() != mul(mul(count, hidden_size()), 2U)) {
            throw runtime::Error::invalid_request("invalid MLX boundary shape");
          }
          // Decode into an owned float vector; avoid aliasing byte storage or
          // relying on a transport buffer's alignment/lifetime.
          std::vector<float> values;
          values.reserve(mul(count, hidden_size()));
          for (std::size_t i = 0U; i < boundary.payload.size(); i += 2U) {
            auto bits = static_cast<std::uint16_t>(
                std::to_integer<unsigned>(boundary.payload[i]) |
                (std::to_integer<unsigned>(boundary.payload[i + 1U]) << 8U));
            auto value = runtime::float16_to_float(bits);
            if (!std::isfinite(value)) throw runtime::Error::invalid_request("non-finite MLX boundary");
            values.push_back(value);
          }
          return mx::astype(mx::array(values.begin(), {dimension(count), dimension(hidden_size())}),
                            dtype_);
        }();
        if (timing) mx::eval(hidden);
        timer.mark(tokens ? "embedding" : "from-wire");
        for (auto index = start_; index < end_; ++index) {
          running(cancelled);
          auto& cache = state->caches.at(index - start_);
          hidden = layer(std::move(hidden), index, position, cache);
          timer.mark("layer", index);
          if (trace) {
            trace->layers.push_back(snapshot(hidden));
            trace->keys.push_back(
                snapshot(slice_axis(cache.keys, 1, 0, dimension(position + count))));
            trace->values.push_back(
                snapshot(slice_axis(cache.values, 1, 0, dimension(position + count))));
          }
        }
        finite(hidden);
        running(cancelled);
        timer.mark("validation");
        runtime::StageOutput result;
        if (final_) {
          auto last = norm(slice_axis(hidden, 0, hidden.shape(0) - 1, hidden.shape(0)),
                           weight("model.norm.weight"));
          if (timing) mx::eval(last);
          timer.mark("final_norm");
          auto logits = mx::matmul(
              last, mx::transpose(weight(tied_ ? "model.embed_tokens.weight" : "lm_head.weight")));
          finite(logits);
          timer.mark("lm_head");
          if (trace) trace->last_logits = snapshot(logits);
          result = runtime::SampledToken{mx::argmax(logits, -1).item<std::uint32_t>()};
          timer.mark("sampling");
        } else {
          auto boundary = mx::contiguous(mx::astype(hidden, mx::float16));
          finite(boundary);
          boundary.eval();
          runtime::BoundaryActivation output{
              count, hidden_size(), std::vector<std::byte>(mul(mul(count, hidden_size()), 2U))};
          std::memcpy(output.payload.data(), boundary.data<mx::float16_t>(), output.payload.size());
          result = std::move(output);
          timer.mark("to-wire");
        }
        running(cancelled);
        state->length += count;
        return result;
      });
    } catch (...) {
      state->failed = true;
      throw;
    }
  }
  model::DenseConfig config_;
  std::size_t vocab_, start_, end_;
  bool first_, final_, tied_;
  mx::Dtype dtype_;
  mx::Stream stream_;
  std::size_t bytes_{};
  std::map<std::string, mx::array> weights_;
};
}  // namespace
std::unique_ptr<runtime::StageBackend> load_device_stage(const model::DenseSource& source,
                                                         const runtime::MemoryAmounts& capacity) {
  std::scoped_lock lock(device_mutex());
  return std::make_unique<MlxStage>(source, capacity);
}
}  // namespace hllm::mlx
