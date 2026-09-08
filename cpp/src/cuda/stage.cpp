#include "hllm/cuda/stage.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

#include "device.hpp"
#include "execution_stream.hpp"
#include "pinned_buffer.hpp"
#include "hllm/runtime/checked_size.hpp"
#include "hllm/runtime/error.hpp"
#include "hllm/runtime/half.hpp"

namespace hllm::cuda {
namespace {
constexpr auto add = runtime::checked_add;
constexpr auto mul = runtime::checked_multiply;

std::int64_t dimension(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw runtime::Error::resource_exhausted("tensor dimension exceeds ATen capacity");
  }
  return static_cast<std::int64_t>(value);
}
void running(const std::atomic_bool& cancelled) {
  if (cancelled.load()) {
    throw std::runtime_error("request cancelled");
  }
}
void finite(const at::Tensor& value) {
  if (!at::isfinite(value).all().item<bool>()) {
    throw std::runtime_error("non-finite CUDA tensor");
  }
}
std::vector<float> snapshot(const at::Tensor& value) {
  auto host = value.to(at::kCPU, at::kFloat).contiguous();
  const auto count = static_cast<std::size_t>(host.numel());
  return {host.const_data_ptr<float>(), host.const_data_ptr<float>() + count};
}

// All operations use one guarded stream and complete before the common runtime
// can release request state. The caching allocator tracks temporary tensor uses
// on that same stream; on exceptions, drain it before propagating the failure.
template <class Function>
auto completed(c10::cuda::CUDAStream stream, Function&& function) {
  const c10::cuda::CUDAStreamGuard guard(stream);
  const c10::InferenceMode inference;
  try {
    auto result = function();
    stream.synchronize();
    return result;
  } catch (const c10::OutOfMemoryError&) {
    static_cast<void>(cudaStreamSynchronize(stream.stream()));
    throw std::bad_alloc();
  } catch (...) {
    static_cast<void>(cudaStreamSynchronize(stream.stream()));
    throw;
  }
}

struct Cache {
  at::Tensor keys;
  at::Tensor values;
};
struct CudaSequence final : runtime::SequenceState {
  const void* owner{};
  std::size_t capacity{};
  std::size_t length{};
  bool failed{false};
  std::vector<Cache> caches;
  std::unique_ptr<PinnedBuffer> staging;
};

class CudaStage final : public ReferenceStage {
 public:
  CudaStage(const model::DenseSource& source, int device_id, const runtime::MemoryAmounts& capacity, bool pinned)
      : config_(source.config),
        vocab_(source.vocabulary_size),
        start_(source.layer_start),
        end_(source.layer_end),
        first_(source.first),
        final_(source.final),
        tied_(source.tied_head),
        pinned_(pinned),
        dtype_(source.execution_dtype == runtime::DataType::kF16 ? at::kHalf : at::kFloat),
        stream_(execution_stream(device_id)) {
    if (std::endian::native != std::endian::little) {
      throw runtime::Error::incompatible_worker(
          "CUDA boundary transfer requires a little-endian host");
    }
    if (pinned_ && capacity.pinned_host_bytes == 0U) {
      throw runtime::Error::incompatible_worker(
          "pinned transfer mode requires pinned host capacity");
    }
    bytes_ = source.float32_weight_bytes / sizeof(float) * element_bytes();
    // Reserve the source payload plus conversion/upload temporaries in each
    // domain. Uploads block; no full-model CPU replica is retained.
    runtime::require_memory(
        {add(add(add(source.largest_payload_bytes, source.verification_workspace_bytes), mul(source.largest_float32_tensor_bytes, 2U)),
             65536U),
         add(bytes_, source.largest_float32_tensor_bytes), 0U},
        capacity);
    // Check every ATen dimension before any weight payload is allocated.
    for (const auto& [name, tensor] : source.tensors) {
      static_cast<void>(name);
      for (const auto size : tensor.shape) {
        static_cast<void>(dimension(size));
      }
    }
    completed(stream_, [&] {
      for (const auto& [name, tensor] : source.tensors) {
        auto host = source.read_float32(name);
        if (dtype_ == at::kHalf) {
          for (const auto value : host) {
            if ((runtime::float_to_float16(value) & 0x7c00U) == 0x7c00U) {
              throw runtime::Error::incompatible_worker(
                  "weight is not representable as finite F16: " + name);
            }
          }
        }
        std::vector<std::int64_t> shape;
        for (const auto size : tensor.shape) {
          shape.push_back(dimension(size));
        }
        auto weight = at::from_blob(host.data(), shape, at::TensorOptions().dtype(at::kFloat))
                          .to(options(), false, true);
        weights_.emplace(name, std::move(weight));
      }
      return true;
    });
  }

  runtime::MemoryAmounts weight_memory() const override { return {0U, bytes_, 0U}; }
  std::size_t hidden_size() const override { return config_.hidden_size; }
  std::size_t vocabulary_size() const override { return vocab_; }
  std::size_t maximum_tokens() const override { return config_.maximum_sequence_length; }
  runtime::SequenceMemory sequence_memory(std::size_t tokens) const override {
    if (tokens == 0U || tokens > maximum_tokens()) {
      throw runtime::Error::invalid_request("request exceeds model context capacity");
    }
    static_cast<void>(dimension(tokens));
    const auto cache =
        mul(mul(mul(mul(tokens, end_ - start_), config_.key_value_heads), config_.head_dimension),
            2U * element_bytes());
    // The initial implementation uses explicit dense attention, including F32
    // scores/softmax. Reserve the worst prefill/decode shape for this sequence.
    const auto projections = mul(config_.attention_heads, config_.head_dimension);
    const auto width = add(add(mul(hidden_size(), 32U), mul(projections, 32U)),
                           mul(config_.intermediate_size, 16U));
    const auto linear = mul(mul(tokens, width), sizeof(float));
    const auto attention =
        mul(mul(mul(tokens, tokens), config_.attention_heads), 4U * sizeof(float));
    const auto device = add(add(add(linear, attention), mul(vocab_, 32U)), 8U * 1024U * 1024U);
    const auto host = add(add(mul(mul(tokens, hidden_size()), 6U), mul(tokens, 16U)), 65536U);
    const auto staging = staging_bytes(tokens);
    return {{0U, cache, 0U}, {add(host, staging), device, staging}};
  }
  std::unique_ptr<runtime::SequenceState> allocate_sequence(std::size_t tokens) const override {
    static_cast<void>(sequence_memory(tokens));
    std::scoped_lock lock(mutex_);
    return completed(stream_, [&]() -> std::unique_ptr<runtime::SequenceState> {
      auto state = std::make_unique<CudaSequence>();
      state->owner = this;
      state->capacity = tokens;
      if (const auto bytes = staging_bytes(tokens); bytes != 0U) {
        state->staging = std::make_unique<PinnedBuffer>(bytes, stream_.stream());
      }
      for (auto layer = start_; layer < end_; ++layer) {
        const std::vector<std::int64_t> shape{dimension(config_.key_value_heads), dimension(tokens),
                                              dimension(config_.head_dimension)};
        state->caches.push_back({at::empty(shape, options()), at::empty(shape, options())});
      }
      return state;
    });
  }
  runtime::StageOutput execute(runtime::StageInput input, std::size_t position,
                               runtime::SequenceState& state,
                               const std::atomic_bool& cancelled) const override {
    return run(std::move(input), position, state, cancelled, nullptr);
  }
  runtime::StageOutput execute_traced(runtime::StageInput input, std::size_t position,
                                      runtime::SequenceState& state,
                                      const std::atomic_bool& cancelled,
                                      ExecutionTrace& trace) const override {
    trace = {};
    return run(std::move(input), position, state, cancelled, &trace);
  }

 private:
  std::size_t staging_bytes(std::size_t tokens) const {
    return pinned_ && !(first_ && final_)
               ? std::min(mul(mul(tokens, hidden_size()), 2U), std::size_t{8U * 1024U * 1024U})
               : 0U;
  }
  std::size_t element_bytes() const { return dtype_ == at::kHalf ? 2U : 4U; }
  at::TensorOptions options() const {
    return at::TensorOptions().device(stream_.device()).dtype(dtype_);
  }
  const at::Tensor& weight(const std::string& name) const { return weights_.at(name); }
  at::Tensor norm(const at::Tensor& input, const at::Tensor& scale) const {
    auto value = input.to(at::kFloat);
    return (value * at::rsqrt(value.square().mean(-1, true) + config_.rms_norm_epsilon))
               .to(dtype_) *
           scale;
  }
  at::Tensor rope(const at::Tensor& input, std::size_t position) const {
    const auto half = dimension(config_.head_dimension / 2U);
    auto exponent = at::arange(half, options().dtype(at::kFloat)) *
                    (2.0 / static_cast<double>(config_.head_dimension));
    auto frequency = at::pow(config_.rope_theta, -exponent);
    auto positions = at::arange(dimension(position), dimension(position) + input.size(0),
                                options().dtype(at::kFloat));
    auto angle = (positions.unsqueeze(1) * frequency.unsqueeze(0)).unsqueeze(1);
    auto cosine = angle.cos();
    auto sine = angle.sin();
    auto a = input.slice(-1, 0, half).to(at::kFloat);
    auto b = input.slice(-1, half).to(at::kFloat);
    return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1).to(dtype_);
  }
  at::Tensor layer(at::Tensor hidden, std::size_t index, std::size_t position, Cache& cache) const {
    const auto prefix = "model.layers." + std::to_string(index) + ".";
    const auto project = [&](const at::Tensor& input, const std::string& name) {
      return at::matmul(input, weight(prefix + name + ".weight").t());
    };
    const auto count = hidden.size(0);
    auto normalized = norm(hidden, weight(prefix + "input_layernorm.weight"));
    auto queries =
        project(normalized, "self_attn.q_proj")
            .view({count, dimension(config_.attention_heads), dimension(config_.head_dimension)});
    auto keys =
        project(normalized, "self_attn.k_proj")
            .view({count, dimension(config_.key_value_heads), dimension(config_.head_dimension)});
    auto values =
        project(normalized, "self_attn.v_proj")
            .view({count, dimension(config_.key_value_heads), dimension(config_.head_dimension)});
    if (config_.query_key_norm) {
      queries = norm(queries, weight(prefix + "self_attn.q_norm.weight"));
      keys = norm(keys, weight(prefix + "self_attn.k_norm.weight"));
    }
    queries = rope(queries, position).transpose(0, 1);
    keys = rope(keys, position).transpose(0, 1);
    const auto past = dimension(position);
    const auto length = past + count;
    cache.keys.slice(1, past, length).copy_(keys);
    cache.values.slice(1, past, length).copy_(values.transpose(0, 1));
    const auto repeat = dimension(config_.attention_heads / config_.key_value_heads);
    auto all_keys = cache.keys.slice(1, 0, length).repeat_interleave(repeat, 0);
    auto all_values = cache.values.slice(1, 0, length).repeat_interleave(repeat, 0);
    auto scores = at::matmul(queries.to(at::kFloat), all_keys.to(at::kFloat).transpose(1, 2)) /
                  std::sqrt(static_cast<double>(config_.head_dimension));
    auto query_positions = at::arange(past, length, options().dtype(at::kLong)).unsqueeze(1);
    auto key_positions = at::arange(length, options().dtype(at::kLong)).unsqueeze(0);
    scores.masked_fill_(key_positions > query_positions, -std::numeric_limits<float>::infinity());
    auto attended = at::matmul(scores.softmax(-1), all_values.to(at::kFloat))
                        .to(dtype_)
                        .transpose(0, 1)
                        .contiguous()
                        .view({count, dimension(config_.attention_heads * config_.head_dimension)});
    hidden = hidden + project(attended, "self_attn.o_proj");
    auto post = norm(hidden, weight(prefix + "post_attention_layernorm.weight"));
    auto gated = at::silu(project(post, "mlp.gate_proj")) * project(post, "mlp.up_proj");
    return hidden + project(gated, "mlp.down_proj");
  }
  runtime::StageOutput run(runtime::StageInput input, std::size_t position,
                           runtime::SequenceState& opaque, const std::atomic_bool& cancelled,
                           ExecutionTrace* trace) const {
    std::scoped_lock lock(mutex_);
    auto* state = dynamic_cast<CudaSequence*>(&opaque);
    if (!state || state->owner != this || state->failed || position != state->length) {
      throw runtime::Error::invalid_request("CUDA sequence state or position is invalid");
    }
    const auto* tokens = std::get_if<runtime::TokenInput>(&input);
    const auto count =
        tokens ? tokens->ids.size() : std::get<runtime::BoundaryActivation>(input).tokens;
    if ((tokens != nullptr) != first_ || count == 0U || position > state->capacity ||
        count > state->capacity - position) {
      throw runtime::Error::invalid_request("invalid CUDA stage input or context capacity");
    }
    running(cancelled);
    try {
      return completed(stream_, [&]() -> runtime::StageOutput {
        at::Tensor hidden;
        if (tokens) {
          std::vector<std::int64_t> ids;
          for (const auto id : tokens->ids) {
            if (id >= vocab_) {
              throw runtime::Error::invalid_request("token ID exceeds vocabulary");
            }
            ids.push_back(dimension(id));
          }
          auto index = at::from_blob(ids.data(), {dimension(ids.size())},
                                     at::TensorOptions().dtype(at::kLong))
                           .to(stream_.device(), at::kLong, false, true);
          hidden = weight("model.embed_tokens.weight").index_select(0, index);
        } else {
          const auto& boundary = std::get<runtime::BoundaryActivation>(input);
          if (boundary.width != hidden_size() ||
              boundary.payload.size() != mul(mul(count, hidden_size()), 2U)) {
            throw runtime::Error::invalid_request("invalid CUDA boundary shape");
          }
          if (state->staging) {
            auto incoming = at::empty({dimension(count), dimension(hidden_size())},
                                      options().dtype(at::kHalf));
            state->staging->upload(incoming.mutable_data_ptr(), boundary.payload.data(),
                                   boundary.payload.size());
            hidden = incoming.to(dtype_);
          } else {
            hidden = at::from_blob(const_cast<std::byte*>(boundary.payload.data()),
                                   {dimension(count), dimension(hidden_size())},
                                   at::TensorOptions().dtype(at::kHalf))
                         .to(options(), false, true);
          }
          finite(hidden);
        }
        for (auto index = start_; index < end_; ++index) {
          running(cancelled);
          hidden = layer(std::move(hidden), index, position, state->caches.at(index - start_));
          if (trace) {
            trace->layers.push_back(snapshot(hidden));
            trace->keys.push_back(snapshot(
                state->caches.at(index - start_).keys.slice(1, 0, dimension(position + count))));
            trace->values.push_back(snapshot(
                state->caches.at(index - start_).values.slice(1, 0, dimension(position + count))));
          }
        }
        finite(hidden);
        running(cancelled);
        runtime::StageOutput result;
        if (final_) {
          auto last = norm(hidden.slice(0, hidden.size(0) - 1), weight("model.norm.weight"));
          auto logits =
              at::matmul(last, weight(tied_ ? "model.embed_tokens.weight" : "lm_head.weight").t());
          finite(logits);
          if (trace) {
            trace->last_logits = snapshot(logits);
          }
          result = runtime::SampledToken{
              static_cast<std::uint64_t>(logits.argmax(-1).item<std::int64_t>())};
        } else {
          auto boundary = hidden.to(at::kHalf);
          finite(boundary);
          boundary = boundary.contiguous();
          runtime::BoundaryActivation output{
              count, hidden_size(), std::vector<std::byte>(mul(mul(count, hidden_size()), 2U))};
          if (state->staging) {
            state->staging->download(output.payload.data(), boundary.const_data_ptr(),
                                     output.payload.size());
          } else {
            auto host = boundary.to(at::kCPU).contiguous();
            std::memcpy(output.payload.data(), host.const_data_ptr(), output.payload.size());
          }
          result = std::move(output);
        }
        running(cancelled);
        state->length += count;
        return result;
      });
    } catch (...) {
      // An interrupted layer may have written part of its cache. Never resume
      // such a state; normal request cleanup destroys it after stream completion.
      state->failed = true;
      throw;
    }
  }

  model::DenseConfig config_;
  std::size_t vocab_, start_, end_;
  bool first_, final_, tied_, pinned_;
  at::ScalarType dtype_;
  c10::cuda::CUDAStream stream_;
  std::size_t bytes_{};
  std::map<std::string, at::Tensor> weights_;
  mutable std::mutex mutex_;
};
}  // namespace

std::unique_ptr<runtime::StageBackend> load_device_stage(const model::DenseSource& source,
                                                         int device_id,
                                                         const runtime::MemoryAmounts& capacity, bool pinned) {
  try {
    return std::make_unique<CudaStage>(source, device_id, capacity, pinned);
  } catch (const c10::OutOfMemoryError&) {
    throw std::bad_alloc();
  }
}
}  // namespace hllm::cuda
