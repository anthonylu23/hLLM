#include "hllm/runtime/buffer.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hllm::runtime {
namespace {

struct Block {
  std::byte* data;
  std::size_t capacity;
  std::size_t alignment;
};

[[nodiscard]] bool valid_alignment(const std::size_t alignment) noexcept {
  return alignment >= alignof(std::max_align_t) &&
         (alignment & (alignment - 1U)) == 0U;
}

void delete_block(const Block& block) noexcept {
  ::operator delete(block.data, std::align_val_t(block.alignment));
}

}  // namespace

struct BufferPoolState {
  explicit BufferPoolState(const std::size_t maximum) : maximum_cached_bytes(maximum) {}

  ~BufferPoolState() {
    for (const auto& block : cached) {
      delete_block(block);
    }
  }

  std::mutex mutex;
  std::vector<Block> cached;
  std::size_t maximum_cached_bytes;
  std::size_t cached_bytes{0U};
  std::size_t buffers_in_use{0U};
  std::size_t bytes_in_use{0U};
  std::size_t allocations{0U};
};

struct Buffer::Control {
  std::weak_ptr<BufferPoolState> pool;
};

Buffer::Buffer() noexcept = default;

Buffer::Buffer(std::byte* const data, const std::size_t capacity,
               const std::size_t alignment, const MemoryKind memory_kind,
               std::unique_ptr<Control> control) noexcept
    : data_(data),
      capacity_(capacity),
      alignment_(alignment),
      memory_kind_(memory_kind),
      control_(std::move(control)) {}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0U)),
      capacity_(std::exchange(other.capacity_, 0U)),
      alignment_(other.alignment_),
      memory_kind_(other.memory_kind_),
      control_(std::move(other.control_)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0U);
    capacity_ = std::exchange(other.capacity_, 0U);
    alignment_ = other.alignment_;
    memory_kind_ = other.memory_kind_;
    control_ = std::move(other.control_);
  }
  return *this;
}

void Buffer::resize(const std::size_t logical_size) {
  if (logical_size > capacity_) {
    throw std::length_error("buffer logical size exceeds capacity");
  }
  size_ = logical_size;
}

void Buffer::release() noexcept {
  if (data_ == nullptr) {
    return;
  }

  const Block block{data_, capacity_, alignment_};
  auto pool = control_ == nullptr ? nullptr : control_->pool.lock();
  if (pool != nullptr) {
    std::scoped_lock lock(pool->mutex);
    --pool->buffers_in_use;
    pool->bytes_in_use -= capacity_;
    const bool cache_is_unbounded = pool->maximum_cached_bytes == 0U;
    const bool fits_cache = capacity_ <= pool->maximum_cached_bytes -
                                             std::min(pool->cached_bytes,
                                                      pool->maximum_cached_bytes);
    if (cache_is_unbounded || fits_cache) {
      pool->cached.push_back(block);
      pool->cached_bytes += capacity_;
    } else {
      delete_block(block);
    }
  } else {
    delete_block(block);
  }

  data_ = nullptr;
  size_ = 0U;
  capacity_ = 0U;
  control_.reset();
}

BufferPool::BufferPool(const std::size_t maximum_cached_bytes)
    : state_(std::make_shared<BufferPoolState>(maximum_cached_bytes)) {}

Buffer BufferPool::acquire(const std::size_t minimum_capacity,
                           const std::size_t alignment) {
  if (minimum_capacity == 0U) {
    throw std::invalid_argument("buffer capacity must be positive");
  }
  if (!valid_alignment(alignment)) {
    throw std::invalid_argument(
        "buffer alignment must be a power of two and at least max_align_t");
  }

  Block block{};
  {
    std::scoped_lock lock(state_->mutex);
    const auto candidate = std::min_element(
        state_->cached.begin(), state_->cached.end(),
        [minimum_capacity, alignment](const Block& lhs, const Block& rhs) {
          const auto lhs_fits = lhs.capacity >= minimum_capacity &&
                                lhs.alignment >= alignment;
          const auto rhs_fits = rhs.capacity >= minimum_capacity &&
                                rhs.alignment >= alignment;
          if (lhs_fits != rhs_fits) {
            return lhs_fits;
          }
          return lhs.capacity < rhs.capacity;
        });
    if (candidate != state_->cached.end() &&
        candidate->capacity >= minimum_capacity &&
        candidate->alignment >= alignment) {
      block = *candidate;
      state_->cached_bytes -= candidate->capacity;
      state_->cached.erase(candidate);
    } else {
      block = Block{
          static_cast<std::byte*>(
              ::operator new(minimum_capacity, std::align_val_t(alignment))),
          minimum_capacity,
          alignment,
      };
      ++state_->allocations;
    }
    ++state_->buffers_in_use;
    state_->bytes_in_use += block.capacity;
  }

  auto control = std::make_unique<Buffer::Control>();
  control->pool = state_;
  return Buffer(block.data, block.capacity, block.alignment, MemoryKind::kHost,
                std::move(control));
}

BufferPoolStats BufferPool::stats() const {
  std::scoped_lock lock(state_->mutex);
  return BufferPoolStats{
      .buffers_in_use = state_->buffers_in_use,
      .bytes_in_use = state_->bytes_in_use,
      .buffers_cached = state_->cached.size(),
      .bytes_cached = state_->cached_bytes,
      .allocations = state_->allocations,
  };
}

void BufferPool::clear_cached() {
  std::vector<Block> cached;
  {
    std::scoped_lock lock(state_->mutex);
    cached = std::exchange(state_->cached, {});
    state_->cached_bytes = 0U;
  }
  for (const auto& block : cached) {
    delete_block(block);
  }
}

}  // namespace hllm::runtime
