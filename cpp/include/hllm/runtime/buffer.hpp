#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace hllm::runtime {

struct BufferPoolState;

enum class MemoryKind : std::uint8_t {
  kHost,
  kCudaPinnedHost,
  kMlxHostVisible,
  kCudaDevice,
};

class Buffer final {
 public:
  Buffer() noexcept;
  ~Buffer();

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  [[nodiscard]] std::byte* data() noexcept { return data_; }
  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] MemoryKind memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }
  [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

  [[nodiscard]] std::span<std::byte> bytes() noexcept { return {data_, size_}; }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }

  void resize(std::size_t logical_size);
  void clear() noexcept { size_ = 0U; }

 private:
  struct Control;
  friend class BufferPool;

  Buffer(std::byte* data, std::size_t capacity, std::size_t alignment,
         MemoryKind memory_kind, std::unique_ptr<Control> control) noexcept;
  void release() noexcept;

  std::byte* data_{nullptr};
  std::size_t size_{0U};
  std::size_t capacity_{0U};
  std::size_t alignment_{alignof(std::max_align_t)};
  MemoryKind memory_kind_{MemoryKind::kHost};
  std::unique_ptr<Control> control_;
};

struct BufferPoolStats {
  std::size_t buffers_in_use;
  std::size_t bytes_in_use;
  std::size_t buffers_cached;
  std::size_t bytes_cached;
  std::size_t allocations;
};

class BufferPool final {
 public:
  explicit BufferPool(std::size_t maximum_cached_bytes = 0U);
  ~BufferPool() = default;

  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  BufferPool(BufferPool&&) noexcept = default;
  BufferPool& operator=(BufferPool&&) noexcept = default;

  [[nodiscard]] Buffer acquire(
      std::size_t minimum_capacity,
      std::size_t alignment = alignof(std::max_align_t));
  [[nodiscard]] BufferPoolStats stats() const;
  void clear_cached();

 private:
  std::shared_ptr<BufferPoolState> state_;
};

}  // namespace hllm::runtime
