#include "hllm/runtime/safetensors.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace hllm::runtime {
namespace {

class TemporarySafetensors final {
 public:
  TemporarySafetensors(std::string header, const std::string& payload) {
    static std::atomic<std::uint64_t> sequence{0U};
    path_ = std::filesystem::temp_directory_path() /
            ("hllm-safetensors-test-" + std::to_string(sequence.fetch_add(1U)) + ".bin");
    std::ofstream stream(path_, std::ios::binary);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (std::size_t index = 0U; index < 8U; ++index) {
      stream.put(static_cast<char>((header_size >> (index * 8U)) & 0xffU));
    }
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }

  ~TemporarySafetensors() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  TemporarySafetensors(const TemporarySafetensors&) = delete;
  TemporarySafetensors& operator=(const TemporarySafetensors&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(SafetensorsFileTest, ParsesMetadataAndReadsOnlyRequestedTensor) {
  const TemporarySafetensors file(
      R"({"first":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"second":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})",
      std::string{"\x01\x02\x03\x04\x05\x06", 6U});

  const SafetensorsFile inspected(file.path());
  ASSERT_EQ(inspected.tensors().size(), 2U);
  EXPECT_EQ(inspected.tensor("second").dtype, DataType::kBF16);
  EXPECT_EQ(inspected.tensor("second").shape, std::vector<std::size_t>{1U});

  BufferPool pool;
  const auto contents = inspected.read_tensor("second", pool);
  ASSERT_EQ(contents.size(), 2U);
  EXPECT_EQ(contents.bytes()[0], std::byte{0x05});
  EXPECT_EQ(contents.bytes()[1], std::byte{0x06});
}

TEST(SafetensorsFileTest, RejectsInconsistentTensorLength) {
  const TemporarySafetensors file(
      R"({"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,6]}})",
      std::string(6U, '\0'));
  EXPECT_THROW(static_cast<void>(SafetensorsFile(file.path())), SafetensorsError);
}

TEST(SafetensorsFileTest, RejectsOverlappingTensors) {
  const TemporarySafetensors file(
      R"({"first":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"second":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}})",
      std::string(6U, '\0'));
  EXPECT_THROW(static_cast<void>(SafetensorsFile(file.path())), SafetensorsError);
}

TEST(SafetensorsFileTest, RejectsPayloadRangeBeyondFile) {
  const TemporarySafetensors file(
      R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
      std::string(4U, '\0'));
  EXPECT_THROW(static_cast<void>(SafetensorsFile(file.path())), SafetensorsError);
}

TEST(SafetensorsFileTest, RejectsZeroLengthTensorStartingBeyondPayload) {
  const TemporarySafetensors file(
      R"({"weight":{"dtype":"F32","shape":[0],"data_offsets":[5,5]}})", "");
  EXPECT_THROW(static_cast<void>(SafetensorsFile(file.path())), SafetensorsError);
}

}  // namespace
}  // namespace hllm::runtime

#include "hllm/model/dense_source.hpp"

namespace hllm::runtime {
namespace {
TEST(DenseSourceTest, ConvertsAcrossChunkBoundaryAndDetectsLateTiedHeadCorruption) {
  const auto bytes = model::DenseSource::conversion_chunk_bytes + 16U;
  const auto elements = bytes / 2U;
  std::string payload;
  for (std::size_t i = 0; i < elements; ++i) payload += std::string("\x80\x3f", 2U);  // BF16 1.0
  auto head = payload;
  head.back() = '\0';
  const auto header =
      std::string("{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[") +
      std::to_string(elements) + "],\"data_offsets\":[0," + std::to_string(bytes) +
      "]},\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(elements) +
      "],\"data_offsets\":[" + std::to_string(bytes) + "," + std::to_string(bytes * 2U) + "]}}";
  const TemporarySafetensors file(header, payload + head);
  model::DenseSource source;
  source.files.emplace("weights", SafetensorsFile(file.path()));
  source.tensors.emplace("model.embed_tokens.weight",
                         model::TensorSource{"weights", {elements}, DataType::kBF16});
  source.largest_payload_bytes = bytes;
  source.verification_workspace_bytes = model::DenseSource::conversion_chunk_bytes;
  std::size_t seen = 0, calls = 0;
  source.for_each_float32_chunk("model.embed_tokens.weight", [&](std::span<const float> chunk) {
    ++calls;
    seen += chunk.size();
    EXPECT_LE(chunk.size(), model::DenseSource::conversion_chunk_bytes / 2U);
    EXPECT_TRUE(std::all_of(chunk.begin(), chunk.end(), [](float value) { return value == 1.0F; }));
  });
  EXPECT_EQ(seen, elements);
  EXPECT_EQ(calls, 2U);
  source.redundant_head_file = "weights";
  EXPECT_THROW(source.for_each_float32_chunk("model.embed_tokens.weight", [](auto) {}), Error);
  // Metadata was valid when inspected; a later truncation must still fail while
  // reading the second conversion chunk, rather than accepting a partial tensor.
  source.redundant_head_file.reset();
  std::filesystem::resize_file(file.path(), std::filesystem::file_size(file.path()) - bytes - 1U);
  calls = 0;
  EXPECT_THROW(source.for_each_float32_chunk("model.embed_tokens.weight", [&](auto) { ++calls; }),
               Error);
  EXPECT_EQ(calls, 1U);
}
}  // namespace
}  // namespace hllm::runtime
