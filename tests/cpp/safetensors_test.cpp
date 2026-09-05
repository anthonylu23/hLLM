#include "hllm/runtime/safetensors.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <gtest/gtest.h>

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
