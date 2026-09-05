#include "hllm/worker/control_service.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace hllm::worker {
namespace {

class ModelDirectory final {
 public:
  ModelDirectory() {
    static std::atomic<std::uint64_t> sequence{0U};
    path_ = std::filesystem::temp_directory_path() /
            ("hllm-worker-test-" + std::to_string(sequence.fetch_add(1U)));
    std::filesystem::create_directory(path_);
    const std::string header =
        R"({"model.embed_tokens.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}})";
    std::ofstream stream(path_ / "model.safetensors", std::ios::binary);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (std::size_t index = 0U; index < 8U; ++index) {
      stream.put(static_cast<char>((header_size >> (index * 8U)) & 0xffU));
    }
    stream << header;
    stream.put('\0');
    stream.put('\0');
  }

  ~ModelDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void populate_valid_load(v1::LoadStageRequest& request) {
  auto* plan = request.mutable_plan();
  plan->set_plan_id("plan-1");
  plan->set_plan_digest("plan-digest");
  plan->set_manifest_digest("manifest-digest");
  plan->set_deployment_version(1U);
  auto* stage_zero = plan->add_stages();
  stage_zero->set_stage_index(0U);
  stage_zero->set_worker_id("cpu-a");
  stage_zero->set_layer_start(0U);
  stage_zero->set_layer_end(1U);
  stage_zero->set_owns_token_embedding(true);
  auto* stage_one = plan->add_stages();
  stage_one->set_stage_index(1U);
  stage_one->set_worker_id("cpu-b");
  stage_one->set_layer_start(1U);
  stage_one->set_layer_end(2U);

  auto* manifest = request.mutable_manifest();
  manifest->set_manifest_digest("manifest-digest");
  manifest->mutable_architecture()->set_architecture_id("llama.v1");
  manifest->mutable_config()->set_num_layers(2U);
  manifest->mutable_config()->set_hidden_activation("silu");
  auto* tensor = manifest->add_tensors();
  tensor->set_name("model.embed_tokens.weight");
  tensor->set_file("model.safetensors");
  tensor->set_dtype(v1::DATA_TYPE_F16);
  tensor->add_shape(1U);
  tensor->set_data_offset(0U);
  tensor->set_byte_length(2U);
  tensor->set_role(v1::TENSOR_ROLE_TOKEN_EMBEDDING);
  request.set_stage_index(0U);
}

TEST(WorkerControlTest, LoadsReservesCancelsAndUnloadsOneCpuStage) {
  const ModelDirectory model;
  ControlService service({
      .worker_id = "cpu-a",
      .endpoint = "127.0.0.1:50051",
      .model_root = model.path(),
      .host_memory_capacity_bytes = 1'000'000U,
  });
  grpc::ServerContext context;
  v1::LoadStageRequest load;
  populate_valid_load(load);
  v1::LoadStageResponse loaded;
  EXPECT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  ASSERT_TRUE(loaded.accepted()) << loaded.detail();

  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("plan-1");
  reserve.set_deployment_version(1U);
  reserve.set_request_id("request-1");
  reserve.set_maximum_total_tokens(16U);
  v1::ReserveResponse reserved;
  EXPECT_TRUE(service.ReserveRequest(&context, &reserve, &reserved).ok());
  EXPECT_TRUE(reserved.accepted());

  v1::CancelRequestMessage cancel;
  cancel.set_plan_id("plan-1");
  cancel.set_deployment_version(1U);
  cancel.set_request_id("request-1");
  v1::Empty empty;
  EXPECT_TRUE(service.CancelRequest(&context, &cancel, &empty).ok());

  v1::UnloadStageRequest unload;
  unload.set_plan_id("plan-1");
  unload.set_deployment_version(1U);
  EXPECT_TRUE(service.UnloadStage(&context, &unload, &empty).ok());
}

TEST(WorkerControlTest, RejectsTraversalAndStaleDeployment) {
  const ModelDirectory model;
  ControlService service({
      .worker_id = "cpu-a",
      .endpoint = "127.0.0.1:50051",
      .model_root = model.path(),
      .host_memory_capacity_bytes = 1'000'000U,
  });
  grpc::ServerContext context;
  v1::LoadStageRequest load;
  populate_valid_load(load);
  load.mutable_manifest()->mutable_tensors(0)->set_file("../model.safetensors");
  v1::LoadStageResponse loaded;
  EXPECT_TRUE(service.LoadStage(&context, &load, &loaded).ok());
  EXPECT_FALSE(loaded.accepted());
  EXPECT_EQ(loaded.error().code(), v1::ERROR_CODE_INCOMPATIBLE_WORKER);

  v1::ReserveRequestMessage reserve;
  reserve.set_plan_id("other");
  reserve.set_deployment_version(2U);
  reserve.set_request_id("request-1");
  reserve.set_maximum_total_tokens(16U);
  v1::ReserveResponse reserved;
  EXPECT_TRUE(service.ReserveRequest(&context, &reserve, &reserved).ok());
  EXPECT_FALSE(reserved.accepted());
  EXPECT_EQ(reserved.error().code(), v1::ERROR_CODE_STALE_DEPLOYMENT);
}

}  // namespace
}  // namespace hllm::worker
