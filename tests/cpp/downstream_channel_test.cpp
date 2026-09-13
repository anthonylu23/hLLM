#include <gtest/gtest.h>

#include <array>
#include <thread>

#include "hllm/worker/execution_service.hpp"

namespace hllm::worker {
namespace {

void endpoint(LoadedDeployment& deployment, const std::string& target) {
  auto* stage = deployment.spec.add_stage_endpoints();
  stage->set_stage_index(1U);
  stage->set_endpoint(target);
}

TEST(DownstreamChannel, ConcurrentRequestsShareDeploymentChannel) {
  LoadedDeployment deployment;
  endpoint(deployment, "127.0.0.1:50051");
  std::array<std::shared_ptr<grpc::Channel>, 8> channels;
  std::array<std::thread, 8> threads;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] { channels[i] = deployment.downstream_channel(); });
  }
  for (auto& thread : threads) thread.join();
  ASSERT_NE(channels[0], nullptr);
  for (const auto& channel : channels) EXPECT_EQ(channel, channels[0]);
}

TEST(DownstreamChannel, ChannelLifetimeFollowsDeploymentAndOutstandingUsers) {
  std::weak_ptr<grpc::Channel> observed;
  std::shared_ptr<grpc::Channel> in_flight;
  {
    auto deployment = std::make_shared<LoadedDeployment>();
    endpoint(*deployment, "127.0.0.1:50051");
    in_flight = deployment->downstream_channel();
    observed = in_flight;
    auto replacement = std::make_shared<LoadedDeployment>();
    endpoint(*replacement, "127.0.0.1:50052");
    EXPECT_NE(replacement->downstream_channel(), in_flight);
    deployment.reset();
    EXPECT_FALSE(observed.expired());
  }
  in_flight.reset();
  EXPECT_TRUE(observed.expired());
}

TEST(DownstreamChannel, MissingEndpointFailsBeforeCreatingChannel) {
  LoadedDeployment deployment;
  EXPECT_THROW(deployment.downstream_channel(), std::logic_error);
}

}  // namespace
}  // namespace hllm::worker
