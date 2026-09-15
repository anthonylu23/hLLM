#include <gtest/gtest.h>

#include <array>
#include <chrono>
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

TEST(DownstreamChannel, CachedChannelReconnectsAfterPeerRestart) {
  using namespace std::chrono_literals;
  class Peer final : public v1::WorkerControl::Service {
    grpc::Status GetCapabilities(grpc::ServerContext*, const v1::Empty*,
                                  v1::Capabilities*) override {
      return grpc::Status::OK;
    }
  } service;
  int port = 0;
  auto start = [&](const std::string& address) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    return builder.BuildAndStart();
  };
  auto server = start("127.0.0.1:0");
  ASSERT_TRUE(server);
  const auto address = "127.0.0.1:" + std::to_string(port);
  LoadedDeployment deployment;
  endpoint(deployment, address);
  const auto cached = deployment.downstream_channel();
  auto stub = v1::WorkerControl::NewStub(cached);
  auto call = [&] {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 2s);
    v1::Capabilities response;
    return stub->GetCapabilities(&context, {}, &response);
  };
  ASSERT_TRUE(call().ok());
  server->Shutdown();
  server.reset();
  EXPECT_EQ(call().error_code(), grpc::StatusCode::UNAVAILABLE);
  const auto until = std::chrono::system_clock::now() + 5s;
  while (cached->GetState(true) != GRPC_CHANNEL_TRANSIENT_FAILURE &&
         std::chrono::system_clock::now() < until) {
    std::this_thread::sleep_for(5ms);
  }
  ASSERT_EQ(cached->GetState(false), GRPC_CHANNEL_TRANSIENT_FAILURE);
  server = start(address);
  ASSERT_TRUE(server);
  EXPECT_EQ(deployment.downstream_channel(), cached);
  EXPECT_TRUE(cached->WaitForConnected(std::chrono::system_clock::now() + 10s));
  EXPECT_TRUE(call().ok());
  server->Shutdown();
}

}  // namespace
}  // namespace hllm::worker
