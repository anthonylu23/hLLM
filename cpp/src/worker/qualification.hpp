#pragma once
#include <openssl/evp.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <optional>
#include <array>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <mach-o/dyld.h>
#endif

namespace hllm::worker {
inline std::optional<std::uint64_t> available_host_memory() {
#ifdef __APPLE__
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_size_t page = 0;
  const auto host = mach_host_self();
  const bool ok = host_page_size(host, &page) == KERN_SUCCESS &&
      host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS;
  mach_port_deallocate(mach_task_self(), host);
  if (ok) return (static_cast<std::uint64_t>(vm.free_count) + vm.inactive_count) * page;
#else
  std::ifstream input("/proc/meminfo");
  std::string line;
  while (std::getline(input, line)) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream values(line.substr(13));
      std::uint64_t kb;
      if (values >> kb) return kb * 1024U;
    }
  }
#endif
  return std::nullopt;
}
inline std::string qualified_device_name(std::string name) {
#ifdef __APPLE__
  const auto value = [](const char* key) {
    std::size_t size = 0;
    if (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) return std::string{};
    std::string result(size, '\0');
    if (sysctlbyname(key, result.data(), &size, nullptr, 0) != 0) return std::string{};
    if (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
  };
  name += "; " + value("machdep.cpu.brand_string") + "; " + value("hw.model");
#endif
  return name;
}
inline std::string text_digest(const std::string& value) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
  unsigned int length = 0;
  if (EVP_Digest(value.data(), value.size(), hash.data(), &length, EVP_sha256(), nullptr) != 1) return {};
  std::ostringstream result;
  for (unsigned int i = 0; i < length; ++i) result << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(hash[i]);
  return result.str();
}
inline std::string executable_digest() {
  std::string path = "/proc/self/exe";
#ifdef __APPLE__
  std::uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  path.resize(size);
  if (_NSGetExecutablePath(path.data(), &size) != 0) return {};
  path.resize(path.find('\0'));
#endif
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) return {};
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    if (EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(input.gcount())) != 1) return {};
  }
  if (!input.eof()) return {};
  std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(ctx.get(), hash.data(), &length) != 1) return {};
  std::ostringstream result;
  for (unsigned int i = 0; i < length; ++i) result << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(hash[i]);
  return result.str();
}
}  // namespace hllm::worker
