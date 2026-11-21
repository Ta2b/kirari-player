#include "gpu/zml/platform.h"

#include <cstdlib>
#include <mutex>
#include <string>

#include "core/utf8_path.h"

namespace kirari::gpu::zml {
void prepare_platform(GpuVendor vendor, const std::filesystem::path& plugin) {
#ifndef _WIN32
  static std::mutex mutex;
  const std::scoped_lock lock(mutex);
  const auto root = plugin.parent_path().parent_path();
  std::error_code ec;
  if (vendor == GpuVendor::Apple && std::filesystem::is_directory(root / "toolchain/bin", ec))
    ::setenv("METAL_TOOLCHAIN", (root / "toolchain/bin").c_str(),
             0);  // env: système, compilateur Metal du SDK
  if (vendor == GpuVendor::Amd)
    ::setenv("ROCM_PATH", root.c_str(), 0);  // env: système, assembleur et bitcode ROCm
  if (vendor == GpuVendor::Intel && std::filesystem::is_regular_file(root / "lib/libze_intel_gpu.so.1", ec))
    ::setenv("ZE_ENABLE_ALT_DRIVERS", (root / "lib/libze_intel_gpu.so.1").c_str(),
             0);  // env: système, Level Zero du SDK
  if (vendor == GpuVendor::Nvidia && std::filesystem::exists(root / "nvvm/libdevice", ec)) {
    const char* existing = std::getenv("XLA_FLAGS");  // env: système, options héritées du compilateur XLA
    std::string flags = existing ? existing : "";
    if (flags.find("--xla_gpu_cuda_data_dir") == std::string::npos) {
      auto path = core::to_utf8(root);
      if (path.find_first_of("\\\"\r\n") == std::string::npos) {
        flags += " --xla_gpu_cuda_data_dir=\"" + path + "\"";
        ::setenv("XLA_FLAGS", flags.c_str(), 1);  // env: système, libdevice et ptxas de ce greffon
      }
    }
  }
#else
  static_cast<void>(vendor);
  static_cast<void>(plugin);
#endif
}
}  // namespace kirari::gpu::zml
