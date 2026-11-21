#include "core/zml_plugin.h"

#include <cstdlib>
#include <string>

#include "core/utf8_path.h"

namespace kirari {
std::filesystem::path zml_runtime_asset(const std::filesystem::path& module,
                                        const std::filesystem::path& relative) {
  std::error_code ec;
  auto bundled = module.parent_path() / "zml" / relative;
  if (std::filesystem::is_regular_file(bundled, ec)) return bundled;
#ifdef __linux__
  std::filesystem::path data;
  if (const char* xdg = std::getenv("XDG_DATA_HOME")) data = core::path_from_utf8(xdg);  // env: système
  if (!data.is_absolute()) {
    const char* home = std::getenv("HOME");  // env: système
    if (home == nullptr || !core::path_from_utf8(home).is_absolute()) return {};
    data = core::path_from_utf8(home) / ".local/share";
  }
  auto external = data / "kirari/zml" / relative;
  if (std::filesystem::is_regular_file(external, ec)) return external;
#endif
  return {};
}

std::filesystem::path zml_plugin_path(const std::filesystem::path& module, GpuVendor vendor) {
  const char* platform = nullptr;
  const char* development = nullptr;
  switch (vendor) {
    case GpuVendor::Nvidia:
      platform = "cuda";
      development = KIRARI_ZML_CUDA_PLUGIN;
      break;
    case GpuVendor::Amd:
      platform = "rocm";
      development = KIRARI_ZML_ROCM_PLUGIN;
      break;
    case GpuVendor::Intel:
      platform = "oneapi";
      development = KIRARI_ZML_ONEAPI_PLUGIN;
      break;
    case GpuVendor::Apple:
      platform = "metal";
      development = KIRARI_ZML_METAL_PLUGIN;
      break;
    default:
      return {};
  }
#if defined(_WIN32)
  auto library = std::string("zml_") + platform + ".dll";
#elif defined(__APPLE__)
  auto library = std::string("libzml_") + platform + ".dylib";
#else
  auto library = std::string("libzml_") + platform + ".so";
#endif
  const auto relative = std::filesystem::path(platform) / "lib" / library;
  if (auto installed = zml_runtime_asset(module, relative); !installed.empty()) return installed;
  auto dev = core::path_from_utf8(development);
  std::error_code ec;
  if (dev.is_absolute() && std::filesystem::is_regular_file(dev, ec)) return dev;
  return {};
}
}  // namespace kirari
