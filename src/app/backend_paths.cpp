#include "app/backend_paths.h"

#include <string>
#include <string_view>
#include <vector>

#include "app/play/player.h"

namespace kirari::app {
namespace {

// CMake donne aux bibliothèques MODULE le suffixe .so aussi sous macOS.
#ifdef _WIN32
constexpr std::string_view kModuleSuffix = ".dll";
#else
constexpr std::string_view kModuleSuffix = ".so";
#endif

std::vector<std::filesystem::path> module_dirs(const std::filesystem::path& base) {
  std::vector<std::filesystem::path> dirs{base};
#ifdef __APPLE__
  // Dans un paquet .app, SDL_GetBasePath donne Contents/Resources.
  dirs.push_back(base / ".." / "PlugIns");
#endif
  return dirs;
}

}  // namespace

void add_backend_paths(play::PlayerPaths& paths, const std::filesystem::path& base) {
  const auto dirs = module_dirs(base);
  const auto beside = [&](std::string_view backend) {
    const std::string name = "kirari_backend_" + std::string(backend) + std::string(kModuleSuffix);
    for (const auto& dir : dirs) paths.backends.push_back(dir / name);
  };
  beside("tensorrt");
#ifdef KIRARI_BACKEND_TENSORRT
  paths.backends.emplace_back(KIRARI_BACKEND_TENSORRT);
#endif
  beside("coreml");
#ifdef KIRARI_BACKEND_COREML
  paths.backends.emplace_back(KIRARI_BACKEND_COREML);
#endif
  beside("vulkan");
#ifdef KIRARI_BACKEND_VULKAN
  paths.backends.emplace_back(KIRARI_BACKEND_VULKAN);
#endif
  beside("zml");
#ifdef KIRARI_BACKEND_ZML
  paths.backends.emplace_back(KIRARI_BACKEND_ZML);
#endif
  beside("hip");
#ifdef KIRARI_BACKEND_HIP
  paths.backends.emplace_back(KIRARI_BACKEND_HIP);
#endif
  beside("slang");
#ifdef KIRARI_BACKEND_SLANG
  paths.backends.emplace_back(KIRARI_BACKEND_SLANG);
#endif
  beside("triton");
#ifdef KIRARI_BACKEND_TRITON
  paths.backends.emplace_back(KIRARI_BACKEND_TRITON);
#endif
}

}  // namespace kirari::app
