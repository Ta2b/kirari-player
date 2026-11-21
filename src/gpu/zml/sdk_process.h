#pragma once

#include <chrono>
#include <format>

#include "core/dynamic_library.h"
#include "core/utf8_path.h"
#include "core/zml_plugin.h"
#include "gpu/backend.h"
#include "prepare/process.h"

namespace kirari::gpu::zml {
inline std::filesystem::path sdk_frontend_path() {
  const auto module = library_containing(reinterpret_cast<const void*>(&sdk_frontend_path));
  std::filesystem::path frontend = "frontend/span_frontend";
#ifdef _WIN32
  frontend += ".exe";
#endif
  return zml_runtime_asset(module, frontend);
}
inline std::expected<bool, Refusal> run_sdk_frontend(std::vector<std::string> arguments, Progress& progress) {
  prepare::ProcessSpec spec{
      .program = sdk_frontend_path(), .args = std::move(arguments), .timeout = std::chrono::minutes(5)};
  std::string diagnostic;
  auto launched = prepare::run_process(
      spec,
      [&](std::string_view line) {
        constexpr std::size_t limit = 4096;
        if (!diagnostic.empty()) diagnostic += '\n';
        diagnostic += line.substr(0, limit);
        if (diagnostic.size() > limit) diagnostic.erase(0, diagnostic.size() - limit);
      },
      progress.stop);
  if (!launched) return std::unexpected(launched.error());
  if (launched->kind == prepare::ExitKind::Cancelled)
    return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
  if (launched->kind != prepare::ExitKind::Exited || launched->code != 0)
    return std::unexpected(Refusal{RefusalCode::BuilderFailed, std::format("frontal Zig de ZML, code {} : {}",
                                                                           launched->code, diagnostic)});
  return true;
}
}  // namespace kirari::gpu::zml
