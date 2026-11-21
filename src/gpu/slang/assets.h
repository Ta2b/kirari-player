#pragma once
#include <algorithm>

#include "core/file_sync.h"
#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/compute/refuse.h"
#include "plan.h"
#include "prepare/sha256.h"

namespace kirari::gpu::slang {
namespace fs = std::filesystem;
constexpr Shape kX{{1, 3, 1080, 1920}}, kY{{1, 3, 2160, 3840}};
struct Package {
  std::string_view folder, model, manifest;
};
constexpr Package kPackages[] = {
#include "slang_packages.inc"
};

inline std::string recipe_signature() {
  std::string signature;
  for (const auto& package : kPackages) signature.append(package.manifest);
  return signature;
}

inline auto error(std::string msg) {
  return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "Slang SPAN : " + msg});
}
inline std::string hash(const fs::path& path) {
  auto output = prepare::sha256_file(path);
  if (!output) throw std::runtime_error(output.error().detail);
  return *output;
}
inline c64::balanced::Plan verified(const fs::path& base, const Package& package, bool metal = false) {
  const auto root = base / package.folder;
  c64::require(hash(root / "SHA256SUMS") == package.manifest, "empreinte du paquet Slang incorrecte");
  std::ifstream sums(root / "SHA256SUMS");
  std::string digest, name;
  std::map<fs::path, bool> files;
  while (sums >> digest >> name) {
    c64::require(fs::path(name).filename() == name && name != "." && name != ".." &&
                     files.emplace(root / name, true).second && hash(root / name) == digest,
                 "fichier de paquet Slang incorrect");
  }
  c64::require(sums.eof() && files.contains(root / "plan.tsv"), "plan absent du manifeste");
  auto plan = c64::balanced::load_plan(root, metal, package.model);
  for (const auto& [_, k] : plan.kernels) c64::require(files.contains(k.file), "SPIR-V hors du manifeste");
  for (const auto& [_, v] : plan.views)
    c64::require(v.file.empty() || files.contains(v.file), "poids hors du manifeste");
  return plan;
}

inline vk::Done prepare_model(const ModelSpec& model, std::span<const vk::net::InputShape> inputs,
                              const fs::path& out, Progress& progress) {
  if (progress.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "Slang");
  if (model.precision != TensorType::F16 || inputs.size() != 1 ||
      !std::ranges::equal(inputs[0].dims, kX.dims))
    return refuse(RefusalCode::FormatUnsupported, "Slang demande SPAN FP16 1920×1080");
  const auto package = std::ranges::find(kPackages, model.sha256, &Package::model);
  if (package == std::end(kPackages))
    return refuse(RefusalCode::FormatUnsupported, "Modèle SPAN absent du paquet Slang");
  auto sha = prepare::sha256_file(model.onnx_path);
  if (!sha) return std::unexpected(sha.error());
  if (*sha != package->model)
    return refuse(RefusalCode::FormatUnsupported, "Empreinte du modèle Slang incorrecte");
  return core::write_text_atomically(out, package->manifest);
}

struct Selection {
  const Package* package;
  vk::net::IoNames names;
};

inline std::expected<Selection, Refusal> select_artifact(const fs::path& artifact) {
  std::ifstream token(artifact);
  std::string digest;
  if (!std::getline(token, digest))
    return refuse(RefusalCode::CacheIo, "Le paquet de shaders Slang a changé");
  const auto package = std::ranges::find(kPackages, digest, &Package::manifest);
  if (package == std::end(kPackages))
    return refuse(RefusalCode::CacheIo, "Le paquet de shaders Slang a changé");
  auto names = vk::net::read_io_names(artifact);
  if (!names) return std::unexpected(names.error());
  return Selection{package, std::move(*names)};
}
}  // namespace kirari::gpu::slang
