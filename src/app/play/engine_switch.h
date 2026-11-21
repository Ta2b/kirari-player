// L'onglet Moteurs : SPAN et RIFE passent sur un autre moteur installé sans relancer le lecteur.
// Le moteur choisi devient le préféré des réseaux (settings.toml) ; sa bibliothèque s'essaie alors
// en premier, et Vulkan reste le recours si elle refuse le GPU.
#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "core/engines.h"
#include "core/settings.h"

namespace kirari::app::play {

// Le moteur d'une bibliothèque de backend, par son nom (« kirari_backend_tensorrt.so ») ; rien
// pour un fichier qui n'en est pas une.
std::optional<core::EngineId> engine_of_library(const std::filesystem::path& library);
// Le mode CUDA natif partage le module du runtime TensorRT.
std::uint32_t cuda_inference_mode(const core::Settings& engines);
std::optional<core::EngineId> selected_engine(const std::filesystem::path& library,
                                              const core::Settings& settings);

// Un module local et ses ressources sont présents pour ce fabricant.
bool backend_available(const std::filesystem::path& path, GpuVendor vendor);
// La bibliothèque d'un moteur retiré (MIGraphX, OpenVINO) laissée par une ancienne version.
bool retired_library(const std::filesystem::path& library);

// Les moteurs des réseaux dont la bibliothèque est parmi candidates, sans doublon, dans l'ordre de
// l'onglet : Vulkan, TensorRT, Core ML.
std::vector<core::EngineId> switchable_engines(std::span<const std::filesystem::path> candidates);

// settings avec id choisi, installé et préféré pour SPAN et RIFE.
core::Settings using_engine(core::Settings settings, core::EngineId id);

// candidates, la bibliothèque du moteur préféré pour les réseaux en tête ; les autres gardent leur
// ordre.
std::vector<std::filesystem::path> preferred_first(std::vector<std::filesystem::path> candidates,
                                                   const core::Settings& engines);

}  // namespace kirari::app::play
