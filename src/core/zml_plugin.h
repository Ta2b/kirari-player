#pragma once
#include <filesystem>

#include "core/gpu_device.h"

namespace kirari {
// Ressource du module ou du dossier de données utilisateur sous Linux.
std::filesystem::path zml_runtime_asset(const std::filesystem::path& module,
                                        const std::filesystem::path& relative);
// Même recherche pour l'assistant et le moteur. Vide si ce GPU n'a pas de greffon installé.
std::filesystem::path zml_plugin_path(const std::filesystem::path& module, GpuVendor vendor);
}  // namespace kirari
