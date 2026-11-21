#pragma once
#include <filesystem>

#include "core/gpu_device.h"

namespace kirari::gpu::zml {
// Prépare les outils du SDK livré avant son dlopen, comme le chargeur amont de ZML.
void prepare_platform(GpuVendor vendor, const std::filesystem::path& plugin);
}  // namespace kirari::gpu::zml
