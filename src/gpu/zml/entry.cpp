#include "gpu/vulkan/backend/entry_points.h"
#include "gpu/zml/runtime.h"

namespace {
kirari::gpu::vk::BackendOptions zml_options() {
  return {.runtime = &kirari::gpu::zml::make_runtime, .partial_runtime = true};
}
}  // namespace
KIRARI_VULKAN_BACKEND_ENTRY_POINTS("ZML", zml_options)
