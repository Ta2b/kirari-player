#pragma once

#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/zml/motion_groups.h"

namespace kirari::gpu::zml {
struct DrbaRecipes {
  bool shared_projection = false;
  bool gather_fp16 = false;
  bool output_pixel = false;
};

struct GmfssRecipes {
  bool pair = false;
  bool mlp_strips = false;
  bool global_match = false;
};

std::expected<bool, Refusal> build_motion_with_sdk(const ModelSpec& model,
                                                   std::span<const vk::net::InputShape> inputs,
                                                   const std::filesystem::path& staged, RifeRecipes recipes,
                                                   DrbaRecipes drba, GmfssRecipes gmfss,
                                                   bool artcnn_mixed_body, Progress& progress);
}  // namespace kirari::gpu::zml
