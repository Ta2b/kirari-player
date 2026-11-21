#pragma once
#include "gpu/vulkan/backend/net_runtime.h"

namespace kirari::gpu::zml {
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps);
}
