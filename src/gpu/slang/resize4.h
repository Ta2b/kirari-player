#pragma once

#include "gpu/vulkan/backend/net_engine.h"

namespace kirari::gpu::slang {

std::span<const std::uint32_t> resize4_dispatch(vk::net::Dispatch& dispatch, const vk::net::Target& target);

}  // namespace kirari::gpu::slang
