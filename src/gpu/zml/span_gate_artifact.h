#pragma once

#include "gpu/zml/pjrt.h"
#include "gpu/zml/span_gate_profile.h"

namespace kirari::gpu::zml {
// Prépare côte à côte le module aux portes fusionnées et son équivalent portable aux mêmes poids.
std::expected<bool, Refusal> emit_span_gates(std::vector<std::string> arguments,
                                             const std::filesystem::path& staged,
                                             const SpanGateProfile& profile, bool fused_projection,
                                             Progress& progress);
std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> compile_span_or_portable(
    const std::shared_ptr<Client>& client, std::span<const std::byte> source, const vk::net::IoNames& names,
    const std::filesystem::path& path, Progress* preparation = nullptr);
}  // namespace kirari::gpu::zml
