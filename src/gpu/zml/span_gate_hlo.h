#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace kirari::gpu::zml {
// XlaTransform fournit un xla.HloModuleProto. Un module étranger renvoie un vecteur vide.
// Les régions sont indivisibles : si la forme ou la provenance change, le lecteur reprend le dense.
std::expected<std::vector<std::byte>, std::string> rewrite_span_triton_gates(
    std::span<const std::byte> module);
}  // namespace kirari::gpu::zml
