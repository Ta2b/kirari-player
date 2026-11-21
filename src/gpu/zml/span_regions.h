#pragma once

#include "gpu/zml/pjrt.h"

namespace kirari::gpu::zml {
// Sous le verrou de préparation, le réseau régional est sérialisé avant d'inscrire son empreinte dans le
// dense.
std::expected<void, Refusal> prepare_span_regions(std::shared_ptr<Client> client,
                                                  const std::filesystem::path& dense,
                                                  const std::vector<std::int64_t>& input, Progress& progress);
void load_span_regions(std::shared_ptr<Client> client, const std::filesystem::path& dense,
                       std::span<const std::byte> source, vk::net::RuntimeNet& full);
}  // namespace kirari::gpu::zml
