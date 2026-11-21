#pragma once

#include <expected>

#include "gpu/backend.h"

namespace kirari::gpu {

// Extension d'un moteur DistilDRBA dont le programme chargé ne lit que les plans RGB 3 à 5
// de pair12. Sans cette extension, le lecteur prépare normalement les sept plans.
class DrbaPair12Writer {
 public:
  virtual ~DrbaPair12Writer() = default;
  virtual std::expected<void, Refusal> write_third_image(const Tensor& image, const Tensor& pair12) = 0;
};

}  // namespace kirari::gpu
