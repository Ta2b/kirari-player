// Encodeur DistilDRBA en lots de trois, pour le plan 1920×1088 à réduction 4.
#pragma once

#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda {

class DrbaBatchEncoder {
 public:
  virtual ~DrbaBatchEncoder() = default;

  // Construit le plan fixe et importe les poids ONNX. Aucun moteur de référence.
  static std::unique_ptr<DrbaBatchEncoder> create(std::span<const RifeNativeConv> model, cudaError_t& error);
  static std::size_t scratch_bytes();

  // Entrées RGB planaires 480×272. Sorties encodées planaires aux échelles 1, 1/2 et 1/4,
  // trois images de 16 canaux chacune. Les pointeurs désignent le premier plan encodé,
  // après les neuf plans RGB des trois images.
  virtual cudaError_t enqueue(cudaStream_t stream, const void* input, void* full, void* half, void* quarter,
                              void* scratch) const = 0;
};

}  // namespace kirari::gpu::cuda
