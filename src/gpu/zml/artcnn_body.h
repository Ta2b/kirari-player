#pragma once

#include "gpu/backend.h"
#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
// Deux modèles ONNX du catalogue, un corps appris de sept convolutions, une interface couleur FP32.
bool use_artcnn_half_body(ir::Graph& graph, const ModelSpec& model);
}  // namespace kirari::gpu::zml
