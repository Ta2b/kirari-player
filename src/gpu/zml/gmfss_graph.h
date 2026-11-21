#pragma once

#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
enum class GmfssGraph { Other, Pair, PairWideWeights, BaseFusion };
GmfssGraph gmfss_graph(const ir::Graph& graph);
// Même précision que lower_onnx pour les corrélations GMFlow. Les Cast explicites vers FP16 sont conservés.
void widen_gmflow_intermediates(ir::Graph& graph);
}  // namespace kirari::gpu::zml
