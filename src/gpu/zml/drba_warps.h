#pragma once
#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
// Réunit le RGB et les caractéristiques d'un côté lorsque les deux grilles sont identiques.
int combine_drba_warps(ir::Graph& graph);
}  // namespace kirari::gpu::zml
