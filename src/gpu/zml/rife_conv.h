#pragma once

#include "gpu/zml/hlo.h"

namespace kirari::gpu::zml {
// Dernière tête de mouvement : ne calcule que les canaux lus après le réarrangement spatial.
void prune_motion_heads(ir::Graph& graph);

#ifdef KIRARI_ZML_POLYPHASE
// Transposée 4×4, pas 2, bord 1 : convolution dense 2×2 sur la grille d'entrée.
Value transpose_conv_4x4(Hlo& hlo, const Value& image, const Value& weights);
#endif
}  // namespace kirari::gpu::zml
