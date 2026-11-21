#pragma once

#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
// Réunit les trois encodeurs à poids partagés de DistilDRBA sur l'axe du lot.
// Le graphe d'entrée est déjà de forme fixe, ses constantes pliées.
bool optimize_drba_graph(ir::Graph& graph);
}  // namespace kirari::gpu::zml
