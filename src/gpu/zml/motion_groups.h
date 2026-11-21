#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "gpu/ir/ir.h"

namespace kirari::gpu::zml {
struct MotionGroup {
  std::string op;
  std::vector<ir::TensorId> inputs, outputs;
  ir::Dims integers;
  std::vector<std::size_t> members;
};

// Motifs dont tous les intermédiaires sont privés. Les membres sont donnés en ordre topologique.
std::vector<MotionGroup> motion_groups(const ir::Graph& graph);
struct MotionRecipe {
  bool transposed = false;
  bool resize = false;
  bool d2_centre = false;
};
enum class RifeD2Centre { Disabled, Auto422, Experimental };
struct RifeRecipes {
  bool lite422 = false;
  bool full425 = false;
  bool features = false;
  RifeD2Centre d2 = RifeD2Centre::Disabled;
  bool lite425_d4 = false;
};
// Renseigne les attributs des groupes et renvoie ceux des opérateurs tracés individuellement.
MotionRecipe configure_motion_groups(const ir::Graph& graph, std::vector<MotionGroup>& groups,
                                     RifeRecipes recipes);
std::optional<MotionGroup> rife425_lite_d4(const ir::Graph& graph, const std::vector<MotionGroup>& groups);
std::optional<MotionGroup> rife_d2_centre(const ir::Graph& graph, const std::vector<MotionGroup>& groups,
                                          bool resize_phases);
#ifdef KIRARI_ZML_FOLD_RESCONV
// À la préparation du manifeste SDK, après reconnaissance et sélection des phases.
void fold_motion_residuals(ir::Graph& graph, std::vector<MotionGroup>& groups);
#endif
// Les quatre diffusions du dérivé de banc à sept plans (grille_rife.py), si sa topologie correspond.
std::vector<ir::TensorId> compact_rife_grid(const ir::Graph& graph);
#ifdef KIRARI_ZML_RIFE422_WARP_XY
ir::TensorId rife422_half_warp(const ir::Graph& graph, const std::vector<MotionGroup>& groups);
#endif
}  // namespace kirari::gpu::zml
