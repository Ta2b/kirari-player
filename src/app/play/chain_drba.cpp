#include <format>
#include <utility>

#include "app/play/chain_setup.h"
#include "app/play/drba_fused_choice.h"
#include "app/play/i18n.h"
#include "core/clock.h"
#include "core/log.h"
#include "gpu/cuda_mode.h"
#include "gpu/rife_trials.h"

namespace kirari::app::play {
namespace {

// Ce que DistilDRBA chargé a pris : le fusionné, ou l'entier après un refus du fusionné (A03-21).
void note_drba_fused(ChainEngines& c, const DrbaFusedUse& fused, bool tried) {
  c.drba_fused = fused.used;
  c.drba_cuda_direct = fused.cuda_direct;
  c.drba_fused_refused = fused.refused;
  if (fused.cuda_direct)
    log_channel("app")->info("DistilDRBA : noyaux CUDA directs");
  else if (fused.used)
    log_channel("app")->info("DistilDRBA fusionné : quatre morceaux et noyaux de colle");
  else if (tried)
    log_channel("app")->warn("DistilDRBA fusionné indisponible ({}) : DistilDRBA entier", fused.declined);
}

// DistilDRBA fusionné (kernels/spec/drba_fused.h) : quatre petits moteurs et des noyaux de colle à la place
// du moteur entier, quand le backend le sait, que le réglage le veut et que le cran est Équilibre ; tout
// échec garde le moteur entier. Un refus retenu pour la séance (B08-02) le décline sans reconstruire.
DrbaFusedUse drba_fused_use(const ChainEngines& c, const PlaySettings& settings, InnerFallbacks& inner,
                            const std::string& fused_key) {
  DrbaFusedUse fused;
  fused.declined = drba_fused_declined(settings, c.backend->capabilities().drba_fused);
  if (settings.drba_fused && !fused.declined.empty())
    log_channel("app")->info("DistilDRBA fusionné non utilisé : {}", fused.declined);
  if (!fused.declined.empty()) return fused;
  if (const Refusal* before = inner.refused(ChainStage::Motion, fused_key)) {
    fused.declined = before->detail;
    fused.refused = before->code;
  }
  return fused;
}

// Le moteur DistilDRBA, la variante mouvement d'abord quand le warp suit SPAN ; pose drba_motion et
// drba_size.
std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> build_drba(Player& p,
                                                                               const gpu::BackendInit& init,
                                                                               ChainEngines& c,
                                                                               const PlaySettings& settings,
                                                                               DrbaFusedUse* fused) {
  // Warp après SPAN : la variante mouvement, un SPAN par source au lieu d'un par image créée.
  bool motion =
      settings.upscale && warp_after_span(settings.span_order, c.work.height * kSpanFactor, c.order_factor);
  // D4 : la variante mouvement estime à la taille de RIFE mouvement (720p pour un travail plus grand),
  // comme RIFE ; sans variante mouvement, l'image se refait à la taille de travail.
  const bool smaller = motion && c.rife_size.height < c.work.height;
  const WorkSize at = smaller ? c.rife_size : c.work;
  auto loaded = prepare_distil_drba(p, init, *c.backend, c.model_places, at.width, at.height, c.drba_align,
                                    settings.motion_precision, fused, &motion, smaller);
  // Sans variante mouvement, l'image se fait à la taille de travail : rien n'a été construit plus bas.
  if (smaller &&
      ((!loaded && loaded.error().key == kDrbaImageElsewhereKey) || (loaded && *loaded && !motion))) {
    if (fused != nullptr) *fused = DrbaFusedUse{.declined = fused->declined};
    loaded = prepare_distil_drba(p, init, *c.backend, c.model_places, c.work.width, c.work.height,
                                 c.drba_align, settings.motion_precision, fused, &motion);
  }
  c.drba_motion = loaded && *loaded && motion;
  c.drba_size = c.drba_motion ? at : c.work;
  if (c.drba_motion && smaller)
    log_channel("app")->info("DistilDRBA mouvement : estimé en {}x{}, warp en {}x{}", at.width, at.height,
                             c.work.width, c.work.height);
  return loaded;
}

}  // namespace

// La voie mixte peut garder RIFE si DistilDRBA est refusé. Le choix DistilDRBA propage le refus au lecteur.
// Faux : préparation annulée ; vrai : chargée ou refusée avec le repli de la voie mixte.
std::expected<bool, StageFailure> load_drba(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                            const PlaySettings& settings, InnerFallbacks& inner,
                                            const std::string& library) {
  const auto log = log_channel("app");
  const bool direct = (init.trials & gpu::kRifeCudaDirect) != 0 && gpu::is_cuda_backend(c.backend_name);
  const bool selected = settings.motion_engine == MotionEngine::DistilDrba;
  const std::string drba_key = inner_key(std::format("drba {}", name(settings.motion_precision)), library, {},
                                         c.work.width, c.work.height);
  if (const Refusal* before = inner.refused(ChainStage::Motion, drba_key)) {
    if (selected) return std::unexpected(StageFailure{.refusal = *before, .stage = ChainStage::Motion});
    log->info("DistilDRBA refusé plus tôt dans la séance : {}", c.rife_model ? "RIFE" : "sans interpolation");
    return true;
  }
  const std::string fused_key = inner_key("drba-fused", library, {}, c.work.width, c.work.height);
  DrbaFusedUse fused = direct ? DrbaFusedUse{} : drba_fused_use(c, settings, inner, fused_key);
  const bool try_fused = fused.declined.empty();
  auto loaded = build_drba(p, init, c, settings, try_fused ? &fused : nullptr);
  if (loaded && *loaded) {
    note_drba_fused(c, fused, try_fused);
    if (try_fused && fused.refused)
      inner.failed(ChainStage::Motion, fused_key, Refusal{*fused.refused, fused.declined});
  }
  // Annulée, la construction ne passe pas la main à RIFE : l'interpolation part, SPAN reste (A31-15).
  if (!loaded && (selected || watch_now(loaded.error()) || preparation_stopped(loaded.error())))
    return std::unexpected(StageFailure{.refusal = std::move(loaded.error()), .stage = ChainStage::Motion});
  if (!loaded) {
    log->warn("{} ; {}", loaded.error().detail, c.rife_model ? "RIFE à la place" : "sans interpolation");
    inner.failed(ChainStage::Motion, drba_key, loaded.error());
    p.ui.controls().warn(c.rife_model ? i18n::ui_tr("DistilDRBA unavailable: using RIFE")
                                      : i18n::ui_tr("DistilDRBA unavailable"),
                         host_now_us());
    return true;
  }
  if (!*loaded) return false;
  c.drba = std::move(**loaded);
  return true;
}

}  // namespace kirari::app::play
