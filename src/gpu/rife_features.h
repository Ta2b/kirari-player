#pragma once

namespace kirari::gpu {
// Indication valable pour une seule soumission. Le lecteur connaît les identités des deux
// images et le facteur courant ; un moteur sans cache utilise le contrat Engine ordinaire.
enum class RifeFeatureRequest { Disabled, NewPair, SamePair, Following };
class RifeFeatureHints {
 public:
  virtual ~RifeFeatureHints() = default;
  virtual void next_rife_features(RifeFeatureRequest request) = 0;
};
}  // namespace kirari::gpu
