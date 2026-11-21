#pragma once
#include <cstdint>
#include <memory>

namespace kirari::media {
// Événement GPU à attendre avant de lire l'image (CUevent du flux de décodage), avec ce
// qui le garde en vie. handle 0 : rien à attendre. Une image Vulkan porte son propre
// sémaphore timeline : to_decoded_frame le lit dans l'AVVkFrame.
struct ReadyEvent {
  std::uintptr_t handle = 0;
  std::uint64_t value = 0;  // Vulkan : valeur du sémaphore timeline ; 0 pour un CUevent
  std::shared_ptr<const void> owner;
};

}  // namespace kirari::media
