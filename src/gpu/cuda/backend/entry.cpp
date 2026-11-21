// Points d'entrée de kirari_backend_tensorrt.so (gpu/backend.h). Aucune exception ne traverse la
// frontière C : un échec devient nullptr et un refus dans init->refusal.

#include <filesystem>
#include <format>
#include <mutex>
#include <string>

#include "core/dynamic_library.h"
#include "core/thread.h"
#include "gpu/backend.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda_mode.h"
#include "gpu/guarded_backend.h"
#include "gpu/module_logs.h"
#include "prepare/trtexec_command.h"

namespace {

// trtexec livré à côté de la bibliothèque, d'empreinte vérifiée ; à défaut, dans un build de
// développement seulement, celui trouvé à la configuration (KIRARI_TRTEXEC_PATH).
std::filesystem::path pick_trtexec_path() {
#ifdef _WIN32
  constexpr const char* kTrtexec = "trtexec.exe";
#else
  constexpr const char* kTrtexec = "trtexec";
#endif
#ifdef KIRARI_TRTEXEC_NO_FALLBACK
  const std::filesystem::path fallback;
#else
  const std::filesystem::path fallback = KIRARI_TRTEXEC_PATH;
#endif
  const std::filesystem::path self =
      kirari::library_containing(reinterpret_cast<const void*>(&pick_trtexec_path));
  return kirari::prepare::pick_trtexec(self.empty() ? std::filesystem::path{} : self.parent_path() / kTrtexec,
                                       KIRARI_TRTEXEC_SHA256, fallback);
}

// Le même, cherché une fois par processus : pick_trtexec hache trtexec (plusieurs Mo), et chaque instance du
// backend le redemandait au démarrage (B21-05). Un chemin vide ne se garde pas : trtexec peut arriver après.
std::filesystem::path trtexec_path() {
  static std::mutex lock;
  static std::filesystem::path picked;
  const std::scoped_lock guard(lock);
  if (picked.empty()) picked = pick_trtexec_path();
  return picked;
}

void tell(const kirari::gpu::BackendInit* init, kirari::Refusal refusal) noexcept {
  if (init != nullptr && init->refusal != nullptr) *init->refusal = std::move(refusal);
}

}  // namespace

extern "C" KIRARI_BACKEND_EXPORT int kirari_backend_abi() noexcept {
  return kirari::gpu::kBackendAbi;
}

extern "C" KIRARI_BACKEND_EXPORT std::uint64_t kirari_backend_layout() noexcept {
  return kirari::gpu::kBackendLayout;
}

extern "C" KIRARI_BACKEND_EXPORT std::uint32_t kirari_backend_cuda_modes() noexcept {
  return kirari::gpu::kCudaInferenceModes;
}

extern "C" KIRARI_BACKEND_EXPORT kirari::gpu::Backend* kirari_backend_create(
    int abi, const kirari::gpu::BackendInit* init) noexcept try {
  if (init == nullptr) return nullptr;
  if (abi != kirari::gpu::kBackendAbi) {
    tell(init,
         {kirari::RefusalCode::AbiMismatch,
          std::format("backend tensorrt : ABI {} demandée, {} fournie", abi, kirari::gpu::kBackendAbi)});
    return nullptr;
  }
  // Les canaux de la bibliothèque (prepare, tensorrt) vers le journal du lecteur, comme pour Vulkan.
  kirari::gpu::route_module_logs(init->log);
  auto backend = kirari::gpu::cuda::TensorRtBackend::create(*init, &trtexec_path);
  if (!backend) {
    tell(init, std::move(backend.error()));
    return nullptr;
  }
  return new kirari::gpu::GuardedBackend(std::move(*backend));  // NOLINT(cppcoreguidelines-owning-memory)
} catch (...) {
  tell(init, kirari::refusal_from_exception(kirari::RefusalCode::LibraryUnavailable, "backend tensorrt"));
  return nullptr;
}

extern "C" KIRARI_BACKEND_EXPORT void kirari_backend_destroy(kirari::gpu::Backend* backend) noexcept {
  delete backend;
}
