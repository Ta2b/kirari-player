// Charge un noyau AOT. Chaque plateforme définit le contexte et l'API GPU,
// le runtime commun garde l'ordre des lancements et la durée de vie des tampons.
#pragma once
#include <filesystem>
#include <mutex>
#include <tuple>
#include <vector>

#include "gpu/vulkan/backend/net_runtime.h"
#include "gpu/vulkan/compute/refuse.h"

namespace kirari::gpu::triton {
using Pointer = std::uintptr_t;

template <class Handle>
Handle native_handle(Pointer address) {
  // Le plan conserve les adresses et handles GPU dans un format commun.
  return reinterpret_cast<Handle>(address);  // NOLINT(performance-no-int-to-ptr)
}

struct Kernel {
  Pointer function{};
  unsigned grid{}, block{}, shared{};
  unsigned grid_y{1}, grid_z{1};
};
struct Module {
  Pointer handle{}, function{};
};

struct Launch {
  Kernel kernel;
  std::vector<void*> args;
};

// Créé et utilisé dans le contexte du périphérique, sous Device::sequence. Le
// propriétaire le détruit dans ce contexte, avant de décharger les modules.
class RecordedLaunch {
 public:
  virtual ~RecordedLaunch() = default;
  virtual void launch() = 0;
};

class Device {
 public:
  virtual ~Device() = default;
  std::mutex sequence;
  virtual std::string_view architecture() const = 0;
  virtual vk::Done in_context(const std::function<void()>& work) = 0;
  virtual Pointer allocate(std::size_t size, std::string_view label) = 0;
  virtual void release(Pointer ptr) = 0;
  virtual void upload(Pointer ptr, const void* data, std::size_t size, std::string_view label) = 0;
  virtual void download(void* data, Pointer ptr, std::size_t size) = 0;
  virtual Module load(std::span<const char> data, std::string_view name, unsigned shared) = 0;
  virtual void unload(Pointer module) = 0;
  virtual void launch(const Kernel& kernel, std::span<void*> args) = 0;
  virtual bool can_record() const { return false; }
  // Enregistre les valeurs des arguments. Un refus rend nullptr sans lancer de travail.
  virtual std::unique_ptr<RecordedLaunch> record(std::span<Launch>) { return nullptr; }
  virtual void synchronize() = 0;
  virtual bool shares_device_memory() const { return false; }
  virtual vk::Done import_timeline(int) {
    return refuse(RefusalCode::FeatureUnavailable, "Triton : sémaphore partagé indisponible");
  }
  virtual std::expected<vk::net::DeviceMemory, Refusal> import_memory(Pointer, std::size_t, std::size_t) {
    return refuse(RefusalCode::FeatureUnavailable, "Triton : mémoire GPU partagée indisponible");
  }
  virtual vk::Done wait(std::uint64_t) {
    return refuse(RefusalCode::FeatureUnavailable, "Triton : sémaphore partagé indisponible");
  }
  virtual vk::Done signal(std::uint64_t) {
    return refuse(RefusalCode::FeatureUnavailable, "Triton : sémaphore partagé indisponible");
  }
};

std::expected<std::shared_ptr<Device>, Refusal> make_cuda(const vk::DeviceCaps& caps);
std::expected<std::shared_ptr<Device>, Refusal> make_hip(const vk::DeviceCaps& caps);
std::expected<std::shared_ptr<Device>, Refusal> make_xpu(const vk::DeviceCaps& caps);
}  // namespace kirari::gpu::triton
