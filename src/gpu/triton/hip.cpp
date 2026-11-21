#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "core/dynamic_library.h"
#include "gpu/hip/external_memory.h"
#include "gpu/triton/device.h"

// HIP déclare des macros d'attributs, après lecture des en-têtes C++.
#include <hip/hip_runtime_api.h>

namespace kirari::gpu::triton {
namespace {
void check(hipError_t status, std::string_view action) {
  if (status != hipSuccess)
    throw std::runtime_error(std::format("{} : HIP ({})", action, static_cast<int>(status)));
}

template <class Fn>
Fn* required(const DynamicLibrary& library, const char* name) {
  auto symbol = library.symbol<Fn>(name);
  if (!symbol) throw std::runtime_error(symbol.error().detail);
  return *symbol;
}

struct Api {
  explicit Api(DynamicLibrary library) : lib(std::move(library)) {}
  DynamicLibrary lib;
  decltype(hipInit)* init = required<decltype(hipInit)>(lib, "hipInit");
  decltype(hipGetDeviceCount)* count = required<decltype(hipGetDeviceCount)>(lib, "hipGetDeviceCount");
  decltype(hipDeviceGetUuid)* uuid = required<decltype(hipDeviceGetUuid)>(lib, "hipDeviceGetUuid");
  decltype(hipDeviceGetByPCIBusId)* pci =
      required<decltype(hipDeviceGetByPCIBusId)>(lib, "hipDeviceGetByPCIBusId");
  decltype(hipGetDeviceProperties)* properties =
      required<decltype(hipGetDeviceProperties)>(lib, "hipGetDevicePropertiesR0600");
  decltype(hipGetDevice)* current = required<decltype(hipGetDevice)>(lib, "hipGetDevice");
  decltype(hipSetDevice)* select = required<decltype(hipSetDevice)>(lib, "hipSetDevice");
  using Malloc = hipError_t(void**, std::size_t);
  Malloc* alloc = required<Malloc>(lib, "hipMalloc");
  decltype(hipFree)* free = required<decltype(hipFree)>(lib, "hipFree");
  decltype(hipMemcpy)* copy = required<decltype(hipMemcpy)>(lib, "hipMemcpy");
  decltype(hipStreamCreateWithFlags)* create =
      required<decltype(hipStreamCreateWithFlags)>(lib, "hipStreamCreateWithFlags");
  decltype(hipStreamDestroy)* destroy = required<decltype(hipStreamDestroy)>(lib, "hipStreamDestroy");
  decltype(hipStreamSynchronize)* sync =
      required<decltype(hipStreamSynchronize)>(lib, "hipStreamSynchronize");
  decltype(hipModuleLoadData)* load = required<decltype(hipModuleLoadData)>(lib, "hipModuleLoadData");
  decltype(hipModuleUnload)* unload = required<decltype(hipModuleUnload)>(lib, "hipModuleUnload");
  decltype(hipModuleGetFunction)* function =
      required<decltype(hipModuleGetFunction)>(lib, "hipModuleGetFunction");
  decltype(hipModuleLaunchKernel)* launch =
      required<decltype(hipModuleLaunchKernel)>(lib, "hipModuleLaunchKernel");
};

std::expected<DynamicLibrary, Refusal> open_hip() {
#ifdef _WIN32
  constexpr const char* names[]{"amdhip64_7.dll", "amdhip64_6.dll"};
#else
  constexpr const char* names[]{"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"};
#endif
  for (const auto* name : names) {
    if (auto library = DynamicLibrary::open_system(name); library) return library;
  }
  return refuse(RefusalCode::LibraryUnavailable, "Triton : runtime AMD HIP 6 ou 7 introuvable");
}

std::expected<int, Refusal> ordinal(const vk::DeviceCaps& caps, const Api& api) {
  int count{};
  check(api.count(&count), "hipGetDeviceCount");
  const bool valid_uuid = std::ranges::any_of(caps.uuid, [](auto b) { return b != 0; });
  for (int i = 0; valid_uuid && i < count; ++i) {
    hipUUID id{};
    if (api.uuid(&id, i) == hipSuccess && std::memcmp(id.bytes, caps.uuid.data(), caps.uuid.size()) == 0)
      return i;
  }
  if (caps.pci) {
    const auto& p = *caps.pci;
    const auto address = std::format("{:04x}:{:02x}:{:02x}.{:x}", p.domain, p.bus, p.device, p.function);
    int matched{};
    if (api.pci(&matched, address.c_str()) == hipSuccess && matched >= 0 && matched < count) return matched;
  }
  return refuse(RefusalCode::DeviceMismatch, "Triton : identité du GPU Vulkan absente de HIP");
}

class HipDevice final : public Device, public std::enable_shared_from_this<HipDevice> {
 public:
  HipDevice(std::shared_ptr<Api> api, int ordinal, const vk::DeviceCaps& caps)
      : api_(std::move(api)), ordinal_(ordinal) {
#if defined(__linux__)
    if (std::ranges::contains(caps.extensions, std::string_view(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)))
      external_ = hip_interop::ExternalMemoryApi::load(api_->lib);
#else
    static_cast<void>(caps);
#endif
  }
  ~HipDevice() override {
    if (!stream_) return;
    std::ignore = in_context([&] {
      static_cast<void>(api_->sync(stream_));
      static_cast<void>(api_->destroy(stream_));
    });
  }
  vk::Done open() {
    return in_context(
        [&] { check(api_->create(&stream_, hipStreamNonBlocking), "hipStreamCreateWithFlags"); });
  }
  std::string_view architecture() const override { return TRITON_PACKAGE_HIP_ARCH; }
  vk::Done in_context(const std::function<void()>& work) override {
    int previous{};
    try {
      check(api_->current(&previous), "hipGetDevice");
      check(api_->select(ordinal_), "hipSetDevice");
    } catch (const std::exception& e) {
      return refuse(RefusalCode::VendorCallFailed, e.what());
    }
    try {
      work();
    } catch (const std::exception& e) {
      static_cast<void>(api_->select(previous));
      return refuse(RefusalCode::VendorCallFailed, "Triton HIP : " + std::string(e.what()));
    }
    if (api_->select(previous) != hipSuccess)
      return refuse(RefusalCode::VendorCallFailed, "Triton HIP : restauration du périphérique");
    return {};
  }
  Pointer allocate(std::size_t size, std::string_view name) override {
    void* ptr{};
    check(api_->alloc(&ptr, size), name);
    return reinterpret_cast<Pointer>(ptr);
  }
  void release(Pointer ptr) override { static_cast<void>(api_->free(native_handle<void*>(ptr))); }
  void upload(Pointer ptr, const void* data, std::size_t size, std::string_view label) override {
    check(api_->copy(native_handle<void*>(ptr), data, size, hipMemcpyHostToDevice), label);
  }
  void download(void* data, Pointer ptr, std::size_t size) override {
    check(api_->copy(data, native_handle<const void*>(ptr), size, hipMemcpyDeviceToHost), "sortie HIP");
  }
  Module load(std::span<const char> data, std::string_view name, unsigned) override {
    hipModule_t mod{};
    check(api_->load(&mod, data.data()), name);
    try {
      hipFunction_t function{};
      check(api_->function(&function, mod, std::string(name).c_str()), name);
      return {reinterpret_cast<Pointer>(mod), reinterpret_cast<Pointer>(function)};
    } catch (...) {
      static_cast<void>(api_->unload(mod));
      throw;
    }
  }
  void unload(Pointer mod) override { static_cast<void>(api_->unload(native_handle<hipModule_t>(mod))); }
  void launch(const Kernel& k, std::span<void*> args) override {
    check(api_->launch(native_handle<hipFunction_t>(k.function), k.grid, k.grid_y, k.grid_z, k.block, 1, 1,
                       k.shared, stream_, args.data(), nullptr),
          "hipModuleLaunchKernel");
  }
  void synchronize() override { check(api_->sync(stream_), "hipStreamSynchronize"); }

#if defined(__linux__)
  bool shares_device_memory() const override { return bool(external_); }
  std::expected<vk::net::DeviceMemory, Refusal> import_memory(Pointer fd, std::size_t allocation,
                                                              std::size_t bytes) override {
    // HIP importe la mémoire. Sur cette voie, l'ordre entre Vulkan et HIP vient de l'étape
    // différée du lecteur et de hipStreamSynchronize, sans sémaphore HIP importé.
    return hip_interop::import_buffer(
        external_,
        [self = shared_from_this()](const auto& work) {
          const std::scoped_lock lock(self->sequence);
          return self->in_context(work);
        },
        fd, allocation, bytes);
  }
#endif

 private:
  std::shared_ptr<Api> api_;
  int ordinal_;
  hipStream_t stream_{};
#if defined(__linux__)
  hip_interop::ExternalMemoryApi external_;
#endif
};
}  // namespace

std::expected<std::shared_ptr<Device>, Refusal> make_hip(const vk::DeviceCaps& caps) {
  auto library = open_hip();
  if (!library) return std::unexpected(library.error());
  auto api = std::make_shared<Api>(std::move(*library));
  check(api->init(0), "hipInit");
  const auto device = ordinal(caps, *api);
  if (!device) return std::unexpected(device.error());
  hipDeviceProp_t props{};
  check(api->properties(&props, *device), "hipGetDeviceProperties");
  const std::string_view arch(
      props.gcnArchName,
      static_cast<std::size_t>(std::ranges::find(props.gcnArchName, '\0') - props.gcnArchName));
  if (arch.substr(0, arch.find(':')) != TRITON_PACKAGE_HIP_ARCH)
    return refuse(RefusalCode::DeviceMismatch, "Triton : paquet ROCm absent pour " + std::string(arch));
  auto result = std::make_shared<HipDevice>(std::move(api), *device, caps);
  if (auto ok = result->open(); !ok) return std::unexpected(ok.error());
  return result;
}
}  // namespace kirari::gpu::triton
