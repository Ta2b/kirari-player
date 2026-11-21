#include "gpu/hip/session.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <hip/hip_runtime_api.h>
#include <stdexcept>

#if !defined(KIRARI_HIP_HOST_STAGING)
#include "gpu/hip/external_memory.h"
#endif

#include "core/dynamic_library.h"
#include "gpu/hip/geometry.h"
#include "gpu/vulkan/backend/net_build.h"

namespace kirari::gpu::hip_backend {
namespace fs = std::filesystem;

namespace {
#define KIRARI_HIP_API_LIST(X)   \
  X(hipGetErrorString)           \
  X(hipGetDeviceCount)           \
  X(hipDeviceGetUuid)            \
  X(hipDeviceGetPCIBusId)        \
  X(hipGetDevicePropertiesR0600) \
  X(hipGetDevice)                \
  X(hipSetDevice)                \
  X(hipStreamCreateWithFlags)    \
  X(hipStreamSynchronize)        \
  X(hipStreamDestroy)            \
  X(hipModuleLoadData)           \
  X(hipModuleUnload)             \
  X(hipModuleGetFunction)        \
  X(hipModuleLaunchKernel)       \
  X(hipFree)                     \
  X(hipMemcpy)
class RocmSession final : public Session, public std::enable_shared_from_this<RocmSession> {
 public:
  explicit RocmSession(DynamicLibrary);
  ~RocmSession() override;
  vk::Done open(const vk::DeviceCaps&, const fs::path&);
  void check(hipError_t, std::string_view) const;
  void select() const { check(hipSetDevice(device), "hipSetDevice"); }

  std::string_view architecture() const override { return arch; }
  vk::Done in_context(const std::function<void()>& work) override {
    int previous{};
    check(hipGetDevice(&previous), "hipGetDevice");
    select();
    struct Restore {
      decltype(hipSetDevice) select;
      int previous;
      ~Restore() { static_cast<void>(select(previous)); }
    } restore{hipSetDevice, previous};
    work();
    return {};
  }
  void* allocate(std::size_t bytes, std::string_view name) override {
    void* pointer{};
    check(hipMalloc(&pointer, bytes), name);
    return pointer;
  }
  void release(void* p) noexcept override { static_cast<void>(hipFree(p)); }
  void upload(void* dst, const void* src, std::size_t size, std::string_view name) override {
    check(hipMemcpy(dst, src, size, hipMemcpyHostToDevice), name);
  }
  void download(void* dst, const void* src, std::size_t size) override {
    check(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost), "copie de la sortie HIP");
  }
  void launch(Kernel which, unsigned x, unsigned y, std::span<void*> args) override {
    check(hipModuleLaunchKernel(functions[which], x, y, 1, kWorkers, 1, 1, 0, stream, args.data(), nullptr),
          "hipModuleLaunchKernel");
  }
  void synchronize() override { check(hipStreamSynchronize(stream), "hipStreamSynchronize"); }
#if !defined(KIRARI_HIP_HOST_STAGING)
  bool shares_memory() const override { return bool(external_); }
  std::expected<vk::net::DeviceMemory, Refusal> import_memory(std::uintptr_t, std::size_t,
                                                              std::size_t) override;
#else
  bool shares_memory() const override { return false; }
#endif

  int device = -1;
  std::string arch;
  hipStream_t stream{};
  std::array<hipFunction_t, Count> functions{};
  DynamicLibrary library;
  hipError_t (*hipMalloc)(void**, std::size_t){};
#define HIP_FN(name) decltype(&::name) name{};
  KIRARI_HIP_API_LIST(HIP_FN)
#undef HIP_FN

 private:
#if !defined(KIRARI_HIP_HOST_STAGING)
  hip_interop::ExternalMemoryApi external_;
#endif
  hipModule_t module_{};
};

RocmSession::RocmSession(DynamicLibrary handle) : library(std::move(handle)) {
  const auto required = [&]<class Fn>(Fn*& slot, const char* name) {
    auto loaded = library.symbol<Fn>(name);
    if (!loaded) throw std::runtime_error(loaded.error().detail);
    slot = *loaded;
  };
  required(hipMalloc, "hipMalloc");
#define LOAD_FN(name) required(name, #name);
  KIRARI_HIP_API_LIST(LOAD_FN)
#undef LOAD_FN
#if !defined(KIRARI_HIP_HOST_STAGING)
  external_ = hip_interop::ExternalMemoryApi::load(library);
#endif
}

RocmSession::~RocmSession() {
  if (device < 0 || hipSetDevice(device) != hipSuccess) return;
  if (stream) {
    static_cast<void>(hipStreamSynchronize(stream));
    static_cast<void>(hipStreamDestroy(stream));
  }
  if (module_) static_cast<void>(hipModuleUnload(module_));
}

void RocmSession::check(hipError_t status, std::string_view label) const {
  if (status != hipSuccess)
    throw std::runtime_error(std::format("{} : {}", label, hipGetErrorString(status)));
}

bool same_device(const RocmSession& s, const vk::DeviceCaps& caps, int candidate) {
  if (caps.pci) {
    const auto& pci = *caps.pci;
    const auto expected =
        std::format("{:04x}:{:02x}:{:02x}.{:x}", pci.domain, pci.bus, pci.device, pci.function);
    char address[32]{};
    if (s.hipDeviceGetPCIBusId(address, sizeof(address), candidate) == hipSuccess) return expected == address;
  }
  hipUUID uuid{};
  return std::ranges::any_of(caps.uuid, [](auto value) { return value != 0; }) &&
         s.hipDeviceGetUuid(&uuid, candidate) == hipSuccess &&
         std::memcmp(uuid.bytes, caps.uuid.data(), caps.uuid.size()) == 0;
}

vk::Done RocmSession::open(const vk::DeviceCaps& caps, const fs::path& folder) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices <= 0)
    return refuse_hip(RefusalCode::DeviceUnavailable, "aucun périphérique exposé par le runtime ROCm");
  for (int i = 0; i < devices; ++i) {
    if (!same_device(*this, caps, i)) continue;
    device = i;
    break;
  }
  if (device < 0) return refuse_hip(RefusalCode::DeviceMismatch, "Vulkan et HIP n'exposent pas le même GPU");
  select();
  hipDeviceProp_t prop{};
  check(hipGetDevicePropertiesR0600(&prop, device), "hipGetDeviceProperties");
  const auto end = std::ranges::find(prop.gcnArchName, '\0');
  arch.assign(std::begin(prop.gcnArchName), end);
  if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
  if (arch.size() < 4 || !arch.starts_with("gfx") || !std::ranges::all_of(arch.substr(3), [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
      }))
    return refuse_hip(RefusalCode::DeviceUnavailable, "cible GCN illisible");
  const auto code = folder / ("span-" + arch + ".hsaco");
  if (!fs::is_regular_file(code))
    return refuse_hip(RefusalCode::DeviceUnavailable, std::format("aucun noyau HIP livré pour {}", arch));
  auto binary = vk::net::read_file(code);
  if (!binary) return std::unexpected(binary.error());
  if (binary->empty()) return refuse_hip(RefusalCode::FileUnreadable, "HSACO vide");
  check(hipModuleLoadData(&module_, binary->data()), "hipModuleLoadData");
  for (std::size_t i = 0; i < names.size(); ++i)
    check(hipModuleGetFunction(&functions[i], module_, names[i]), names[i]);
  check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
  return {};
}

#if !defined(KIRARI_HIP_HOST_STAGING)
std::expected<vk::net::DeviceMemory, Refusal> RocmSession::import_memory(std::uintptr_t fd,
                                                                         std::size_t allocation,
                                                                         std::size_t bytes) try {
  return hip_interop::import_buffer(
      external_,
      [self = shared_from_this()](const auto& work) {
        const std::scoped_lock lock(self->sequence);
        return self->in_context(work);
      },
      fd, allocation, bytes);
} catch (const std::exception& e) {
  return refuse_hip(RefusalCode::FeatureUnavailable, e.what());
}

#endif
}  // namespace

std::expected<std::shared_ptr<Session>, Refusal> make_rocm(const vk::DeviceCaps& caps,
                                                           const fs::path& folder) {
#if !defined(KIRARI_HIP_HOST_STAGING)
  if (!std::ranges::contains(caps.extensions, std::string_view(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)))
    return refuse_hip(RefusalCode::FeatureUnavailable, "Vulkan n'expose pas la mémoire GPU externe FD");
#endif
#ifdef _WIN32
  constexpr char runtime[] = "amdhip64_7.dll";
  const auto local_runtime = folder / runtime;
  auto api = fs::is_regular_file(local_runtime) ? DynamicLibrary::open(local_runtime)
                                                : DynamicLibrary::open_system(runtime);
#else
  auto api = DynamicLibrary::open_system("libamdhip64.so.7");
#endif
  if (!api) return std::unexpected(api.error());
  auto session = std::make_shared<RocmSession>(std::move(*api));
  if (auto opened = session->open(caps, folder); !opened) return std::unexpected(opened.error());
  return session;
}

}  // namespace kirari::gpu::hip_backend
