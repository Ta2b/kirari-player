#include <algorithm>
#include <cuda.h>
#include <format>
#include <stdexcept>

#include "core/dynamic_library.h"
#include "gpu/hip/geometry.h"
#include "gpu/hip/session.h"
#include "gpu/vulkan/backend/net_build.h"
#include "media/cuda_driver.h"

namespace kirari::gpu::hip_backend {
namespace {
#define KIRARI_CUDA_CALLS(X)                      \
  X(cuGetErrorString, "cuGetErrorString")         \
  X(cuDeviceGetAttribute, "cuDeviceGetAttribute") \
  X(cuModuleLoadData, "cuModuleLoadData")         \
  X(cuModuleGetFunction, "cuModuleGetFunction")   \
  X(cuModuleUnload, "cuModuleUnload")             \
  X(cuLaunchKernel, "cuLaunchKernel")             \
  X(cuMemAlloc, "cuMemAlloc_v2")                  \
  X(cuMemFree, "cuMemFree_v2")                    \
  X(cuMemcpyHtoD, "cuMemcpyHtoD_v2")              \
  X(cuMemcpyDtoH, "cuMemcpyDtoH_v2")              \
  X(cuStreamSynchronize, "cuStreamSynchronize")

class CudaSession final : public Session {
 public:
  CudaSession(DynamicLibrary library, std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext ctx,
              bool shared)
      : library_(std::move(library)), driver_(std::move(driver)), ctx_(std::move(ctx)), shared_(shared) {
    const auto required = [&]<class F>(F*& slot, const char* name) {
      const auto fn = library_.symbol<F>(name);
      if (!fn) throw std::runtime_error(fn.error().detail);
      slot = *fn;
    };
#define LOAD(fn, name) required(fn, name);
    KIRARI_CUDA_CALLS(LOAD)
#undef LOAD
  }

  ~CudaSession() override {
    static_cast<void>(in_context([&] {
      if (stream_) static_cast<void>(cuStreamSynchronize(cuda_stream()));
      if (module_) static_cast<void>(cuModuleUnload(module_));
    }));
    if (stream_) driver_->destroy_stream(ctx_.context, stream_);
  }

  vk::Done open(int ordinal, const std::filesystem::path& folder) {
    int major{}, minor{};
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, ordinal),
          "capacité CUDA");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, ordinal),
          "capacité CUDA");
    architecture_ = std::format("sm{}{}", major, minor);
    const auto path = folder / ("span-" + architecture_ + ".cubin");
    if (!std::filesystem::is_regular_file(path))
      return refuse_hip(RefusalCode::DeviceUnavailable, "aucun CUBIN HIP livré pour " + architecture_);
    auto binary = vk::net::read_file(path);
    if (!binary) return std::unexpected(binary.error());
    if (binary->empty()) return refuse_hip(RefusalCode::FileUnreadable, "CUBIN HIP vide");
    auto stream = driver_->create_stream(ctx_.context);
    if (!stream) return std::unexpected(stream.error());
    stream_ = *stream;
    return in_context([&] {
      check(cuModuleLoadData(&module_, binary->data()), "charge le CUBIN HIP");
      for (std::size_t i = 0; i < names.size(); ++i)
        check(cuModuleGetFunction(&functions_[i], module_, names[i]), names[i]);
    });
  }

  std::string_view architecture() const override { return architecture_; }
  vk::Done in_context(const std::function<void()>& work) override {
    return driver_->in_context(ctx_.context, work);
  }
  void* allocate(std::size_t bytes, std::string_view name) override {
    CUdeviceptr ptr{};
    check(cuMemAlloc(&ptr, bytes), name);
    return reinterpret_cast<void*>(ptr);
  }
  void release(void* ptr) noexcept override {
    static_cast<void>(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
  }
  void upload(void* dst, const void* src, std::size_t size, std::string_view name) override {
    check(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, size), name);
  }
  void download(void* dst, const void* src, std::size_t size) override {
    check(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), size), "copie de la sortie CUDA");
  }
  void launch(Kernel which, unsigned x, unsigned y, std::span<void*> args) override {
    check(cuLaunchKernel(functions_[which], x, y, 1, kWorkers, 1, 1, 0, cuda_stream(), args.data(), nullptr),
          "exécution HIP sur CUDA");
  }
  void synchronize() override { check(cuStreamSynchronize(cuda_stream()), "cuStreamSynchronize"); }
  bool shares_memory() const override { return shared_; }
  std::expected<vk::net::DeviceMemory, Refusal> import_memory(std::uintptr_t handle, std::size_t allocation,
                                                              std::size_t bytes) override {
    auto imported = driver_->import_memory(ctx_.context, handle, allocation, 0, bytes, true);
    if (!imported) return std::unexpected(imported.error());
    return vk::net::DeviceMemory{imported->pointer, std::move(imported->owner)};
  }

 private:
  CUstream cuda_stream() const { return reinterpret_cast<CUstream>(stream_); }
  void check(CUresult status, std::string_view action) const {
    if (status == CUDA_SUCCESS) return;
    const char* reason = nullptr;
    static_cast<void>(cuGetErrorString(status, &reason));
    throw std::runtime_error(std::format("{} : {} ({})", action, reason ? reason : "CUDA", int(status)));
  }

  DynamicLibrary library_;
  std::shared_ptr<media::CudaDriver> driver_;
  media::PrimaryContext ctx_;
  bool shared_;
  std::string architecture_;
  std::uintptr_t stream_{};
  CUmodule module_{};
  std::array<CUfunction, Count> functions_{};
#define DECLARE(fn, name) decltype(&::fn) fn{};
  KIRARI_CUDA_CALLS(DECLARE)
#undef DECLARE
};
#undef KIRARI_CUDA_CALLS
}  // namespace

std::expected<std::shared_ptr<Session>, Refusal> make_cuda(const vk::DeviceCaps& caps,
                                                           const std::filesystem::path& folder) {
  auto driver = media::CudaDriver::load();
  if (!driver) return std::unexpected(driver.error());
  auto ordinal = (*driver)->ordinal_of(caps.uuid);
  if (!ordinal) return std::unexpected(ordinal.error());
  auto ctx = (*driver)->retain_primary(*ordinal);
  if (!ctx) return std::unexpected(ctx.error());
#ifdef _WIN32
  const bool shared = false;
  auto library = DynamicLibrary::open_system("nvcuda.dll");
#else
  const bool shared =
      std::ranges::contains(caps.extensions, std::string_view(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME));
  auto library = DynamicLibrary::open_system("libcuda.so.1");
#endif
  if (!library) return std::unexpected(library.error());
  auto session =
      std::make_shared<CudaSession>(std::move(*library), std::move(*driver), std::move(*ctx), shared);
  if (auto opened = session->open(*ordinal, folder); !opened) return std::unexpected(opened.error());
  return session;
}
}  // namespace kirari::gpu::hip_backend
