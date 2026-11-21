#include "gpu/zml/cuda_timeline.h"
#ifdef KIRARI_ZML_CUDA_TIMELINE
#include <bit>
#include <cuda.h>
#include <tuple>
#include <unistd.h>

#include "core/dynamic_library.h"

namespace kirari::gpu::zml {
namespace {
class Timeline final : public CudaTimeline {
 public:
  Timeline(DynamicLibrary lib, std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext ctx)
      : lib_(std::move(lib)), driver_(std::move(driver)), ctx_(std::move(ctx)) {}
  vk::Done open(int fd) {
    auto imp = lib_.symbol<decltype(cuImportExternalSemaphore)>("cuImportExternalSemaphore");
    auto wait = lib_.symbol<decltype(cuWaitExternalSemaphoresAsync)>("cuWaitExternalSemaphoresAsync");
    auto signal = lib_.symbol<decltype(cuSignalExternalSemaphoresAsync)>("cuSignalExternalSemaphoresAsync");
    auto destroy = lib_.symbol<decltype(cuDestroyExternalSemaphore)>("cuDestroyExternalSemaphore");
    if (!imp || !wait || !signal || !destroy)
      return refuse(RefusalCode::FeatureUnavailable, "CUDA : sémaphores externes indisponibles");
    wait_ = *wait;
    signal_ = *signal;
    destroy_ = *destroy;
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
    desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
    desc.handle.fd = ::dup(fd);
    if (desc.handle.fd < 0) return refuse(RefusalCode::IoFailed, "dup du sémaphore Vulkan");
    auto ok = call([&] { return (*imp)(&semaphore_, &desc); });
    if (!ok) ::close(desc.handle.fd);
    return ok;
  }
  ~Timeline() override {
    if (semaphore_) std::ignore = call([&] { return destroy_(semaphore_); });
  }
  vk::Done wait(std::uintptr_t stream, std::uint64_t value) override {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    params.params.fence.value = value;
    return call([&] { return wait_(&semaphore_, &params, 1, std::bit_cast<CUstream>(stream)); });
  }
  vk::Done signal(std::uintptr_t stream, std::uint64_t value) override {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    params.params.fence.value = value;
    return call([&] { return signal_(&semaphore_, &params, 1, std::bit_cast<CUstream>(stream)); });
  }

 private:
  template <class F>
  vk::Done call(F f) {
    CUresult status = CUDA_SUCCESS;
    if (auto ok = driver_->in_context(ctx_.context, [&] { status = f(); }); !ok) return ok;
    if (status != CUDA_SUCCESS)
      return refuse(RefusalCode::VendorCallFailed,
                    std::format("sémaphore CUDA : erreur {}", static_cast<int>(status)));
    return {};
  }
  DynamicLibrary lib_;
  std::shared_ptr<media::CudaDriver> driver_;
  media::PrimaryContext ctx_;
  CUexternalSemaphore semaphore_{};
  decltype(&cuWaitExternalSemaphoresAsync) wait_{};
  decltype(&cuSignalExternalSemaphoresAsync) signal_{};
  decltype(&cuDestroyExternalSemaphore) destroy_{};
};
}  // namespace
std::expected<std::shared_ptr<CudaTimeline>, Refusal> CudaTimeline::import(
    std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext ctx, int fd) {
  auto lib = DynamicLibrary::open_system("libcuda.so.1");
  if (!lib) return std::unexpected(lib.error());
  auto timeline = std::make_shared<Timeline>(std::move(*lib), std::move(driver), std::move(ctx));
  if (auto ok = timeline->open(fd); !ok) return std::unexpected(ok.error());
  return timeline;
}
}  // namespace kirari::gpu::zml
#else
namespace kirari::gpu::zml {
std::expected<std::shared_ptr<CudaTimeline>, Refusal> CudaTimeline::import(std::shared_ptr<media::CudaDriver>,
                                                                           media::PrimaryContext, int) {
  return refuse(RefusalCode::FeatureUnavailable, "interface de sémaphore CUDA absente du build");
}
}  // namespace kirari::gpu::zml
#endif
