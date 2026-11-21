#include <cuda.h>
#include <stdexcept>

#include "core/dynamic_library.h"
#include "gpu/triton/device.h"
#include "gpu/zml/cuda_timeline.h"

namespace kirari::gpu::triton {
namespace {
template <class Fn>
Fn* required(const DynamicLibrary& library, const char* name) {
  auto symbol = library.symbol<Fn>(name);
  if (!symbol) throw std::runtime_error(symbol.error().detail);
  return *symbol;
}

#ifdef KIRARI_TRITON_CUDA_GRAPHS
struct GraphApi {
  explicit GraphApi(const DynamicLibrary& lib)
      : create(lib.symbol<decltype(cuGraphCreate)>("cuGraphCreate").value_or(nullptr)),
        add(lib.symbol<decltype(cuGraphAddKernelNode)>("cuGraphAddKernelNode_v2").value_or(nullptr)),
        instantiate(
            lib.symbol<decltype(cuGraphInstantiate)>("cuGraphInstantiateWithFlags").value_or(nullptr)),
        destroy(lib.symbol<decltype(cuGraphDestroy)>("cuGraphDestroy").value_or(nullptr)),
        exec_destroy(lib.symbol<decltype(cuGraphExecDestroy)>("cuGraphExecDestroy").value_or(nullptr)),
        launch(lib.symbol<decltype(cuGraphLaunch)>("cuGraphLaunch").value_or(nullptr)) {}

  explicit operator bool() const { return create && add && instantiate && destroy && exec_destroy && launch; }

  decltype(cuGraphCreate)* create;
  decltype(cuGraphAddKernelNode)* add;
  decltype(cuGraphInstantiate)* instantiate;
  decltype(cuGraphDestroy)* destroy;
  decltype(cuGraphExecDestroy)* exec_destroy;
  decltype(cuGraphLaunch)* launch;
};
#endif

struct Api {
  explicit Api(DynamicLibrary library) : lib(std::move(library)) {}
  DynamicLibrary lib;
  decltype(cuGetErrorString)* error = required<decltype(cuGetErrorString)>(lib, "cuGetErrorString");
  decltype(cuDeviceGetAttribute)* attribute =
      required<decltype(cuDeviceGetAttribute)>(lib, "cuDeviceGetAttribute");
  decltype(cuMemAlloc)* alloc = required<decltype(cuMemAlloc)>(lib, "cuMemAlloc_v2");
  decltype(cuMemFree)* free = required<decltype(cuMemFree)>(lib, "cuMemFree_v2");
  decltype(cuMemcpyHtoD)* upload = required<decltype(cuMemcpyHtoD)>(lib, "cuMemcpyHtoD_v2");
  decltype(cuMemcpyDtoH)* download = required<decltype(cuMemcpyDtoH)>(lib, "cuMemcpyDtoH_v2");
  decltype(cuModuleLoadData)* load = required<decltype(cuModuleLoadData)>(lib, "cuModuleLoadData");
  decltype(cuModuleGetFunction)* function =
      required<decltype(cuModuleGetFunction)>(lib, "cuModuleGetFunction");
  decltype(cuFuncSetAttribute)* set_attribute =
      required<decltype(cuFuncSetAttribute)>(lib, "cuFuncSetAttribute");
  decltype(cuModuleUnload)* unload = required<decltype(cuModuleUnload)>(lib, "cuModuleUnload");
  decltype(cuLaunchKernel)* launch = required<decltype(cuLaunchKernel)>(lib, "cuLaunchKernel");
  decltype(cuStreamSynchronize)* sync = required<decltype(cuStreamSynchronize)>(lib, "cuStreamSynchronize");
#ifdef KIRARI_TRITON_CUDA_GRAPHS
  GraphApi graphs{lib};
#endif

  void check(CUresult status, std::string_view action) const {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr;
    error(status, &message);
    throw std::runtime_error(std::format("{} : {} ({})", action, message ? message : "CUDA", int(status)));
  }
};

#ifdef KIRARI_TRITON_CUDA_GRAPHS
class CudaRecordedLaunch final : public RecordedLaunch {
 public:
  CudaRecordedLaunch(std::shared_ptr<Api> api, CUstream stream) : api_(std::move(api)), stream_(stream) {}
  ~CudaRecordedLaunch() override {
    if (exec_ != nullptr) static_cast<void>(api_->graphs.exec_destroy(exec_));
  }

  bool record(std::span<Launch> commands) {
    const auto& api = api_->graphs;
    struct Graph {
      const GraphApi& api;
      CUgraph handle{};
      ~Graph() {
        if (handle != nullptr) static_cast<void>(api.destroy(handle));
      }
    } graph{api};
    if (api.create(&graph.handle, 0) != CUDA_SUCCESS) return false;
    CUgraphNode previous{};
    for (Launch& cmd : commands) {
      const Kernel& k = cmd.kernel;
      const CUDA_KERNEL_NODE_PARAMS params{.func = native_handle<CUfunction>(k.function),
                                           .gridDimX = k.grid,
                                           .gridDimY = k.grid_y,
                                           .gridDimZ = k.grid_z,
                                           .blockDimX = k.block,
                                           .blockDimY = 1,
                                           .blockDimZ = 1,
                                           .sharedMemBytes = k.shared,
                                           .kernelParams = cmd.args.data()};
      CUgraphNode node{};
      if (api.add(&node, graph.handle, previous ? &previous : nullptr, previous ? 1U : 0U, &params) !=
          CUDA_SUCCESS)
        return false;
      previous = node;
    }
    return api.instantiate(&exec_, graph.handle, 0) == CUDA_SUCCESS;
  }

  void launch() override { api_->check(api_->graphs.launch(exec_, stream_), "cuGraphLaunch"); }

 private:
  std::shared_ptr<Api> api_;
  CUstream stream_;
  CUgraphExec exec_{};
};
#endif

class CudaDevice final : public Device {
 public:
  CudaDevice(std::shared_ptr<Api> api, std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext ctx,
             Pointer stream)
      : api_(std::move(api)), driver_(std::move(driver)), ctx_(std::move(ctx)), stream_(stream) {}
  ~CudaDevice() override {
    std::ignore = driver_->synchronize(ctx_.context, stream_);
    timeline_.reset();
    driver_->destroy_stream(ctx_.context, stream_);
  }

  std::string_view architecture() const override { return TRITON_PACKAGE_CUDA_ARCH; }

  vk::Done in_context(const std::function<void()>& work) override {
    std::string detail;
    auto ok = driver_->in_context(ctx_.context, [&] {
      try {
        work();
      } catch (const std::exception& e) {
        detail = e.what();
      }
    });
    if (!ok) return ok;
    if (!detail.empty()) return refuse(RefusalCode::VendorCallFailed, "Triton CUDA : " + detail);
    return {};
  }
  Pointer allocate(std::size_t size, std::string_view label) override {
    CUdeviceptr ptr{};
    api_->check(api_->alloc(&ptr, size), label);
    return ptr;
  }
  void release(Pointer ptr) override { static_cast<void>(api_->free(ptr)); }
  void upload(Pointer ptr, const void* data, std::size_t size, std::string_view label) override {
    api_->check(api_->upload(ptr, data, size), label);
  }
  void download(void* data, Pointer ptr, std::size_t size) override {
    api_->check(api_->download(data, ptr, size), "sortie CUDA");
  }
  Module load(std::span<const char> data, std::string_view name, unsigned shared) override {
    CUmodule mod{};
    api_->check(api_->load(&mod, data.data()), name);
    try {
      CUfunction function{};
      api_->check(api_->function(&function, mod, std::string(name).c_str()), name);
      if (shared > 49152)
        api_->check(
            api_->set_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(shared)),
            name);
      return {reinterpret_cast<Pointer>(mod), reinterpret_cast<Pointer>(function)};
    } catch (...) {
      static_cast<void>(api_->unload(mod));
      throw;
    }
  }
  void unload(Pointer module) override { static_cast<void>(api_->unload(native_handle<CUmodule>(module))); }
  void launch(const Kernel& k, std::span<void*> args) override {
    api_->check(api_->launch(native_handle<CUfunction>(k.function), k.grid, k.grid_y, k.grid_z, k.block, 1, 1,
                             k.shared, cuda_stream(), args.data(), nullptr),
                "cuLaunchKernel");
  }
#ifdef KIRARI_TRITON_CUDA_GRAPHS
  bool can_record() const override { return static_cast<bool>(api_->graphs); }
  std::unique_ptr<RecordedLaunch> record(std::span<Launch> commands) override {
    if (!can_record() || commands.empty()) return nullptr;
    auto graph = std::make_unique<CudaRecordedLaunch>(api_, cuda_stream());
    if (!graph->record(commands)) return nullptr;
    return graph;
  }
#endif
  void synchronize() override { api_->check(api_->sync(cuda_stream()), "cuStreamSynchronize"); }
  bool shares_device_memory() const override {
#ifdef __linux__
    return true;
#else
    return false;
#endif
  }
  vk::Done import_timeline(int fd) override {
    auto timeline = zml::CudaTimeline::import(driver_, ctx_, fd);
    if (!timeline) return std::unexpected(timeline.error());
    timeline_ = *timeline;
    return {};
  }
  std::expected<vk::net::DeviceMemory, Refusal> import_memory(Pointer fd, std::size_t size,
                                                              std::size_t bytes) override {
    auto mem = driver_->import_memory(ctx_.context, fd, size, 0, bytes, true);
    if (!mem) return std::unexpected(mem.error());
    return vk::net::DeviceMemory{mem->pointer, std::move(mem->owner)};
  }
  vk::Done wait(std::uint64_t value) override {
    if (!timeline_) return refuse(RefusalCode::FeatureUnavailable, "Triton CUDA : sémaphore absent");
    return timeline_->wait(stream_, value);
  }
  vk::Done signal(std::uint64_t value) override {
    if (!timeline_) return refuse(RefusalCode::FeatureUnavailable, "Triton CUDA : sémaphore absent");
    return timeline_->signal(stream_, value);
  }

 private:
  CUstream cuda_stream() const { return native_handle<CUstream>(stream_); }
  std::shared_ptr<Api> api_;
  std::shared_ptr<media::CudaDriver> driver_;
  media::PrimaryContext ctx_;
  Pointer stream_{};
  std::shared_ptr<zml::CudaTimeline> timeline_;
};
}  // namespace

std::expected<std::shared_ptr<Device>, Refusal> make_cuda(const vk::DeviceCaps& caps) {
  auto driver = media::CudaDriver::load();
  if (!driver) return std::unexpected(driver.error());
  auto ordinal = (*driver)->ordinal_of(caps.uuid);
  if (!ordinal) return std::unexpected(ordinal.error());
#ifdef _WIN32
  auto library = DynamicLibrary::open_system("nvcuda.dll");
#else
  auto library = DynamicLibrary::open_system("libcuda.so.1");
#endif
  if (!library) return std::unexpected(library.error());
  auto api = std::make_shared<Api>(std::move(*library));
  int major{}, minor{};
  api->check(api->attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, *ordinal),
             "architecture CUDA");
  api->check(api->attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, *ordinal),
             "architecture CUDA");
  const auto architecture = std::format("sm{}{}", major, minor);
  if (architecture != TRITON_PACKAGE_CUDA_ARCH)
    return refuse(RefusalCode::DeviceMismatch, "Triton : paquet CUDA absent pour " + architecture);
  auto ctx = (*driver)->retain_primary(*ordinal);
  if (!ctx) return std::unexpected(ctx.error());
  auto stream = (*driver)->create_stream(ctx->context);
  if (!stream) return std::unexpected(stream.error());
  try {
    return std::make_shared<CudaDevice>(std::move(api), *driver, *ctx, *stream);
  } catch (...) {
    (*driver)->destroy_stream(ctx->context, *stream);
    throw;
  }
}
}  // namespace kirari::gpu::triton
