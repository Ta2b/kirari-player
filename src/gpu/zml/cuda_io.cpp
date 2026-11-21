// La file Vulkan cède les allocations au runtime avant view() et les reprend après la copie D2D.
#include "gpu/zml/cuda_io.h"

#include "gpu/zml/cuda_timeline.h"

#ifdef KIRARI_ZML_CUDA_IO
#include <algorithm>
#include <array>
#include <bit>

#include "media/cuda_driver.h"
#include "xla/pjrt/c/pjrt_c_api_stream_extension.h"

namespace kirari::gpu::zml {
namespace {
constexpr std::array<std::int64_t, 4> order{3, 2, 1, 0};
class Interop final : public CudaIo {
 public:
  Interop(std::shared_ptr<media::CudaDriver> driver, media::PrimaryContext context,
          const PJRT_Stream_Extension& streams, std::uintptr_t stream, std::intptr_t producer)
      : driver_(std::move(driver)),
        ctx_(std::move(context)),
        ext_(streams),
        stream_(stream),
        producer_(producer) {}
  ~Interop() override { driver_->destroy_stream(ctx_.context, stream_); }
  std::expected<vk::net::DeviceMemory, Refusal> import(std::uintptr_t handle, std::size_t allocation,
                                                       std::size_t bytes) override {
    auto mapped = driver_->import_memory(ctx_.context, handle, allocation, 0, bytes, true);
    if (!mapped) return std::unexpected(mapped.error());
    return vk::net::DeviceMemory{mapped->pointer, std::move(mapped->owner)};
  }
  vk::Done attach(int fd) override {
    auto made = CudaTimeline::import(driver_, ctx_, fd);
    if (!made) return std::unexpected(made.error());
    timeline_ = std::move(*made);
    return {};
  }
  vk::Done wait(std::uint64_t v) override {
    return timeline_->wait(static_cast<std::uintptr_t>(producer_), v);
  }
  vk::Done signal(std::uint64_t v) override { return timeline_->signal(stream_, v); }
  vk::Done drain() override { return driver_->synchronize(ctx_.context, stream_); }
  std::expected<PJRT_Buffer*, Refusal> view(Client& c, const vk::net::HostBinding& in,
                                            PJRT_Buffer_Type type) override {
    PJRT_Buffer_MemoryLayout layout{.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE,
                                    .tiled = {.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE,
                                              .minor_to_major = order.data(),
                                              .minor_to_major_size = order.size()},
                                    .type = PJRT_Buffer_MemoryLayout_Type_Tiled};
    PJRT_Client_CreateViewOfDeviceBuffer_Args args{};
    args.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
    args.client = c.handle;
    args.device_buffer_ptr = std::bit_cast<void*>(in.device);
    args.dims = in.shape.dims.data();
    args.num_dims = in.shape.dims.size();
    args.element_type = type;
    args.layout = &layout;
    args.device = c.device;
    args.on_delete_callback = [](void*, void*) {};
    args.stream = producer_;
    if (auto ok = checked(c.api, c.api.PJRT_Client_CreateViewOfDeviceBuffer(&args), "vue GPU de l’entrée");
        !ok)
      return std::unexpected(ok.error());
    return args.buffer;
  }
  std::expected<void, Refusal> copy(Client& c, PJRT_Buffer* output, const vk::net::HostBinding& target,
                                    std::vector<std::shared_ptr<void>>* keep) override {
    PJRT_Buffer_GetMemoryLayout_Args layout{.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE,
                                            .buffer = output};
    if (auto ok = checked(c.api, c.api.PJRT_Buffer_GetMemoryLayout(&layout), "disposition de la sortie GPU");
        !ok)
      return ok;
    const auto& tiled = layout.layout.tiled;
    if (layout.layout.type != PJRT_Buffer_MemoryLayout_Type_Tiled || tiled.num_tiles != 0 ||
        !std::ranges::equal(order, std::span(tiled.minor_to_major, tiled.minor_to_major_size)))
      return refuse(RefusalCode::FormatUnsupported, "ZML : sortie GPU non dense en NCHW");
    PJRT_Buffer_IncreaseExternalReferenceCount_Args hold{
        .struct_size = PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE, .buffer = output};
    if (auto ok = checked(c.api, c.api.PJRT_Buffer_IncreaseExternalReferenceCount(&hold),
                          "retenue de la sortie GPU");
        !ok)
      return ok;
    auto retained = std::shared_ptr<void>(output, [&api = c.api](void* ptr) {
      PJRT_Buffer_DecreaseExternalReferenceCount_Args down{
          .struct_size = PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE,
          .buffer = static_cast<PJRT_Buffer*>(ptr)};
      discard(api, api.PJRT_Buffer_DecreaseExternalReferenceCount(&down));
    });
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args pointer{
        .struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE, .buffer = output};
    if (auto ok = checked(c.api, c.api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&pointer),
                          "adresse de la sortie GPU");
        !ok)
      return ok;
    PJRT_Wait_Until_Buffer_Ready_On_Stream_Args wait{
        .struct_size = PJRT_Wait_Until_Buffer_Ready_On_Stream_Args_STRUCT_SIZE,
        .stream = static_cast<std::intptr_t>(stream_),
        .buffer = output};
    if (auto ok = checked(c.api, ext_.wait_stream(&wait), "événement PJRT vers CUDA"); !ok) return ok;
    if (auto ok = driver_->copy_2d(ctx_.context, stream_, target.device, target.bytes,
                                   reinterpret_cast<std::uintptr_t>(pointer.device_memory_ptr), target.bytes,
                                   target.bytes, 1);
        !ok)
      return ok;
    if (keep) {
      keep->push_back(std::move(retained));
      return {};
    }
    return drain();
  }

 private:
  std::shared_ptr<media::CudaDriver> driver_;
  media::PrimaryContext ctx_;
  const PJRT_Stream_Extension& ext_;
  std::shared_ptr<CudaTimeline> timeline_;
  std::uintptr_t stream_;
  std::intptr_t producer_;
};
}  // namespace
std::shared_ptr<CudaIo> CudaIo::open(Client& c, const vk::DeviceUuid& uuid) {
  const auto& api = c.api;
  if (api.struct_size < PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_CreateViewOfDeviceBuffer) ||
      !api.PJRT_Client_CreateViewOfDeviceBuffer || !api.PJRT_Buffer_GetMemoryLayout ||
      !api.PJRT_Buffer_IncreaseExternalReferenceCount || !api.PJRT_Buffer_DecreaseExternalReferenceCount ||
      !api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer)
    return {};
  const PJRT_Stream_Extension* streams = nullptr;
  for (auto* ext = api.extension_start; ext; ext = ext->next)
    if (ext->type == PJRT_Extension_Type_Stream && ext->struct_size >= PJRT_Stream_Extension_STRUCT_SIZE)
      streams = reinterpret_cast<const PJRT_Stream_Extension*>(ext);
  if (!streams || !streams->get_stream || !streams->wait_stream) return {};
  auto driver = media::CudaDriver::load();
  if (!driver) return {};
  auto ordinal = (*driver)->ordinal_of(uuid);
  if (!ordinal || !c.local_id || *ordinal != *c.local_id) return {};
  auto ctx = (*driver)->retain_primary(*ordinal);
  if (!ctx) return {};
  PJRT_Get_Stream_For_External_Ready_Events_Args src{
      .struct_size = PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE, .device = c.device};
  if (!checked(api, streams->get_stream(&src), "flux PJRT d’entrée")) return {};
  auto stream = (*driver)->create_stream(ctx->context);
  if (!stream) return {};
  return std::make_shared<Interop>(*driver, std::move(*ctx), *streams, *stream, src.stream);
}
}  // namespace kirari::gpu::zml
#else
namespace kirari::gpu::zml {
std::shared_ptr<CudaIo> CudaIo::open(Client&, const vk::DeviceUuid&) {
  return {};
}
}  // namespace kirari::gpu::zml
#endif
