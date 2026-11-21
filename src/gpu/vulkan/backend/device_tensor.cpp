// Tampon Vulkan dédié, en mémoire locale, partagé avec le runtime d'inférence.
#include <algorithm>
#include <memory>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "gpu/vulkan/backend/host_tensor.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"

namespace kirari::gpu::vk {
namespace {
#if defined(__linux__)
constexpr auto kOpaque = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
class SharedTensor final : public TensorBuffer {
 public:
  SharedTensor(VkContext& context, std::shared_ptr<HostViews> views)
      : ctx_(context), views_(std::move(views)) {}
  ~SharedTensor() override {
    if (tensor_.handle) {
      {
        const std::scoped_lock lock(views_->mutex);
        views_->views.erase(tensor_.handle);
      }
      if (!ctx_.settle()) {
        // La file bloquée peut retenir le prêt de la mémoire, comme dans HostImportedBuffer.
        static_cast<void>(new net::DeviceMemory(std::move(imported_)));
        return;
      }
      ctx_.destroyed().note(tensor_.handle);
    }
    imported_ = {};
    if (buffer_) ctx_.fn().vkDestroyBuffer(ctx_.device(), buffer_, nullptr);
    if (memory_) ctx_.fn().vkFreeMemory(ctx_.device(), memory_, nullptr);
  }
  const Tensor& tensor() const override { return tensor_; }
  Done open(net::Runtime& runtime, TensorType type, const Shape& shape, std::uint64_t bytes) {
    auto get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        ctx_.instance_functions().vkGetDeviceProcAddr(ctx_.device(), "vkGetMemoryFdKHR"));
    if (!get_fd) return refuse(RefusalCode::FeatureUnavailable, "export de mémoire Vulkan indisponible");
    const VkPhysicalDeviceExternalBufferInfo query{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = kUsage,
        .handleType = kOpaque};
    VkExternalBufferProperties properties{.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    ctx_.instance_functions().vkGetPhysicalDeviceExternalBufferProperties(ctx_.physical_device(), &query,
                                                                          &properties);
    if (!(properties.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
      return refuse(RefusalCode::FeatureUnavailable, "tampon de stockage non exportable");
    const VkExternalMemoryBufferCreateInfo ext{.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                               .handleTypes = kOpaque};
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .pNext = &ext,
                                  .size = (bytes + 3) / 4 * 4,
                                  .usage = kUsage,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    const auto& fn = ctx_.fn();
    auto device = ctx_.device();
    if (auto ok = check(fn.vkCreateBuffer(device, &info, nullptr, &buffer_), "vkCreateBuffer (interop GPU)");
        !ok)
      return ok;
    VkMemoryRequirements req{};
    fn.vkGetBufferMemoryRequirements(device, buffer_, &req);
    VkPhysicalDeviceMemoryProperties props{};
    ctx_.instance_functions().vkGetPhysicalDeviceMemoryProperties(ctx_.physical_device(), &props);
    constexpr auto flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    std::uint32_t index = 0;
    for (; index < props.memoryTypeCount; ++index)
      if ((req.memoryTypeBits & (1U << index)) && (props.memoryTypes[index].propertyFlags & flags) == flags)
        break;
    if (index == props.memoryTypeCount)
      return refuse(RefusalCode::FeatureUnavailable, "aucune mémoire GPU locale exportable");
    const VkMemoryDedicatedAllocateInfo dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                  .buffer = buffer_};
    const VkExportMemoryAllocateInfo exp{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &dedicated, .handleTypes = kOpaque};
    const VkMemoryAllocateInfo alloc{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .pNext = &exp,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = index};
    if (auto ok =
            check(fn.vkAllocateMemory(device, &alloc, nullptr, &memory_), "vkAllocateMemory (interop GPU)");
        !ok)
      return ok;
    if (auto ok = check(fn.vkBindBufferMemory(device, buffer_, memory_, 0), "vkBindBufferMemory"); !ok)
      return ok;
    VkMemoryGetFdInfoKHR fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = memory_, .handleType = kOpaque};
    int fd = -1;
    if (auto ok = check(get_fd(device, &fd_info, &fd), "vkGetMemoryFdKHR"); !ok) return ok;
    auto imported = runtime.import_device(static_cast<std::uintptr_t>(fd), req.size, info.size);
    ::close(fd);
    if (!imported) return std::unexpected(imported.error());
    imported_ = std::move(*imported);
    tensor_ = {.type = type, .shape = shape, .handle = detail::handle_bits(buffer_)};
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.device = imported_.address, .bytes = bytes};
    return {};
  }

 private:
  VkContext& ctx_;
  std::shared_ptr<HostViews> views_;
  Tensor tensor_{};
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  net::DeviceMemory imported_;
};
#endif
}  // namespace

std::expected<std::unique_ptr<TensorBuffer>, Refusal> device_shared_tensor(
    VkContext& context, net::Runtime& runtime, TensorType type, const Shape& shape, std::uint64_t bytes,
    std::shared_ptr<HostViews> views) {
#if defined(__linux__)
  if (std::ranges::contains(context.caps().extensions,
                            std::string_view(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME))) {
    auto tensor = std::make_unique<SharedTensor>(context, std::move(views));
    if (auto ok = tensor->open(runtime, type, shape, bytes); !ok) return std::unexpected(ok.error());
    return tensor;
  }
#else
  static_cast<void>(context);
  static_cast<void>(runtime);
  static_cast<void>(type);
  static_cast<void>(shape);
  static_cast<void>(bytes);
  static_cast<void>(views);
#endif
  return refuse(RefusalCode::FeatureUnavailable, "export des allocations GPU indisponible");
}

Done VulkanBackend::open_runtime_timeline() {
#if defined(__linux__)
  if (!std::ranges::contains(context_->caps().extensions,
                             std::string_view(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)))
    return refuse(RefusalCode::FeatureUnavailable, "timeline externe indisponible");
  const auto get = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      context_->instance_functions().vkGetDeviceProcAddr(context_->device(), "vkGetSemaphoreFdKHR"));
  if (!get) return refuse(RefusalCode::FeatureUnavailable, "export de timeline indisponible");
  constexpr auto opaque = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  auto timeline = TimelineSemaphore::create(*context_, 0, opaque);
  if (!timeline) return std::unexpected(timeline.error());
  VkSemaphoreGetFdInfoKHR info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                               .semaphore = timeline->handle(),
                               .handleType = opaque};
  int fd = -1;
  if (auto ok = check(get(context_->device(), &info, &fd), "vkGetSemaphoreFdKHR"); !ok) return ok;
  auto ready = runtime_->import_timeline(fd);
  ::close(fd);
  if (!ready) return ready;
  runtime_timeline_.emplace(std::move(*timeline));
  return {};
#else
  return refuse(RefusalCode::FeatureUnavailable, "timeline externe indisponible");
#endif
}

Done VulkanKernelOps::transfer_ownership(std::span<const VkBuffer> buffers, bool release,
                                         std::span<const TimelinePoint> external) {
  if (buffers.empty()) return {};
  OpKey key;
  key.add(OpCode::ExternalOwnership).add(release);
  for (auto buffer : buffers) key.add(handle_word(buffer));
  const auto emit = [&](Recorder& recorder) {
    return recorder.commands([&](VkCommandBuffer cmd) {
      std::vector<VkBufferMemoryBarrier> barriers;
      barriers.reserve(buffers.size());
      for (auto buffer : buffers) {
        constexpr VkAccessFlags accesses = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barriers.push_back(
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .srcAccessMask = release ? accesses : 0U,
             .dstAccessMask = release ? 0U : accesses,
             .srcQueueFamilyIndex = release ? context_.queue_family() : VK_QUEUE_FAMILY_EXTERNAL,
             .dstQueueFamilyIndex = release ? VK_QUEUE_FAMILY_EXTERNAL : context_.queue_family(),
             .buffer = buffer,
             .offset = 0,
             .size = VK_WHOLE_SIZE});
      }
      context_.fn().vkCmdPipelineBarrier(
          cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
          static_cast<std::uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    });
  };
  const auto submit_barrier = [&](Recording& rec) {
    return queue_.submit(rec, release ? std::span<const TimelinePoint>{} : external,
                         release ? external : std::span<const TimelinePoint>{});
  };
  if (key.valid()) {
    auto rec = recording(key, emit);
    if (!rec) return std::unexpected(rec.error());
    return submit_barrier(**rec);
  }
  // Un réseau peut exposer plus de tampons que la clé fixe du cache. Sa barrière reste atomique.
  auto recorder = Recorder::begin(context_, memory_);
  if (!recorder) return std::unexpected(recorder.error());
  if (auto ok = emit(*recorder); !ok) return ok;
  auto rec = recorder->finish();
  if (!rec) return std::unexpected(rec.error());
  auto submitted = submit_barrier(**rec);
  retired_.retire(std::move(*rec));
  return submitted;
}
}  // namespace kirari::gpu::vk
