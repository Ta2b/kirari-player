// Ce VkDevice est distinct de celui du lecteur mais sélectionné par le même UUID Vulkan.
// Les FD exportés restent la propriété du lecteur. Un import Vulkan réussi consomme le duplicata.
#pragma once
#include <deque>
#include <memory>
#include <mutex>

#if defined(__linux__) && !defined(KIRARI_SLANG_HOST_TRANSPORT)
#define KIRARI_SLANG_FD_TRANSPORT
#include <fcntl.h>
#include <unistd.h>
#endif

#include "network.h"

namespace slang_reader {
using namespace c64;

#ifdef KIRARI_SLANG_FD_TRANSPORT
class FileDescriptor {
 public:
  explicit FileDescriptor(int borrowed) : value(fcntl(borrowed, F_DUPFD_CLOEXEC, 0)) {
    require(value >= 0, "duplication du FD Vulkan");
  }
  ~FileDescriptor() {
    if (value >= 0) close(value);
  }
  int value;
};

#endif

struct Session {
  Device gpu;
  std::mutex queue;
#ifdef KIRARI_SLANG_FD_TRANSPORT
  VkSemaphore timeline{};
#endif
  explicit Session(std::span<const std::uint8_t> uuid) {
#ifdef KIRARI_SLANG_FD_TRANSPORT
    gpu.open(false, true, false, uuid, true);
#else
    gpu.open(false, true, false, uuid, true, false);
#endif
    require(gpu.subgroup == 32 && gpu.tile_n == 16, "wave32 et matrice 16x16x16 FP16 requises");
  }
#ifdef KIRARI_SLANG_FD_TRANSPORT
  ~Session() {
    vkDeviceWaitIdle(gpu.device);
    vkDestroySemaphore(gpu.device, timeline, nullptr);
  }
  void import_timeline(int fd) {
    require(!timeline, "timeline du lecteur déjà chargée");
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    FileDescriptor duplicate(fd);
    VkImportSemaphoreFdInfoKHR info{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    info.fd = duplicate.value;
    const auto import = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(gpu.device, "vkImportSemaphoreFdKHR"));
    require(import != nullptr, "vkImportSemaphoreFdKHR indisponible");
    VkSemaphore semaphore{};
    ck(vkCreateSemaphore(gpu.device, &create, nullptr, &semaphore));
    info.semaphore = semaphore;
    const auto result = import(gpu.device, &info);
    if (result != VK_SUCCESS) {
      vkDestroySemaphore(gpu.device, semaphore, nullptr);
      ck(result);
    }
    timeline = semaphore;
    duplicate.value = -1;
  }
#endif
};

#ifdef KIRARI_SLANG_FD_TRANSPORT
struct Imported : std::enable_shared_from_this<Imported> {
  std::shared_ptr<Session> session;
  Buffer data;
  VkDeviceSize bytes{};
  explicit Imported(std::shared_ptr<Session> s) : session(std::move(s)), data(session->gpu.device) {}
  void load(int fd, VkDeviceSize allocation, VkDeviceSize size) {
    auto& gpu = session->gpu;
    constexpr auto opaque = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = opaque;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ck(vkCreateBuffer(gpu.device, &info, nullptr, &data.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(gpu.device, data.buffer, &req);
    require(req.size == allocation, "les allocations dédiées ne correspondent pas");
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &mem);
    // Même UUID et même VkBufferCreateInfo. Reprend exactement la sélection du producteur
    // device_tensor.cpp ; vkGetMemoryFdPropertiesKHR est interdit pour OPAQUE_FD.
    U index = 0;
    for (; index < mem.memoryTypeCount; ++index)
      if ((req.memoryTypeBits & (1U << index)) &&
          (mem.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        break;
    require(index < mem.memoryTypeCount, "mémoire locale correspondante absente");
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = data.buffer;
    FileDescriptor duplicate(fd);
    VkImportMemoryFdInfoKHR from{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &dedicated, opaque,
                                 duplicate.value};
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &from, allocation, index};
    ck(vkAllocateMemory(gpu.device, &alloc, nullptr, &data.memory));
    duplicate.value = -1;
    ck(vkBindBufferMemory(gpu.device, data.buffer, data.memory, 0));
    bytes = size;
  }
};

class Recording {
  std::shared_ptr<Session> session;
  VkCommandPool pool{};
  VkDescriptorPool descriptors{};
  VkCommandBuffer cmd{};
  void ownership(bool release) const {
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    for (U i = 0; i < 2; ++i) {
      auto& b = barriers[i];
      b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      const VkAccessFlags accesses = i ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
      b.srcAccessMask = release ? accesses : 0U;
      b.dstAccessMask = release ? 0U : accesses;
      b.srcQueueFamilyIndex = release ? session->gpu.family : VK_QUEUE_FAMILY_EXTERNAL;
      b.dstQueueFamilyIndex = release ? VK_QUEUE_FAMILY_EXTERNAL : session->gpu.family;
      b.buffer = tensors[i]->data.buffer;
      b.size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 2, barriers.data(), 0, nullptr);
  }

 public:
  std::array<std::shared_ptr<Imported>, 2> tensors;
  Recording(std::shared_ptr<Session> s, Imported& x, Imported& y)
      : session(std::move(s)), tensors{x.shared_from_this(), y.shared_from_this()} {}
  ~Recording() {
    vkDestroyCommandPool(session->gpu.device, pool, nullptr);
    vkDestroyDescriptorPool(session->gpu.device, descriptors, nullptr);
  }
  void record(balanced::Network& net) {
    auto& gpu = session->gpu;
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pc.queueFamilyIndex = gpu.family;
    ck(vkCreateCommandPool(gpu.device, &pc, nullptr, &pool));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.commandBufferCount = 1;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ck(vkAllocateCommandBuffers(gpu.device, &ai, &cmd));
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
    VkDescriptorPoolCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    di.maxSets = 2;
    di.poolSizeCount = 1;
    di.pPoolSizes = &size;
    ck(vkCreateDescriptorPool(gpu.device, &di, nullptr, &descriptors));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    ck(vkBeginCommandBuffer(cmd, &begin));
    ownership(false);
    net.record_shared(cmd, descriptors, tensors[0]->data.buffer, tensors[1]->data.buffer);
    ownership(true);
    ck(vkEndCommandBuffer(cmd));
  }
  void submit(std::uint64_t wait, std::uint64_t signal) const {
    require(session->timeline && wait < signal, "valeurs ou sémaphore timeline invalides");
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = timeline.signalSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &wait;
    timeline.pSignalSemaphoreValues = &signal;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
    info.waitSemaphoreCount = info.signalSemaphoreCount = 1;
    info.pWaitSemaphores = info.pSignalSemaphores = &session->timeline;
    info.pWaitDstStageMask = &stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;
    ck(vkQueueSubmit(session->gpu.queue, 1, &info, VK_NULL_HANDLE));
  }
};
#endif
}  // namespace slang_reader
