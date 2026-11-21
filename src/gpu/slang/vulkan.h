#pragma once
// Lance un shader de convolution HWC FP16 sur des poids KRSC, avec mesures GPU résidentes.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "data.h"

namespace c64 {
inline void ck(VkResult rc) {
  if (rc != VK_SUCCESS) throw std::runtime_error("Vulkan " + std::to_string(rc));
}
inline bool has_extension(std::span<const VkExtensionProperties> extensions, const char* name) {
  return std::ranges::any_of(extensions,
                             [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}
struct Device {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkPhysicalDeviceProperties props{};
  VkDevice device{};
  VkQueue queue{};
  U family{}, timestamp_bits{}, subgroup{}, tile_n{}, memory_type{};
  bool force_wave32{};
  ~Device() {
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
  }
  void open(bool generic, bool external, bool stats, std::span<const std::uint8_t> uuid = {},
            bool require_wave32 = false, bool import_fd = true);

 private:
  void open_instance();
  void select_physical(std::span<const std::uint8_t> uuid);
  void check_features(std::span<const VkExtensionProperties> extensions, bool generic, bool external,
                      bool require_wave32, bool share);
  void open_logical(std::span<const VkExtensionProperties> extensions, bool external, bool stats, bool share);
};

inline void Device::open_instance() {
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &application;
  U ext_count = 0;
  ck(vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr));
  std::vector<VkExtensionProperties> exts(ext_count);
  ck(vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data()));
  const char* enumeration = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
  if (has_extension(exts, enumeration)) {
    ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = &enumeration;
  }
  ck(vkCreateInstance(&ci, nullptr, &instance));
}

inline void Device::select_physical(std::span<const std::uint8_t> uuid) {
  U count = 0;
  ck(vkEnumeratePhysicalDevices(instance, &count, nullptr));
  std::vector<VkPhysicalDevice> devices(count);
  ck(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
  int best = -1;
  for (auto pd : devices) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(pd, &properties);
    if (!uuid.empty()) {
      VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 details{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
      vkGetPhysicalDeviceProperties2(pd, &details);
      if (!std::ranges::equal(uuid, id.deviceUUID)) continue;
    }
    if (properties.apiVersion < VK_API_VERSION_1_2) continue;
    U nf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> families(nf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, families.data());
    for (U i = 0; i < nf; ++i) {
      const auto& f = families[i];
      if (!(f.queueFlags & VK_QUEUE_COMPUTE_BIT) || !f.timestampValidBits) continue;
      const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
      if (score <= best) continue;
      best = score;
      physical = pd;
      props = properties;
      family = i;
      timestamp_bits = f.timestampValidBits;
    }
  }
  require(physical != VK_NULL_HANDLE, "GPU Vulkan 1.2 avec horodatage compute requis");
}

inline U cooperative_width(VkInstance instance, VkPhysicalDevice physical) {
  auto list = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  require(list != nullptr, "fonction coopmat absente");
  U count = 0;
  ck(list(physical, &count, nullptr));
  std::vector<VkCooperativeMatrixPropertiesKHR> rows(count,
                                                     {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  ck(list(physical, &count, rows.data()));
  U width = 0;
  for (const auto& m : rows)
    if (m.MSize == 16 && m.KSize == 16 && (m.NSize == 8 || m.NSize == 16) &&
        m.scope == VK_SCOPE_SUBGROUP_KHR && m.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        m.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && m.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        m.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR && !m.saturatingAccumulation)
      width = std::max(width, m.NSize);
  return width;
}

inline void Device::check_features(std::span<const VkExtensionProperties> extensions, bool generic,
                                   bool external, bool require_wave32, bool share) {
  VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
  VkPhysicalDeviceSubgroupSizeControlPropertiesEXT wave_properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
  const bool has_wave_control = has_extension(extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
  sg.pNext = has_wave_control ? &wave_properties : nullptr;
  vkGetPhysicalDeviceProperties2(physical, &p2);
  subgroup = sg.subgroupSize;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
  f11.pNext = &f12;
  const bool available = has_extension(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT wave_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
  void* wave = has_wave_control ? &wave_features : nullptr;
  f12.pNext = available ? &coop : wave;
  coop.pNext = wave;
  vkGetPhysicalDeviceFeatures2(physical, &features);
  if (require_wave32 && subgroup != 32) {
    require(wave_features.subgroupSizeControl && wave_properties.minSubgroupSize <= 32 &&
                wave_properties.maxSubgroupSize >= 32 &&
                (wave_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT),
            "les shaders Slang exigent des sous-groupes de 32");
    force_wave32 = true;
    subgroup = 32;
  }
  if (!generic && available && coop.cooperativeMatrix && f11.storageBuffer16BitAccess && f12.shaderFloat16 &&
      f12.vulkanMemoryModel && (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
      (sg.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
      4 * subgroup <= props.limits.maxComputeWorkGroupInvocations)
    tile_n = cooperative_width(instance, physical);
  if (share)
    require(has_extension(extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
                has_extension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) &&
                f12.timelineSemaphore,
            "imports FD et sémaphore timeline requis par le lecteur Slang");
  if (external)
    require(tile_n && f11.uniformAndStorageBuffer16BitAccess && f12.vulkanMemoryModelDeviceScope,
            "le SPIR-V coopératif externe exige stockage uniforme 16 bits et VulkanMemoryModelDeviceScope");
}

inline void Device::open_logical(std::span<const VkExtensionProperties> extensions, bool external, bool stats,
                                 bool share) {
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  std::vector<const char*> names;
  if (share) {
    names.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    names.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  }
  if (stats) {
    require(has_extension(extensions, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME),
            "VK_KHR_pipeline_executable_properties indisponible sur ce pilote");
    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &executable};
    vkGetPhysicalDeviceFeatures2(physical, &query);
    require(executable.pipelineExecutableInfo, "statistiques de l'exécutable indisponibles");
    names.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
  }
  if (has_extension(extensions, "VK_KHR_portability_subset")) names.push_back("VK_KHR_portability_subset");
  if (force_wave32) names.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
  if (tile_n) names.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  // N'activer que les capacités requises par la variante retenue.
  VkPhysicalDeviceSubgroupSizeControlFeaturesEXT wave_features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT, stats ? &executable : nullptr};
  wave_features.subgroupSizeControl = force_wave32;
  void* next = stats ? &executable : nullptr;
  if (force_wave32) next = &wave_features;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, next};
  coop.cooperativeMatrix = tile_n != 0;
  if (tile_n) next = &coop;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, next};
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
  f11.storageBuffer16BitAccess = f12.vulkanMemoryModel = f12.shaderFloat16 = tile_n != 0;
  f11.uniformAndStorageBuffer16BitAccess = f12.vulkanMemoryModelDeviceScope = external;
  f12.timelineSemaphore = share;
  float priority = 1;
  VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_ci.queueFamilyIndex = family;
  queue_ci.queueCount = 1;
  queue_ci.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_ci.queueCreateInfoCount = 1;
  device_ci.pQueueCreateInfos = &queue_ci;
  device_ci.enabledExtensionCount = static_cast<U>(names.size());
  device_ci.ppEnabledExtensionNames = names.data();
  device_ci.pNext = tile_n || share ? &f11 : next;
  ck(vkCreateDevice(physical, &device_ci, nullptr, &device));
  vkGetDeviceQueue(device, family, 0, &queue);
}

inline void Device::open(bool generic, bool external, bool stats, std::span<const std::uint8_t> uuid,
                         bool require_wave32, bool import_fd) {
  open_instance();
  select_physical(uuid);
  U count = 0;
  ck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
  std::vector<VkExtensionProperties> extensions(count);
  ck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()));
  const bool share = import_fd && !uuid.empty();
  check_features(extensions, generic, external, require_wave32, share);
  open_logical(extensions, external, stats, share);
}

struct Buffer {
  VkDevice device;
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  U type{};
  explicit Buffer(VkDevice d) : device(d) {}
  ~Buffer() {
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
  }
  void create(const Device& gpu, VkDeviceSize bytes, bool host) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 (host ? 0 : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ck(vkCreateBuffer(device, &info, nullptr, &buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &mem);
    const VkMemoryPropertyFlags wanted =
        host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (type = 0; type < mem.memoryTypeCount; ++type)
      if ((req.memoryTypeBits & (1u << type)) && (mem.memoryTypes[type].propertyFlags & wanted) == wanted)
        break;
    require(type < mem.memoryTypeCount, "type de mémoire absent");
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    ck(vkAllocateMemory(device, &ai, nullptr, &memory));
    ck(vkBindBufferMemory(device, buffer, memory, 0));
  }
};

inline void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags writes,
                    VkPipelineStageFlags dest, VkAccessFlags reads) {
  VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  dependency.srcAccessMask = writes;
  dependency.dstAccessMask = reads;
  vkCmdPipelineBarrier(cmd, src, dest, 0, 1, &dependency, 0, nullptr, 0, nullptr);
}

}  // namespace c64
