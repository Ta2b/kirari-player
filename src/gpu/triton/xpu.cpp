#include <algorithm>
#include <cstring>
#include <level_zero/ze_api.h>
#include <stdexcept>

#include "core/dynamic_library.h"
#include "gpu/triton/device.h"

namespace kirari::gpu::triton {
namespace {
void check(ze_result_t status, std::string_view function) {
  if (status != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::format("{} : Level Zero ({:#x})", function, unsigned(status)));
}

template <class Fn>
Fn* required(const DynamicLibrary& lib, const char* name) {
  auto symbol = lib.symbol<Fn>(name);
  if (!symbol) throw std::runtime_error(symbol.error().detail);
  return *symbol;
}

struct Api {
  explicit Api(DynamicLibrary library) : lib(std::move(library)) {}
  DynamicLibrary lib;
  decltype(::zeInit)* zeInit = required<decltype(::zeInit)>(lib, "zeInit");
  decltype(::zeDriverGet)* zeDriverGet = required<decltype(::zeDriverGet)>(lib, "zeDriverGet");
  decltype(::zeDeviceGet)* zeDeviceGet = required<decltype(::zeDeviceGet)>(lib, "zeDeviceGet");
  decltype(::zeDeviceGetProperties)* zeDeviceGetProperties =
      required<decltype(::zeDeviceGetProperties)>(lib, "zeDeviceGetProperties");
  decltype(::zeDeviceGetCommandQueueGroupProperties)* zeDeviceGetCommandQueueGroupProperties =
      required<decltype(::zeDeviceGetCommandQueueGroupProperties)>(lib,
                                                                   "zeDeviceGetCommandQueueGroupProperties");
  decltype(::zeContextCreate)* zeContextCreate =
      required<decltype(::zeContextCreate)>(lib, "zeContextCreate");
  decltype(::zeContextDestroy)* zeContextDestroy =
      required<decltype(::zeContextDestroy)>(lib, "zeContextDestroy");
  decltype(::zeMemAllocDevice)* zeMemAllocDevice =
      required<decltype(::zeMemAllocDevice)>(lib, "zeMemAllocDevice");
  decltype(::zeMemAllocHost)* zeMemAllocHost = required<decltype(::zeMemAllocHost)>(lib, "zeMemAllocHost");
  decltype(::zeMemFree)* zeMemFree = required<decltype(::zeMemFree)>(lib, "zeMemFree");
  decltype(::zeCommandListCreateImmediate)* zeCommandListCreateImmediate =
      required<decltype(::zeCommandListCreateImmediate)>(lib, "zeCommandListCreateImmediate");
  decltype(::zeCommandListDestroy)* zeCommandListDestroy =
      required<decltype(::zeCommandListDestroy)>(lib, "zeCommandListDestroy");
  decltype(::zeCommandListAppendMemoryCopy)* zeCommandListAppendMemoryCopy =
      required<decltype(::zeCommandListAppendMemoryCopy)>(lib, "zeCommandListAppendMemoryCopy");
  decltype(::zeCommandListAppendLaunchKernel)* zeCommandListAppendLaunchKernel =
      required<decltype(::zeCommandListAppendLaunchKernel)>(lib, "zeCommandListAppendLaunchKernel");
  decltype(::zeCommandListAppendBarrier)* zeCommandListAppendBarrier =
      required<decltype(::zeCommandListAppendBarrier)>(lib, "zeCommandListAppendBarrier");
  decltype(::zeCommandListHostSynchronize)* zeCommandListHostSynchronize =
      required<decltype(::zeCommandListHostSynchronize)>(lib, "zeCommandListHostSynchronize");
  decltype(::zeModuleCreate)* zeModuleCreate = required<decltype(::zeModuleCreate)>(lib, "zeModuleCreate");
  decltype(::zeModuleDestroy)* zeModuleDestroy =
      required<decltype(::zeModuleDestroy)>(lib, "zeModuleDestroy");
  decltype(::zeKernelCreate)* zeKernelCreate = required<decltype(::zeKernelCreate)>(lib, "zeKernelCreate");
  decltype(::zeKernelDestroy)* zeKernelDestroy =
      required<decltype(::zeKernelDestroy)>(lib, "zeKernelDestroy");
  decltype(::zeKernelSetGroupSize)* zeKernelSetGroupSize =
      required<decltype(::zeKernelSetGroupSize)>(lib, "zeKernelSetGroupSize");
  decltype(::zeKernelSetArgumentValue)* zeKernelSetArgumentValue =
      required<decltype(::zeKernelSetArgumentValue)>(lib, "zeKernelSetArgumentValue");
  decltype(::zeKernelGetProperties)* zeKernelGetProperties =
      required<decltype(::zeKernelGetProperties)>(lib, "zeKernelGetProperties");
};

struct Adapter {
  ze_driver_handle_t driver{};
  ze_device_handle_t device{};
};

std::expected<Adapter, Refusal> select_adapter(const Api& api, const vk::DeviceCaps& caps) {
  uint32_t count{};
  check(api.zeDriverGet(&count, nullptr), "zeDriverGet");
  std::vector<ze_driver_handle_t> drivers(count);
  check(api.zeDriverGet(&count, drivers.data()), "zeDriverGet");
  const bool has_uuid = std::ranges::any_of(caps.uuid, [](auto byte) { return byte != 0; });
  const auto pci_api = api.lib.symbol<decltype(::zeDevicePciGetPropertiesExt)>("zeDevicePciGetPropertiesExt");
  for (auto driver : drivers) {
    uint32_t devices_count{};
    check(api.zeDeviceGet(driver, &devices_count, nullptr), "zeDeviceGet");
    std::vector<ze_device_handle_t> devices(devices_count);
    check(api.zeDeviceGet(driver, &devices_count, devices.data()), "zeDeviceGet");
    for (auto device : devices) {
      ze_device_properties_t props{.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      check(api.zeDeviceGetProperties(device, &props), "zeDeviceGetProperties");
      if (props.vendorId != caps.vendor_id || props.deviceId != caps.device_id) continue;
      if (has_uuid && std::memcmp(props.uuid.id, caps.uuid.data(), caps.uuid.size()) == 0)
        return Adapter{driver, device};
      if (caps.pci && pci_api) {
        ze_pci_ext_properties_t pci{.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
        if ((*pci_api)(device, &pci) == ZE_RESULT_SUCCESS) {
          const auto& address = pci.address;
          if (*caps.pci == vk::PciAddress{address.domain, address.bus, address.device, address.function})
            return Adapter{driver, device};
        }
      }
    }
  }
  return refuse(RefusalCode::DeviceMismatch, "Triton : identité du GPU Vulkan absente de Level Zero");
}

struct LoadedKernel {
  ze_module_handle_t module;
  ze_kernel_handle_t function;
  std::uint32_t arguments;
};

class XpuDevice final : public Device {
 public:
  XpuDevice(std::shared_ptr<Api> api, Adapter adapter) : api_(std::move(api)), adapter_(adapter) {}
  ~XpuDevice() override {
    if (queue_) {
      static_cast<void>(api_->zeCommandListHostSynchronize(queue_, UINT64_MAX));
      static_cast<void>(api_->zeCommandListDestroy(queue_));
    }
    if (staging_) static_cast<void>(api_->zeMemFree(context_, staging_));
    if (context_) static_cast<void>(api_->zeContextDestroy(context_));
  }

  void open() {
    ze_context_desc_t context_desc{.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    check(api_->zeContextCreate(adapter_.driver, &context_desc, &context_), "zeContextCreate");
    uint32_t count{};
    check(api_->zeDeviceGetCommandQueueGroupProperties(adapter_.device, &count, nullptr),
          "zeDeviceGetCommandQueueGroupProperties");
    std::vector<ze_command_queue_group_properties_t> groups(
        count,
        ze_command_queue_group_properties_t{.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
    check(api_->zeDeviceGetCommandQueueGroupProperties(adapter_.device, &count, groups.data()),
          "zeDeviceGetCommandQueueGroupProperties");
    for (std::uint32_t i = 0; i < count; ++i) {
      if (!(groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) || groups[i].numQueues == 0)
        continue;
      const ze_command_queue_desc_t config{.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                           .ordinal = i,
                                           .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS};
      check(api_->zeCommandListCreateImmediate(context_, adapter_.device, &config, &queue_),
            "zeCommandListCreateImmediate");
      return;
    }
    throw std::runtime_error("aucune file Level Zero de calcul");
  }

  std::string_view architecture() const override { return "arc-a770"; }

  vk::Done in_context(const std::function<void()>& work) override {
    try {
      work();
      return {};
    } catch (const std::exception& e) {
      return refuse(RefusalCode::VendorCallFailed, "Triton Intel : " + std::string(e.what()));
    }
  }
  Pointer allocate(std::size_t size, std::string_view label) override {
    const ze_device_mem_alloc_desc_t desc{.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* pointer{};
    check(api_->zeMemAllocDevice(context_, &desc, size, 64, adapter_.device, &pointer), label);
    return reinterpret_cast<Pointer>(pointer);
  }
  void release(Pointer address) override {
    static_cast<void>(api_->zeMemFree(context_, native_handle<void*>(address)));
  }
  void upload(Pointer address, const void* data, std::size_t size, std::string_view label) override {
    ensure_staging(size);
    std::memcpy(staging_, data, size);
    copy(native_handle<void*>(address), staging_, size, label);
  }
  void download(void* data, Pointer address, std::size_t size) override {
    ensure_staging(size);
    copy(staging_, native_handle<const void*>(address), size, "sortie Intel");
    std::memcpy(data, staging_, size);
  }
  Module load(std::span<const char> image, std::string_view name, unsigned shared) override {
    const ze_module_desc_t desc{.stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
                                .format = ZE_MODULE_FORMAT_IL_SPIRV,
                                .inputSize = image.size(),
                                .pInputModule = reinterpret_cast<const std::uint8_t*>(image.data())};
    ze_module_handle_t module{};
    check(api_->zeModuleCreate(context_, adapter_.device, &desc, &module, nullptr), name);
    ze_kernel_handle_t function{};
    try {
      const std::string entry_name(name);
      const ze_kernel_desc_t config{.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
                                    .pKernelName = entry_name.c_str()};
      check(api_->zeKernelCreate(module, &config, &function), name);
      ze_kernel_properties_t props{.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
      check(api_->zeKernelGetProperties(function, &props), name);
      if (shared && !props.numKernelArgs) throw std::runtime_error("argument LDS Intel absent");
      auto loaded =
          std::make_unique<LoadedKernel>(module, function, props.numKernelArgs - unsigned(shared != 0));
      const auto token = reinterpret_cast<Pointer>(loaded.release());
      return {token, token};
    } catch (...) {
      if (function) static_cast<void>(api_->zeKernelDestroy(function));
      static_cast<void>(api_->zeModuleDestroy(module));
      throw;
    }
  }
  void unload(Pointer address) override {
    const std::unique_ptr<LoadedKernel> loaded(native_handle<LoadedKernel*>(address));
    static_cast<void>(api_->zeKernelDestroy(loaded->function));
    static_cast<void>(api_->zeModuleDestroy(loaded->module));
  }
  void launch(const Kernel& k, std::span<void*> args) override {
    const auto& loaded = *native_handle<const LoadedKernel*>(k.function);
    const auto kernel = loaded.function;
    const auto count = loaded.arguments;
    if (count != args.size()) throw std::runtime_error("signature de lancement Level Zero incompatible");
    check(api_->zeKernelSetGroupSize(kernel, k.block, 1, 1), "zeKernelSetGroupSize");
    for (uint32_t i = 0; i < count; ++i)
      check(api_->zeKernelSetArgumentValue(kernel, i, sizeof(Pointer), args[i]), "zeKernelSetArgumentValue");
    if (k.shared)
      check(api_->zeKernelSetArgumentValue(kernel, count, k.shared, nullptr), "allocation locale Level Zero");
    const ze_group_count_t groups{k.grid, k.grid_y, k.grid_z};
    check(api_->zeCommandListAppendLaunchKernel(queue_, kernel, &groups, nullptr, 0, nullptr),
          "zeCommandListAppendLaunchKernel");
    check(api_->zeCommandListAppendBarrier(queue_, nullptr, 0, nullptr), "zeCommandListAppendBarrier");
  }
  void synchronize() override {
    check(api_->zeCommandListHostSynchronize(queue_, UINT64_MAX), "zeCommandListHostSynchronize");
  }

 private:
  void ensure_staging(std::size_t bytes) {
    if (capacity_ >= bytes) return;
    if (staging_) check(api_->zeMemFree(context_, staging_), "zeMemFree");
    staging_ = nullptr;
    capacity_ = 0;
    const ze_host_mem_alloc_desc_t desc{.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    check(api_->zeMemAllocHost(context_, &desc, bytes, 64, &staging_), "zeMemAllocHost");
    capacity_ = bytes;
  }
  void copy(void* destination, const void* source, std::size_t size, std::string_view label) {
    check(api_->zeCommandListAppendMemoryCopy(queue_, destination, source, size, nullptr, 0, nullptr), label);
    synchronize();
  }

  std::shared_ptr<Api> api_;
  Adapter adapter_;
  ze_context_handle_t context_{};
  ze_command_list_handle_t queue_{};
  void* staging_{};
  std::size_t capacity_{};
};
}  // namespace

std::expected<std::shared_ptr<Device>, Refusal> make_xpu(const vk::DeviceCaps& caps) {
  if (caps.device_id != 0x56a0)
    return refuse(RefusalCode::DeviceMismatch, "Triton XPU : le paquet cible Intel Arc A770 (56a0)");
#ifdef _WIN32
  auto library = DynamicLibrary::open_system("ze_loader.dll");
#else
  auto library = DynamicLibrary::open_system("libze_loader.so.1");
#endif
  if (!library) return std::unexpected(library.error());
  auto api = std::make_shared<Api>(std::move(*library));
  check(api->zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");
  auto adapter = select_adapter(*api, caps);
  if (!adapter) return std::unexpected(adapter.error());
  auto device = std::make_shared<XpuDevice>(std::move(api), *adapter);
  device->open();
  return device;
}
}  // namespace kirari::gpu::triton
