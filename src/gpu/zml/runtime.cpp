#include <format>

#include "core/zml_plugin.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/cuda_io.h"
#include "gpu/zml/hlo.h"
#include "gpu/zml/motion_frontend.h"
#include "gpu/zml/pjrt.h"
#include "gpu/zml/platform.h"
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
#include "gpu/zml/pjrt_transform.h"
#include "gpu/zml/span_gate_artifact.h"
#endif
#include "gpu/zml/rife_features.h"
#include "gpu/zml/sdk_process.h"
#include "gpu/zml/span_frontend.h"
#ifdef KIRARI_ZML_SPAN_MOBILE_REGIONS
#include "gpu/zml/span_regions.h"
#endif
#include "media/cuda_driver.h"

namespace kirari::gpu::zml {
void discard(const PJRT_Api& api, PJRT_Error* error) {
  if (!error) return;
  PJRT_Error_Destroy_Args args{.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = error};
  api.PJRT_Error_Destroy(&args);
}
std::expected<void, Refusal> checked(const PJRT_Api& api, PJRT_Error* error, std::string_view call) {
  if (!error) return {};
  PJRT_Error_Message_Args args{.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = error};
  api.PJRT_Error_Message(&args);
  std::string msg(args.message, args.message_size);
  discard(api, error);
  return std::unexpected(Refusal{RefusalCode::VendorCallFailed, std::format("ZML, {} : {}", call, msg)});
}
Client::~Client() {
  if (handle) {
    PJRT_Client_Destroy_Args args{.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE, .client = handle};
    discard(api, api.PJRT_Client_Destroy(&args));
  }
}
namespace {
using vk::net::RuntimeNet;
#if defined(KIRARI_ZML_SPAN_MOBILE_REGIONS) || defined(KIRARI_ZML_RIFE_FEATURES)
vk::Done optional_program(vk::Done result, Progress& progress, std::string_view model) {
  if (result) return {};
  if (progress.stop.stop_requested() || result.error().code == RefusalCode::Cancelled) return result;
  if (progress.report)
    progress.report(0.6F, std::format("{} dense retenu : {}", model, result.error().detail));
  return {};
}
#endif
class Runtime final : public vk::net::Runtime {
 public:
  Runtime(std::shared_ptr<Client> client, GpuVendor vendor) : client_(std::move(client)) {
    prepare_binary_ = vendor == GpuVendor::Nvidia && can_cache_executable(client_->api);
    std::error_code ec;
    const auto frontend = sdk_frontend_path();
    sdk_available_ = std::filesystem::is_regular_file(frontend, ec);
    recipe_ = "stablehlo-15";
    if (sdk_available_) {
      const auto bytes = std::filesystem::file_size(frontend, ec);
      if (ec) sdk_available_ = false;
      const auto installed = std::filesystem::last_write_time(frontend, ec);
      if (ec) sdk_available_ = false;
      if (sdk_available_)
        recipe_ += std::format("-sdk-v2-{}-{}", bytes, installed.time_since_epoch().count());
    }
    span_sdk_.available = sdk_available_;
    // zml_plugin_path associe Nvidia à PJRT CUDA, même sans import de mémoire externe.
    if (sdk_available_ && vendor == GpuVendor::Nvidia) configure_cuda_recipes();
    append_experimental_recipe();
  }
  std::string_view name() const override { return "zml"; }
  std::string device_name() const override { return client_->name; }
  std::string version() const override { return client_->version; }
  std::string_view recipe() const override { return recipe_; }
  vk::Done import_timeline(int fd) override { return client_->cuda->attach(fd); }
  bool shares_device_memory() const override { return static_cast<bool>(client_->cuda); }
  std::expected<vk::net::DeviceMemory, Refusal> import_device(std::uintptr_t handle, std::size_t allocation,
                                                              std::size_t bytes) override {
    return client_->cuda->import(handle, allocation, bytes);
  }
  std::string_view extension() const override { return ".mlir"; }
  std::expected<void, Refusal> build(const ModelSpec& model, std::span<const vk::net::InputShape> inputs,
                                     const std::filesystem::path& path, Progress& progress) override {
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
    auto span = build_recognized_span(model, inputs, path, span_sdk_, progress);
    if (!span) return std::unexpected(span.error());
    if (*span) {
#ifdef KIRARI_ZML_SPAN_MOBILE_REGIONS
      if (prepare_binary_ && sdk_available_ && client_->cuda && inputs.size() == 1) {
        return optional_program(prepare_span_regions(client_, path, inputs.front().dims, progress), progress,
                                "SPAN");
      }
#endif
      return {};
    }
    if (sdk_available_) {
      auto sdk = build_motion_with_sdk(model, inputs, path, rife_recipes_, drba_recipes_, gmfss_recipes_,
                                       artcnn_mixed_body_, progress);
      if (!sdk) return std::unexpected(sdk.error());
      if (*sdk) {
#ifdef KIRARI_ZML_RIFE_FEATURES
        if (rife_recipes_.features)
          return optional_program(prepare_rife_features(client_, path, progress), progress, "RIFE");
#endif
        return {};
      }
    }
    std::vector<ir::InputShape> shapes;
    for (const auto& input : inputs) shapes.push_back({input.name, input.dims});
    if (progress.report) progress.report(0.1F, "conversion ONNX vers StableHLO");
    auto hlo = lower_onnx(model.onnx_path, shapes, model.precision == TensorType::F32 ? "f32" : "f16");
    if (!hlo) return std::unexpected(hlo.error());
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
    return vk::net::write_file(path, std::as_bytes(std::span(*hlo)));
  }
  std::optional<bool> compiled_artifact_ready(const std::filesystem::path& path) const override {
    if (!prepare_binary_) return {};
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return false;
    const auto source = vk::net::read_file(path);
    if (!source) return false;
    const auto binary = executable_cache(path, *source, client_->device_id);
    if (!std::filesystem::is_regular_file(binary, ec)) return false;
    const auto size = std::filesystem::file_size(binary, ec);
    return !ec && size > 0;
  }
  std::expected<void, Refusal> prepare_compiled_artifact(const std::filesystem::path& path,
                                                         const vk::net::IoNames& names,
                                                         Progress& progress) override {
    auto source = vk::net::read_file(path);
    if (!source) return std::unexpected(source.error());
    if (progress.report) progress.report(0.65F, "compilation PJRT du réseau sur la carte");
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
    // PJRT_Client_Compile est synchrone. Une demande reçue pendant l'appel est traitée au retour.
    auto network = compile_model(*source, names, path, &progress);
    if (!network) return std::unexpected(network.error());
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
    return {};
  }
  std::expected<std::unique_ptr<RuntimeNet>, Refusal> load(const std::filesystem::path& path) override {
    auto names = vk::net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto bytes = vk::net::read_file(path);
    if (!bytes) return std::unexpected(bytes.error());
    auto network = compile_model(*bytes, *names, path);
#ifdef KIRARI_ZML_SPAN_MOBILE_REGIONS
    if (network && prepare_binary_ && client_->cuda) load_span_regions(client_, path, *bytes, **network);
#endif
#ifdef KIRARI_ZML_RIFE_FEATURES
    if (network && rife_recipes_.features) load_rife_features(client_, path, *bytes, **network);
#endif
    return network;
  }

 private:
  std::expected<std::unique_ptr<RuntimeNet>, Refusal> compile_model(std::span<const std::byte> source,
                                                                    const vk::net::IoNames& names,
                                                                    const std::filesystem::path& path,
                                                                    Progress* progress = nullptr) const {
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
    return compile_span_or_portable(client_, source, names, path, progress);
#else
    return compile(client_, source, names, path, progress);
#endif
  }

  void configure_span_cuda() {
#if defined(KIRARI_ZML_SPAN_TRITON_GATES) && !defined(KIRARI_ZML_SPAN_ATTENTION_DOT) && \
    !defined(KIRARI_ZML_SPAN_MOBILE_REGIONS) && !defined(KIRARI_ZML_SPAN_STRIPS)
    if (prepare_binary_ && client_->name == "NVIDIA GeForce RTX 4070 Ti" &&
        has_span_transform(client_->api)) {
#if KIRARI_ZML_SPAN_BALANCED_TRITON_GATES && defined(KIRARI_ZML_AUTO_SPAN_BALANCED_PROJECTION_STRIPS)
      span_sdk_.balanced_gates = true;
      recipe_ += "-span-balanced-gates8-v1";
#if KIRARI_ZML_SPAN_FUSED_PROJECTION
      span_sdk_.fused_projection = true;
      recipe_ += "-span-balanced-projection-fusion-v1";
#endif
#endif
#if KIRARI_ZML_SPAN_PERFORMANCE_TRITON_GATES && defined(KIRARI_ZML_AUTO_SPAN_PERFORMANCE_PIXEL_STRIPS)
      span_sdk_.performance_gates = true;
      recipe_ += "-span-performance-gates5-v1";
#endif
    }
#endif
#ifdef KIRARI_ZML_AUTO_SPAN_BALANCED_PROJECTION_STRIPS
    span_sdk_.balanced_strips = true;
    recipe_ += "-balanced-projection-strips30-v1";
#endif
#ifdef KIRARI_ZML_AUTO_SPAN_PERFORMANCE_PROJECTION_STRIPS
    span_sdk_.performance_strips = true;
#ifdef KIRARI_ZML_AUTO_SPAN_PERFORMANCE_PIXEL_STRIPS
    span_sdk_.pixel_strips = true;
    recipe_ += "-performance-pixel-strips30-v1";
#else
    recipe_ += "-performance-projection-strips30-v1";
#endif
#endif
  }

  void configure_cuda_recipes() {
    configure_span_cuda();
#ifdef KIRARI_ZML_ARTCNN_MIXED_BODY
    artcnn_mixed_body_ = sdk_available_ && client_->name == "NVIDIA GeForce RTX 4070 Ti";
    if (artcnn_mixed_body_) recipe_ += "-artcnn-c4-body-fp16-1080-v1";
#endif
#ifdef KIRARI_ZML_GMFSS_PAIR_SDK
    gmfss_recipes_.pair = true;
    recipe_ += "-gmflow-sdk-fp16-v4";
#endif
#ifdef KIRARI_ZML_GMFSS_MLP_STRIPS
    gmfss_recipes_ = {.pair = true, .mlp_strips = true};
    recipe_ += "-gmflow-quarter-mlp2160-v1";
#endif
#ifdef KIRARI_ZML_GMFSS_GLOBAL_MATCH
    gmfss_recipes_.pair = true;
    gmfss_recipes_.global_match = true;
    recipe_ += "-gmflow-global-36x60-online720-v1";
#endif
#ifdef KIRARI_ZML_SPAN_MOBILE_REGIONS
    if (prepare_binary_ && client_->cuda) recipe_ += "-span-regions96x256-v1";
#endif
#ifdef KIRARI_ZML_DRBA_SHARED_PROJECTION
    drba_recipes_.shared_projection = true;
    recipe_ += "-drba-shared-rgb-1280x768-v1";
#endif
#ifdef KIRARI_ZML_DRBA_OUTPUT_PIXEL
    drba_recipes_.output_pixel = true;
    recipe_ += "-drba-output-rgb64-1920x1088-v1";
#endif
#ifdef KIRARI_ZML_DRBA_GATHER_FP16
    drba_recipes_.gather_fp16 = true;
    recipe_ += "-drba-central-gather-f16-1280x768-v1";
#endif
    rife_recipes_ = {.lite422 = KIRARI_ZML_AUTO_RIFE422_PHASES,
                     .full425 = KIRARI_ZML_AUTO_RIFE425_TRANSPOSE_PHASES};
#ifdef KIRARI_ZML_RIFE_FEATURES
    rife_recipes_.features = prepare_binary_ && client_->cuda;
    if (rife_recipes_.features) recipe_ += "-rife425-features-x8-v1";
#endif
#if KIRARI_ZML_AUTO_RIFE425_LITE_D4
    rife_recipes_.lite425_d4 = true;
    recipe_ += "-rife425lite-d4-1280x768-v1";
#endif
    if (rife_recipes_.lite422) recipe_ += "-rife422-phases-v1";
    if (rife_recipes_.full425) recipe_ += "-rife425-transpose-phases-v1";
#if KIRARI_ZML_AUTO_RIFE422_D2_CENTRE && !defined(KIRARI_ZML_RIFE_D2_CENTRE)
    rife_recipes_.d2 = RifeD2Centre::Auto422;
    recipe_ += "-rife422-d2-centre-1280x768-1920x1088-v2";
#endif
#ifdef KIRARI_ZML_RIFE422_WARP_XY
    if (rife_recipes_.lite422) recipe_ += "-rife422-warp-xy-v1";
#endif
  }

  void append_experimental_recipe() {
#ifdef KIRARI_ZML_SPLIT_PROJECTION
    recipe_ += "-split-projection";
#endif
#ifdef KIRARI_ZML_FOLD_RESCONV
    if (sdk_available_) recipe_ += "-rife-resconv-select-v2";
#endif
#ifdef KIRARI_ZML_SPAN_WINOGRAD
    recipe_ += "-f23";
#endif
#ifdef KIRARI_ZML_SPAN_ATTENTION_DOT
    recipe_ += "-attention-dot";
#endif
#ifdef KIRARI_ZML_SPAN_PROJECTION_DOT
    if (sdk_available_) recipe_ += "-projection-dot";
#elif defined(KIRARI_ZML_SPAN_PROJECTION_STRIPS)
    if (sdk_available_) recipe_ += "-projection-strips30";
#endif
#ifdef KIRARI_ZML_SPAN_STRIPS
    recipe_ += "-span-strips4";
#endif
#ifdef KIRARI_ZML_POLYPHASE
    recipe_ += "-polyphase";
#endif
#ifdef KIRARI_ZML_RIFE_SELECT_ACTIVATION
    recipe_ += "-rife-leaky-select";
#endif
#ifdef KIRARI_ZML_RIFE_D2_CENTRE
    if (sdk_available_) {
      rife_recipes_.d2 = RifeD2Centre::Experimental;
      recipe_ += "-rife-d2-centre-v1";
    }
#endif
#ifdef KIRARI_ZML_RESIZE_PHASES
    recipe_ += "-resize-phases";
#endif
#ifdef KIRARI_ZML_LINEAR_WARP
    recipe_ += "-linear-warp";
#endif
#ifdef KIRARI_ZML_DRBA_DIRECT_RESIZE
    recipe_ += "-drba-direct-resize";
#endif
#ifdef KIRARI_ZML_DRBA_OUTPUT_WARP
    if (sdk_available_) recipe_ += "-drba-output-warp";
#endif
  }

  std::shared_ptr<Client> client_;
  bool prepare_binary_ = false;
  bool sdk_available_ = false;
  RifeRecipes rife_recipes_;
  DrbaRecipes drba_recipes_;
  GmfssRecipes gmfss_recipes_;
  bool artcnn_mixed_body_ = false;
  SpanSdk span_sdk_;
  std::string recipe_;
};

bool supports_executables(const PJRT_Api& api) {
  return api.PJRT_Client_Compile && api.PJRT_LoadedExecutable_Destroy &&
         api.PJRT_LoadedExecutable_GetExecutable && api.PJRT_Executable_NumOutputs &&
         api.PJRT_Executable_Destroy && api.PJRT_Client_BufferFromHostBuffer &&
         api.PJRT_LoadedExecutable_Execute && api.PJRT_Buffer_Destroy && api.PJRT_Buffer_ElementType &&
         api.PJRT_Buffer_Dimensions && api.PJRT_Buffer_ToHostBuffer && api.PJRT_Event_Destroy &&
         api.PJRT_Event_Await;
}

bool compatible_api(const PJRT_Api* api) {
  return api && api->struct_size >= PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Buffer_ToHostBuffer) &&
         api->pjrt_api_version.major_version == PJRT_API_MAJOR && api->PJRT_Error_Destroy &&
         api->PJRT_Error_Message && api->PJRT_Plugin_Initialize && api->PJRT_Client_Create &&
         api->PJRT_Client_Destroy && api->PJRT_Client_AddressableDevices &&
         api->PJRT_Client_PlatformVersion && api->PJRT_Device_GetDescription &&
         api->PJRT_DeviceDescription_Kind && api->PJRT_DeviceDescription_Id && supports_executables(*api);
}

std::expected<void, Refusal> select_render_device(Client& client, const vk::DeviceCaps& caps,
                                                  GpuVendor vendor) {
  const auto& api = client.api;
  PJRT_Client_AddressableDevices_Args devices{.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
                                              .client = client.handle};
  if (auto ok = checked(api, api.PJRT_Client_AddressableDevices(&devices), "cartes disponibles"); !ok)
    return std::unexpected(ok.error());
  std::optional<int> render_ordinal;
  if (vendor == GpuVendor::Nvidia) {
    if (auto driver = media::CudaDriver::load()) {
      if (auto ordinal = (*driver)->ordinal_of(caps.uuid)) render_ordinal = *ordinal;
    }
  }
  for (auto* device : std::span(devices.addressable_devices, devices.num_addressable_devices)) {
    PJRT_Device_GetDescription_Args desc{.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
                                         .device = device};
    if (auto ok = checked(api, api.PJRT_Device_GetDescription(&desc), "description de la carte"); !ok)
      return std::unexpected(ok.error());
    PJRT_DeviceDescription_Kind_Args kind{.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE,
                                          .device_description = desc.device_description};
    if (auto ok = checked(api, api.PJRT_DeviceDescription_Kind(&kind), "modèle de la carte"); !ok)
      return std::unexpected(ok.error());
    const std::string name(kind.device_kind, kind.device_kind_size);
    std::optional<int> local_id;
    if (api.struct_size >= PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Device_LocalHardwareId) &&
        api.PJRT_Device_LocalHardwareId) {
      PJRT_Device_LocalHardwareId_Args local{.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
                                             .device = device};
      if (auto result = checked(api, api.PJRT_Device_LocalHardwareId(&local), "identifiant local"); result)
        local_id = local.local_hardware_id;
    }
    if (render_ordinal && local_id) {
      if (*render_ordinal != *local_id) continue;
    } else if (devices.num_addressable_devices > 1 && name != caps.device_name) {
      continue;
    }
    if (client.device)
      return std::unexpected(
          Refusal{RefusalCode::DeviceMismatch, "ZML : plusieurs cartes correspondent au rendu"});
    PJRT_DeviceDescription_Id_Args id{.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE,
                                      .device_description = desc.device_description};
    if (auto ok = checked(api, api.PJRT_DeviceDescription_Id(&id), "identifiant de la carte"); !ok)
      return std::unexpected(ok.error());
    client.device = device;
    client.name = name;
    client.device_id = id.id;
    client.local_id = local_id;
  }
  if (!client.device)
    return std::unexpected(Refusal{RefusalCode::DeviceMismatch, "ZML : carte de rendu introuvable"});
  return {};
}

std::expected<std::shared_ptr<Client>, Refusal> open(const vk::DeviceCaps& caps) {
  const auto vendor = vendor_from_pci(caps.vendor_id);
  const auto module = library_containing(reinterpret_cast<const void*>(&make_runtime));
  const auto path = zml_plugin_path(module, vendor);
  if (path.empty())
    return std::unexpected(Refusal{RefusalCode::LibraryUnavailable,
                                   std::format("ZML : aucun plugin configuré pour {}", caps.device_name)});
  prepare_platform(vendor, path);
  auto lib = DynamicLibrary::open(path, SymbolScope::GlobalResident);
  if (!lib) return std::unexpected(lib.error());
  auto entry = lib->symbol<const PJRT_Api*()>("GetPjrtApi");
  if (!entry) return std::unexpected(entry.error());
  const auto* api = (*entry)();
  if (!compatible_api(api))
    return std::unexpected(Refusal{RefusalCode::SdkRejected, "ZML : version de PJRT incompatible"});
  auto c = std::make_shared<Client>(std::move(*lib), *api);
  PJRT_Plugin_Initialize_Args init{.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
  if (auto ok = checked(*api, api->PJRT_Plugin_Initialize(&init), "initialisation"); !ok)
    return std::unexpected(ok.error());
  PJRT_NamedValue preallocate{.struct_size = PJRT_NamedValue_STRUCT_SIZE,
                              .name = "preallocate",
                              .name_size = 11,
                              .type = PJRT_NamedValue_kBool,
                              .bool_value = false,
                              .value_size = 1};
  PJRT_Client_Create_Args create{.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
  if (vendor == GpuVendor::Nvidia || vendor == GpuVendor::Amd) {
    create.create_options = &preallocate;
    create.num_options = 1;
  }
  if (auto ok = checked(*api, api->PJRT_Client_Create(&create), "création du client"); !ok)
    return std::unexpected(ok.error());
  c->handle = create.client;
  PJRT_Client_PlatformVersion_Args version{.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE,
                                           .client = c->handle};
  if (auto ok = checked(*api, api->PJRT_Client_PlatformVersion(&version), "version"); !ok)
    return std::unexpected(ok.error());
  c->version = std::format("{} (PJRT {}.{})",
                           std::string_view(version.platform_version, version.platform_version_size),
                           api->pjrt_api_version.major_version, api->pjrt_api_version.minor_version);
  if (auto ok = select_render_device(*c, caps, vendor); !ok) return std::unexpected(ok.error());
  if (vendor == GpuVendor::Nvidia) c->cuda = CudaIo::open(*c, caps.uuid);

  return c;
}
}  // namespace
std::expected<std::shared_ptr<vk::net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) {
  auto client = open(caps);
  if (!client) return std::unexpected(client.error());
  return std::make_shared<Runtime>(std::move(*client), vendor_from_pci(caps.vendor_id));
}
}  // namespace kirari::gpu::zml
