#include "gpu/zml/pjrt_transform.h"

#include <string_view>

#include "gpu/zml/pjrt.h"
#include "gpu/zml/span_gate_profile.h"

#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>

#include "gpu/zml/span_gate_hlo.h"
#endif

namespace kirari::gpu::zml {
namespace {
#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
// Le registre XLA appartient au greffon, commun à tous les clients. Les compilations
// ordinaires restent concurrentes. Seule une compilation du module spécial prend l'exclusivité.
enum class Registration { Idle, Requested, Applied, Unavailable };
struct TransformRegistry {
  std::shared_mutex mutex;
  std::atomic<Registration> registration{Registration::Idle};
};
TransformRegistry& transform_registry() {
  static TransformRegistry registry;
  return registry;
}
constexpr std::string_view transform_name = "kirari.span.gates.v1";
constexpr auto stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;

const PJRT_Xla_Transform_Extension* extension(const PJRT_Api& api) {
  for (auto* e = api.extension_start; e; e = e->next) {
    if (e->type != PJRT_Extension_Type_XlaTransform ||
        e->struct_size < PJRT_STRUCT_SIZE(PJRT_Xla_Transform_Extension, clear_xla_transform))
      continue;
    const auto* ext = reinterpret_cast<const PJRT_Xla_Transform_Extension*>(e);
    if (ext->register_xla_transform && ext->clear_xla_transform) return ext;
  }
  return nullptr;
}

void callback_error(PJRT_XlaTransform_Args& args, std::string_view message) noexcept {
  args.header.has_error = true;
  args.header.code = PJRT_Error_Code_INTERNAL;
  args.header.error_msg = {message.data(), message.size()};
}

void transform(PJRT_XlaTransform_Callbacks*, PJRT_XlaTransform_Args* args) noexcept {
  if (args->header.api_version != PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION ||
      args->struct_size < PJRT_STRUCT_SIZE(PJRT_XlaTransform_Args, changed)) {
    callback_error(*args, "Version incompatible de la transformation HLO");
    return;
  }
  args->changed = false;
  auto& registration = transform_registry().registration;
  const auto current = registration.load();
  if (current != Registration::Requested && current != Registration::Applied) return;
  try {
    auto updated =
        rewrite_span_triton_gates(std::as_bytes(std::span(args->hlo_module.data, args->hlo_module.size)));
    if (!updated) {
      auto* message = new std::string(std::move(updated.error()));
      args->header.data = message;
      args->header.cleanup_fn = [](void* data) { delete static_cast<std::string*>(data); };
      callback_error(*args, *message);
      return;
    }
    if (updated->empty()) return;
    auto* out = new std::vector<std::byte>(std::move(*updated));
    args->header.data = out;
    args->header.cleanup_fn = [](void* data) { delete static_cast<std::vector<std::byte>*>(data); };
    args->transformed_hlo_module = {reinterpret_cast<const char*>(out->data()), out->size()};
    args->changed = true;
    auto requested = Registration::Requested;
    registration.compare_exchange_strong(requested, Registration::Applied);
  } catch (...) {
    callback_error(*args, "Exception pendant la transformation des portes SPAN");
  }
}

std::expected<void, Refusal> compile_gates(const PJRT_Api& api, const PJRT_Xla_Transform_Extension& ext,
                                           PJRT_Client_Compile_Args& args) {
  // Un greffon peut retenir les callbacks jusqu'à la fin des passes en cours.
  static PJRT_XlaTransform_Callbacks callbacks{.version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION,
                                               .transform_hlo_module = transform};
  auto& registry = transform_registry();
  std::unique_lock lock(registry.mutex);
  auto& registration = registry.registration;
  if (registration.load() == Registration::Unavailable)
    return refuse(RefusalCode::SdkRejected, "PJRT : un précédent retrait de la transformation SPAN a échoué");
  PJRT_Register_Xla_Transform_Args add{.struct_size = PJRT_Register_Xla_Transform_Args_STRUCT_SIZE,
                                       .name = transform_name.data(),
                                       .name_size = transform_name.size(),
                                       .stage = stage,
                                       .callbacks = &callbacks};
  if (auto ok = checked(api, ext.register_xla_transform(&add), "inscription des portes SPAN"); !ok) return ok;
  registration.store(Registration::Requested);
  auto* compile_error = api.PJRT_Client_Compile(&args);
  PJRT_Clear_Xla_Transform_Args remove{.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE,
                                       .stage = stage,
                                       .name = transform_name.data(),
                                       .name_size = transform_name.size()};
  auto unregistered = checked(api, ext.clear_xla_transform(&remove), "retrait des portes SPAN");
  const auto result =
      registration.exchange(unregistered && remove.cleared ? Registration::Idle : Registration::Unavailable);
  if (auto ok = checked(api, compile_error, "compilation des portes SPAN"); !ok) return ok;
  if (unregistered && remove.cleared && result == Registration::Applied) return {};
  PJRT_LoadedExecutable_Destroy_Args release{.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
                                             .executable = args.executable};
  discard(api, api.PJRT_LoadedExecutable_Destroy(&release));
  args.executable = nullptr;
  if (!unregistered) return unregistered;
  return refuse(RefusalCode::SdkRejected, "PJRT : transformation des portes SPAN non confirmée");
}
#endif
}  // namespace

bool has_span_transform(const PJRT_Api& api) {
#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
  return extension(api) != nullptr;
#else
  (void)api;
  return false;
#endif
}

std::expected<void, Refusal> compile_with_span_transform(const PJRT_Api& api,
                                                         PJRT_Client_Compile_Args& args) {
  const auto* program = args.program;
  const auto module = span_gate_module({program->code, program->code_size});
  if (span_gate_profile(module)) {
#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
    if (const auto* ext = extension(api)) return compile_gates(api, *ext, args);
#endif
    return refuse(RefusalCode::SdkRejected, "PJRT : extension XlaTransform indisponible");
  }
#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
  std::shared_lock lock(transform_registry().mutex);
#endif
  return checked(api, api.PJRT_Client_Compile(&args), "compilation StableHLO");
}
}  // namespace kirari::gpu::zml
