#include <algorithm>
#include <array>
#include <numeric>

#include "gpu/ir/protobuf.h"
#include "gpu/vulkan/backend/deferred_steps.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/cuda_io.h"
#include "gpu/zml/drba_pair12.h"
#include "gpu/zml/pjrt.h"
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
#include "gpu/zml/pjrt_transform.h"
#endif
#include "gpu/zml/rife_features.h"
#include "prepare/key_lock.h"

namespace kirari::gpu::zml {
bool can_cache_executable(const PJRT_Api& api) {
  return api.struct_size >= PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Executable_DeserializeAndLoad) &&
         api.PJRT_Executable_DeserializeAndLoad && api.PJRT_Executable_Serialize;
}

std::filesystem::path executable_cache(const std::filesystem::path& path, std::span<const std::byte> source,
                                       int device) {
  auto cache = path;
  cache += std::format(".{:016x}.{}.pjrt", vk::net::fnv1a(source), device);
  return cache;
}

namespace {
using vk::net::HostBinding;

std::vector<std::byte> compile_options(int device) {
  // CompileOptionsProto.executable_build_options.device_assignment, une réplique sur la carte de rendu.
  ir::proto::Writer computation, assignment, build, options;
  computation.varint(1, static_cast<unsigned>(device));
  assignment.varint(1, 1);
  assignment.varint(2, 1);
  assignment.message(3, computation);
  build.varint(4, 1);
  build.varint(5, 1);
  build.message(9, assignment);
  options.message(3, build);
  return std::move(options).take();
}

std::expected<PJRT_LoadedExecutable*, Refusal> compile_program(const Client& client,
                                                               std::span<const std::byte> source) {
  PJRT_Program program{.struct_size = PJRT_Program_STRUCT_SIZE,
                       .code = const_cast<char*>(reinterpret_cast<const char*>(source.data())),
                       .code_size = source.size(),
                       .format = "mlir",
                       .format_size = 4};
  auto options = compile_options(client.device_id);
  PJRT_Client_Compile_Args args{.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                .client = client.handle,
                                .program = &program,
                                .compile_options = reinterpret_cast<const char*>(options.data()),
                                .compile_options_size = options.size()};
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
  if (auto ok = compile_with_span_transform(client.api, args); !ok) return std::unexpected(ok.error());
#else
  if (auto ok = checked(client.api, client.api.PJRT_Client_Compile(&args), "compilation StableHLO"); !ok)
    return std::unexpected(ok.error());
#endif
  return args.executable;
}

void clear_event(const PJRT_Api& api, PJRT_Event* event) {
  if (!event) return;
  PJRT_Event_Destroy_Args destroy{.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = event};
  discard(api, api.PJRT_Event_Destroy(&destroy));
}
struct Buffers {
  const PJRT_Api& api;
  std::vector<PJRT_Buffer*> items;
  bool borrowed_tail = false;  // dernier argument PJRT possédé par Loan::references, au lieu de cette liste
  ~Buffers() {
    for (auto* item : std::span(items).first(items.size() - static_cast<std::size_t>(borrowed_tail)))
      if (item) {
        PJRT_Buffer_Destroy_Args args{.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = item};
        discard(api, api.PJRT_Buffer_Destroy(&args));
      }
  }
};
struct Loan {
  explicit Loan(std::shared_ptr<Client> c, std::size_t count)
      : client(std::move(c)), ins{client->api, {}}, outs{client->api, std::vector<PJRT_Buffer*>(count)} {}
  std::shared_ptr<Client> client;
  Buffers ins, outs;
  std::vector<std::shared_ptr<void>> references;
};
PJRT_Buffer_Type element(TensorType type) {
  if (type == TensorType::F16) return PJRT_Buffer_Type_F16;
  if (type == TensorType::F32) return PJRT_Buffer_Type_F32;
  return PJRT_Buffer_Type_INVALID;
}

std::expected<void, Refusal> serialize(const PJRT_Api& api, PJRT_Executable* executable,
                                       const std::filesystem::path& cache) {
  PJRT_Executable_Serialize_Args save{.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE,
                                      .executable = executable};
  if (auto ok = checked(api, api.PJRT_Executable_Serialize(&save), "sérialisation de l'exécutable"); !ok)
    return ok;
  std::expected<void, Refusal> status;
  if (!save.serialized_executable_deleter || !save.serialized_bytes || !save.serialized_bytes_size)
    status = std::unexpected(Refusal{RefusalCode::CacheIo, "ZML : sérialisation de l'exécutable vide"});
  else
    status = vk::net::write_file(cache,
                                 std::as_bytes(std::span(save.serialized_bytes, save.serialized_bytes_size)));
  if (save.serialized_executable_deleter) save.serialized_executable_deleter(save.serialized_executable);
  return status;
}

class Network final : public vk::net::RuntimeNet, public FeatureExecutable {
 public:
  Network(std::shared_ptr<Client> client, vk::net::IoNames names)
      : client_(std::move(client)), names_(std::move(names)), input_indices_(names_.inputs.size()) {
    std::iota(input_indices_.begin(), input_indices_.end(), std::int64_t{0});
  }
  ~Network() override {
    steps_.reset();  // Termine aussi les téléchargements vers les tenseurs Vulkan.
    if (exec_) {
      PJRT_LoadedExecutable_Destroy_Args args{.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
                                              .executable = exec_};
      discard(client_->api, client_->api.PJRT_LoadedExecutable_Destroy(&args));
    }
  }
  std::expected<void, Refusal> build(std::span<const std::byte> source, const std::filesystem::path& path,
                                     Progress* preparation, bool cached_only) {
    const auto& api = client_->api;
    const auto cache = executable_cache(path, source, client_->device_id);
    const bool cacheable = can_cache_executable(api);
    const auto try_load = [&] {
      auto bytes = vk::net::read_file(cache);
      if (bytes) {
        PJRT_Executable_DeserializeAndLoad_Args load{
            .struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE,
            .client = client_->handle,
            .serialized_executable = reinterpret_cast<const char*>(bytes->data()),
            .serialized_executable_size = bytes->size()};
        if (auto* error = api.PJRT_Executable_DeserializeAndLoad(&load))
          discard(api, error);
        else
          exec_ = load.loaded_executable;
      }
    };
    std::optional<prepare::KeyLock> held;
    if (cached_only) {
      if (cacheable) try_load();
      if (!exec_)
        return std::unexpected(Refusal{RefusalCode::CacheIo, "exécutable régional ZML indisponible"});
    } else if (cacheable && !preparation) {
      try_load();
      if (!exec_) {
        // Un chargement direct peut rencontrer un ancien cache source seul, ou un binaire
        // incompatible. Revoir ce binaire sous le même verrou que le constructeur du lecteur.
        auto lock = prepare::KeyLock::acquire(vk::net::artifact_lock_path(path));
        if (lock) {
          held.emplace(std::move(*lock));
          try_load();
        } else if (lock.error().code != RefusalCode::CacheIo) {
          return std::unexpected(lock.error());
        }
      }
    }
    const bool rebuilt = !exec_;
    if (rebuilt) {
      auto compiled = compile_program(*client_, source);
      if (!compiled) return std::unexpected(compiled.error());
      exec_ = *compiled;
    }
    PJRT_LoadedExecutable_GetExecutable_Args get{
        .struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE, .loaded_executable = exec_};
    if (auto ok = checked(api, api.PJRT_LoadedExecutable_GetExecutable(&get), "interface compilée"); !ok)
      return ok;
    PJRT_Executable_NumOutputs_Args count{.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE,
                                          .executable = get.executable};
    auto status = checked(api, api.PJRT_Executable_NumOutputs(&count), "nombre de sorties");
    if (status && count.num_outputs != names_.outputs.size())
      status = std::unexpected(Refusal{RefusalCode::InputMismatch, "ZML : nombre de sorties incompatible"});
    if (preparation && preparation->stop.stop_requested())
      status = std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
    if (status && rebuilt && cacheable) {
      auto saved = serialize(api, get.executable, cache);
      if (preparation) status = std::move(saved);
    }
    PJRT_Executable_Destroy_Args destroy{.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE,
                                         .executable = get.executable};
    discard(api, api.PJRT_Executable_Destroy(&destroy));
    if (!status) return status;
    if (!preparation) steps_.emplace("zml", "ZML", [this](auto in, auto out) { return run(in, out); });
    return {};
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
  std::expected<void, Refusal> run_deferred(std::span<const HostBinding> in, std::span<const HostBinding> out,
                                            std::function<std::expected<void, Refusal>()> ready,
                                            std::function<void(std::expected<void, Refusal>)> done) override {
    if (!steps_) return refuse(RefusalCode::FeatureUnavailable, "ZML : exécution différée indisponible");
    return steps_->push(in, out, std::move(ready), std::move(done));
  }

  std::expected<void, Refusal> run(std::span<const HostBinding> inputs,
                                   std::span<const HostBinding> outputs) override {
    Loan work(client_, outputs.size());
    return execute(inputs, outputs, work, false);
  }
  std::expected<std::shared_ptr<void>, Refusal> run_device(std::span<const HostBinding> inputs,
                                                           std::span<const HostBinding> outputs,
                                                           std::uint64_t wait, std::uint64_t done) override {
    return run_features(inputs, outputs, wait, done, nullptr);
  }
  std::expected<std::shared_ptr<void>, Refusal> run_features(std::span<const HostBinding> inputs,
                                                             std::span<const HostBinding> outputs,
                                                             std::uint64_t wait, std::uint64_t done,
                                                             FeatureIo* features) override {
    auto work = std::make_shared<Loan>(client_, outputs.size() + (features && features->save));
    if (auto ok = client_->cuda->wait(wait); !ok) return std::unexpected(ok.error());
    auto ok = execute(inputs, outputs, *work, true, features);
    if (ok) ok = client_->cuda->signal(done);
    if (!ok) {
      if (auto drain = client_->cuda->drain(); !drain) return std::unexpected(drain.error());
      return std::unexpected(ok.error());
    }
    return work;
  }

 private:
  vk::Done import_inputs(std::span<const HostBinding> inputs, Buffers& ins) {
    const auto& api = client_->api;
    ins.items.reserve(names_.inputs.size());
    for (const auto& in : inputs) {
      if (!valid(in))
        return std::unexpected(Refusal{RefusalCode::InputMismatch, "ZML : tampon d’entrée invalide"});
      if (in.device && client_->cuda) {
        auto view = client_->cuda->view(*client_, in, element(in.type));
        if (!view) return std::unexpected(view.error());
        ins.items.push_back(*view);
        continue;
      }
      PJRT_Client_BufferFromHostBuffer_Args args{
          .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
          .client = client_->handle,
          .data = in.data,
          .type = element(in.type),
          .dims = in.shape.dims.data(),
          .num_dims = in.shape.dims.size(),
          .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
          .device = client_->device};
      if (auto ok = checked(api, api.PJRT_Client_BufferFromHostBuffer(&args), "transfert de l’entrée"); !ok)
        return ok;
      ins.items.push_back(args.buffer);
      clear_event(api, args.done_with_host_buffer);
    }
    return {};
  }
  std::shared_ptr<PJRT_Buffer> own(PJRT_Buffer* buffer) const {
    return {buffer, [client = client_](PJRT_Buffer* owned) {
              PJRT_Buffer_Destroy_Args args{.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
                                            .buffer = owned};
              discard(client->api, client->api.PJRT_Buffer_Destroy(&args));
            }};
  }
  vk::Done execute(std::span<const HostBinding> inputs, std::span<const HostBinding> outputs, Loan& loan,
                   bool async, FeatureIo* features = nullptr) {
    if (inputs.size() + (features && features->previous) != names_.inputs.size() ||
        loan.outs.items.size() != names_.outputs.size())
      return std::unexpected(Refusal{RefusalCode::InputMismatch, "ZML : interface inattendue"});
    const auto& api = client_->api;
    auto& ins = loan.ins;
    auto& outs = loan.outs;
    if (auto ok = import_inputs(inputs, ins); !ok) return ok;
    if (features && features->previous) {
      loan.references.push_back(features->previous);
      ins.items.push_back(features->previous.get());
      ins.borrowed_tail = true;
    }
    auto* arg_list = ins.items.data();
    auto* out_list = outs.items.data();
    PJRT_Event* completed = nullptr;
    const bool shared = std::ranges::any_of(inputs, [](const auto& b) { return b.device != 0; });
    // Vulkan peut relire ces entrées après le signal de retour, y compris le SPAN précédent.
    // Un alias choisi par le compilateur doit donc conserver leur contenu pendant le calcul.
    PJRT_ExecuteOptions options{.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
                                .non_donatable_input_indices = shared ? input_indices_.data() : nullptr,
                                .num_non_donatable_input_indices = shared ? input_indices_.size() : 0};
    PJRT_LoadedExecutable_Execute_Args execute{.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
                                               .executable = exec_,
                                               .options = &options,
                                               .argument_lists = &arg_list,
                                               .num_devices = 1,
                                               .num_args = ins.items.size(),
                                               .output_lists = &out_list,
                                               .device_complete_events = shared ? &completed : nullptr};
    if (auto ok = checked(api, api.PJRT_LoadedExecutable_Execute(&execute), "exécution"); !ok) return ok;
    std::expected<void, Refusal> status;
    for (std::size_t i = 0; i < outputs.size() && status; ++i)
      status = download(outs.items[i], outputs[i], async ? &loan.references : nullptr);
    // Même si une sortie est refusée, les calculs qui lisent les vues doivent finir avant de rendre Vulkan
    // propriétaire.
    if (completed && (!async || !status)) {
      PJRT_Event_Await_Args wait{.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = completed};
      auto done = checked(api, api.PJRT_Event_Await(&wait), "fin du prêt GPU");
      if (!done && status) status = done;
    }
    clear_event(api, completed);
    if (status && features && features->save) {
      if (!outs.items.back())
        return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "ZML : traits RIFE absents"});
      // Le prochain instant reprend ce tampon PJRT, y compris si Vulkan termine encore la passe courante.
      features->next = own(std::exchange(outs.items.back(), nullptr));
      loan.references.push_back(features->next);
    }
    return status;
  }

  static bool valid(const HostBinding& b) {
    return (b.data || b.device) && element(b.type) != PJRT_Buffer_Type_INVALID && b.bytes > 0 &&
           b.bytes == dense_tensor_bytes(b.type, b.shape);
  }
  std::expected<void, Refusal> download(PJRT_Buffer* src, const HostBinding& out,
                                        std::vector<std::shared_ptr<void>>* keep) {
    const auto& api = client_->api;
    if (!src || !valid(out))
      return std::unexpected(Refusal{RefusalCode::InputMismatch, "ZML : tampon de sortie invalide"});
    PJRT_Buffer_ElementType_Args dt{.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE, .buffer = src};
    if (auto ok = checked(api, api.PJRT_Buffer_ElementType(&dt), "type de sortie"); !ok) return ok;
    PJRT_Buffer_Dimensions_Args shape{.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE, .buffer = src};
    if (auto ok = checked(api, api.PJRT_Buffer_Dimensions(&shape), "forme de sortie"); !ok) return ok;
    if (element(out.type) != dt.type ||
        !std::ranges::equal(std::span(shape.dims, shape.num_dims), out.shape.dims))
      return std::unexpected(
          Refusal{RefusalCode::ShapeInvalid, "ZML : type ou dimensions de la sortie incompatibles"});
    constexpr std::array<std::int64_t, 4> order{3, 2, 1, 0};
    PJRT_Buffer_MemoryLayout layout{.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE,
                                    .tiled = {.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE,
                                              .minor_to_major = order.data(),
                                              .minor_to_major_size = order.size()},
                                    .type = PJRT_Buffer_MemoryLayout_Type_Tiled};
    if (out.device && client_->cuda) return client_->cuda->copy(*client_, src, out, keep);
    PJRT_Buffer_ToHostBuffer_Args copy{.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
                                       .src = src,
                                       .host_layout = &layout,
                                       .dst = out.data,
                                       .dst_size = static_cast<std::size_t>(out.bytes)};
    if (auto ok = checked(api, api.PJRT_Buffer_ToHostBuffer(&copy), "transfert de sortie"); !ok) return ok;
    PJRT_Event_Await_Args wait{.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = copy.event};
    auto ready = checked(api, api.PJRT_Event_Await(&wait), "fin du transfert");
    clear_event(api, copy.event);
    return ready;
  }

  std::shared_ptr<Client> client_;
  vk::net::IoNames names_;
  std::vector<std::int64_t> input_indices_;
  PJRT_LoadedExecutable* exec_ = nullptr;
  std::optional<vk::net::DeferredSteps> steps_;
};
}  // namespace
std::expected<std::unique_ptr<vk::net::RuntimeNet>, Refusal> compile(
    std::shared_ptr<Client> client, std::span<const std::byte> program, vk::net::IoNames names,
    const std::filesystem::path& path, Progress* preparation, bool cached_only) {
  auto net = std::make_unique<Network>(std::move(client), std::move(names));
  if (auto ok = net->build(program, path, preparation, cached_only); !ok) return std::unexpected(ok.error());
#ifdef KIRARI_ZML_DRBA_PACK_PAIR12
  if (!preparation && net->input_names().size() == 2 && net->input_names()[0] == "pair01" &&
      net->input_names()[1] == "pair12") {
    const std::string_view source(reinterpret_cast<const char*>(program.data()), program.size());
    net->pair12_rgb_only = reads_only_pair12_rgb(source);
  }
#endif
  return net;
}
std::expected<std::unique_ptr<FeatureExecutable>, Refusal> load_feature_program(
    std::shared_ptr<Client> client, std::span<const std::byte> source, vk::net::IoNames names,
    const std::filesystem::path& path) {
  auto net = std::make_unique<Network>(std::move(client), std::move(names));
  if (auto ok = net->build(source, path, nullptr, true); !ok) return std::unexpected(ok.error());
  return net;
}
}  // namespace kirari::gpu::zml
