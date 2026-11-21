// Le backend Vulkan dont les réseaux passent par un autre runtime (net_runtime.h) : construction
// et chargement par le runtime, tenseurs en mémoire hôte visible (celle du runtime, importée, sur
// un GPU intégré), enqueue différé au fil du runtime (run_deferred : l'attente des entrées et le signal de
// l'étape hors du fil du GPU) quand la file n'est qu'au backend, sinon synchrone sur ce fil après l'attente
// de la file.
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "gpu/drba_pair12.h"
#include "gpu/vulkan/backend/host_tensor.h"
#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/backend/vulkan_backend_detail.h"
#include "gpu/vulkan/compute/buffer.h"
#include "prepare/key_lock.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace kirari::gpu::vk {
namespace {

// Échecs de suite d'étapes différées au-delà desquels le backend refuse tout nouveau réseau.
constexpr int kMaxStepFailures = 3;

using detail::refuse_backend;

// Le temps d'un réseau du runtime, des entrées prêtes (l'attente de la file finie) aux sorties
// écrites, mesuré sur l'hôte (le runtime y rend la main) ; résumé au journal du backend quand le
// moteur est rendu. Une image en retard se lit alors du réseau ou du reste de la chaîne.
class RuntimeTiming {
 public:
  using Clock = std::chrono::steady_clock;
  void add(Clock::duration took, bool shared) {
    const std::scoped_lock lock(mutex_);
    if (shared) ++shared_;
    ++total_;
    if (samples_.size() < kMaxSamples)
      samples_.push_back(std::chrono::duration<float, std::milli>(took).count());
  }
  std::array<std::size_t, 3> transfers() {
    const std::scoped_lock lock(mutex_);
    return {shared_, total_, regional_};
  }
  void submitted(bool regional) {
    const std::scoped_lock lock(mutex_);
    ++shared_;
    ++total_;
    if (regional) ++regional_;
  }
  // « médiane 18.20 ms, p95 21.04 ms (1234 passes) », ou rien sans mesure.
  std::optional<std::string> summary() {
    const std::scoped_lock lock(mutex_);
    if (samples_.empty()) return std::nullopt;
    std::ranges::sort(samples_);
    return std::format("médiane {:.2f} ms, p95 {:.2f} ms ({} passes)", samples_[samples_.size() / 2],
                       samples_[samples_.size() * 95 / 100], samples_.size());
  }

 private:
  static constexpr std::size_t kMaxSamples = std::size_t{1} << 18;
  std::mutex mutex_;  // le fil du runtime (étapes différées) ou celui du GPU (run)
  std::vector<float> samples_;
  std::size_t shared_ = 0, total_ = 0, regional_ = 0;
};

class RuntimeEngine : public Engine, public RifeFeatureHints {
 public:
  RuntimeEngine(std::unique_ptr<net::RuntimeNet> net, net::IoNames names, std::string label, BackendLog log)
      : label_(std::move(label)), log_(std::move(log)), net_(std::move(net)), names_(std::move(names)) {}
  RuntimeEngine(const RuntimeEngine&) = delete;
  RuntimeEngine& operator=(const RuntimeEngine&) = delete;
  RuntimeEngine(RuntimeEngine&&) = delete;
  RuntimeEngine& operator=(RuntimeEngine&&) = delete;
  // Le réseau d'abord : ses étapes en file faites et mesurées, puis le résumé.
  ~RuntimeEngine() override {
    if (log_ && net_->rife_features) log_(BackendLogLevel::Info, net_->rife_features->summary());
    if (log_ && net_->time_reuse) log_(BackendLogLevel::Info, net_->time_reuse->summary());
    net_.reset();
    if (log_) {
      const auto [shared, total, regional] = timing_->transfers();
      if (shared)
        log_(BackendLogLevel::Info,
             std::format("interop · {} : entrée GPU partagée, sortie GPU→GPU pour {}/{} passes", label_,
                         shared, total));
      if (regional)
        log_(BackendLogLevel::Info, std::format("SPAN régional · {} : {} soumissions GPU", label_, regional));
      if (const auto summary = timing_->summary())
        log_(BackendLogLevel::Info, std::format("profil · {} : {}", label_, *summary));
    }
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
  void next_rife_features(RifeFeatureRequest request) override { features_ = request; }
  void next_times_only() override { times_only_ = net_->time_reuse != nullptr; }
  RifeFeatureRequest take_features() { return std::exchange(features_, RifeFeatureRequest::Disabled); }
  bool take_times_only() { return std::exchange(times_only_, false); }
  void forget_features() {
    if (net_->rife_features) net_->rife_features->forget();
    if (net_->time_reuse) net_->time_reuse->forget();
  }
  struct RegionJob {
    std::optional<Tensor> previous;
    std::vector<std::uint16_t> metadata;
  };
  kernels::SpanTileGrid span_tile_grid() const override {
    return net_->span_regions ? net_->span_regions->grid() : kernels::SpanTileGrid{};
  }
  bool next_span_tiles(std::span<const std::uint8_t> core, int dx, int dy, const Tensor* previous) override {
    if (!net_->span_regions) return false;
    const std::scoped_lock lock(region_mutex_);
    pending_.reset();
    if (dx || dy || core.empty()) return false;
    if (previous &&
        (previous->type != TensorType::F16 || previous->shape != net_->span_regions->previous_shape()))
      return false;
    try {
      auto metadata = net_->span_regions->plan(core);
      if (metadata.empty()) return false;
      pending_ = RegionJob{previous ? std::optional<Tensor>(*previous) : std::nullopt, std::move(metadata)};
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  }
  std::optional<RegionJob> take_region() {
    if (!net_->span_regions) return {};
    const std::scoped_lock lock(region_mutex_);
    return std::exchange(pending_, std::nullopt);
  }
  Done check_bindings(std::size_t inputs, std::size_t outputs) const {
    if (inputs != input_names().size() || outputs != output_names().size()) {
      return refuse_backend(RefusalCode::InputMismatch,
                            std::format("{} entrées et {} sorties, {} et {} attendues", inputs, outputs,
                                        input_names().size(), output_names().size()));
    }
    return {};
  }
  net::RuntimeNet& net() { return *net_; }
  const std::shared_ptr<RuntimeTiming>& timing() const { return timing_; }

 private:
  RifeFeatureRequest features_ = RifeFeatureRequest::Disabled;
  bool times_only_ = false;
  std::string label_;
  BackendLog log_;
  std::shared_ptr<RuntimeTiming> timing_ = std::make_shared<RuntimeTiming>();
  std::unique_ptr<net::RuntimeNet> net_;
  std::optional<RegionJob> pending_;
  std::mutex region_mutex_;
  net::IoNames names_;
};

class DrbaRgbRuntimeEngine final : public RuntimeEngine, public DrbaPair12Writer {
 public:
  DrbaRgbRuntimeEngine(std::unique_ptr<net::RuntimeNet> net, net::IoNames names, std::string label,
                       BackendLog log, VulkanKernelOps& kernels)
      : RuntimeEngine(std::move(net), std::move(names), std::move(label), std::move(log)),
        kernels_(kernels) {}
  Done write_third_image(const Tensor& image, const Tensor& pair12) override {
    return kernels_.write_drba_third_image(image, pair12);
  }

 private:
  VulkanKernelOps& kernels_;
};

// « coreml 2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16 » : le runtime et
// le nom de l'artefact sans sa clé.
std::string runtime_label(std::string_view runtime, const std::filesystem::path& artifact) {
  std::string stem = artifact.stem().string();
  if (const std::size_t dash = stem.rfind('-'); dash != std::string::npos && dash > 0) stem.resize(dash);
  return std::format("{} {}", runtime, stem);
}

// Tampon hôte visible, inscrit dans les projections du backend tant qu'il vit.
class HostTensorBuffer final : public TensorBuffer {
 public:
  HostTensorBuffer(VkContext& context, Buffer buffer, TensorType type, const Shape& shape,
                   std::shared_ptr<HostViews> views)
      : context_(context),
        buffer_(std::move(buffer)),
        tensor_{.type = type, .shape = shape, .handle = detail::handle_bits(buffer_.handle()), .offset = 0},
        views_(std::move(views)) {
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = buffer_.mapped(), .bytes = buffer_.size()};
  }
  HostTensorBuffer(const HostTensorBuffer&) = delete;
  HostTensorBuffer& operator=(const HostTensorBuffer&) = delete;
  HostTensorBuffer(HostTensorBuffer&&) = delete;
  HostTensorBuffer& operator=(HostTensorBuffer&&) = delete;
  // Ce que la file a reçu finit avant la libération (VulkanTensorBuffer).
  ~HostTensorBuffer() override {
    {
      const std::scoped_lock lock(views_->mutex);
      views_->views.erase(tensor_.handle);
    }
    context_.settle();
  }
  const Tensor& tensor() const override { return tensor_; }

 private:
  VkContext& context_;
  Buffer buffer_;
  Tensor tensor_;
  std::shared_ptr<HostViews> views_;
};

std::expected<std::vector<net::InputShape>, Refusal> runtime_inputs(const net::IoNames& names,
                                                                    std::vector<net::InputShape> declared,
                                                                    const ShapeSet& shapes) {
  // Les entrées statiques gardent leur forme. Les entrées jumelles de TemporalFix partagent
  // le profil spatial de la première et gardent chacune leur nombre de canaux.
  std::vector<net::InputShape> inputs;
  if (names.inputs.size() == 1)
    inputs.push_back({.name = names.inputs[0], .dims = {shapes.opt.dims.begin(), shapes.opt.dims.end()}});
  else {
    for (auto& input : declared) {
      if (std::ranges::none_of(input.dims, [](auto d) { return d < 0; })) continue;
      if (input.dims.size() != shapes.opt.dims.size() || declared.front().dims.size() != input.dims.size())
        return refuse_backend(RefusalCode::ShapeInvalid,
                              "profil dynamique incompatible avec le rang des entrées");
      for (std::size_t k = 0; k < input.dims.size(); ++k)
        if (input.dims[k] < 0) input.dims[k] = shapes.opt.dims[k];
      inputs.push_back(input);
    }
  }
  return inputs;
}

Done stage_runtime_artifact(net::Runtime& runtime, const ModelSpec& model,
                            std::span<const net::InputShape> inputs, const net::IoNames& names,
                            const Artifact& artifact, Progress& progress, bool compile_binary,
                            const std::optional<Refusal>& invalid_source) {
  std::error_code ec;
  // Construit dans un dossier de travail propre à cet appel, puis publié fichier par fichier
  // (net::publish_artifact) : un arrêt en cours de route, ou le lecteur et un export qui
  // compilent le même réseau, ne laissent jamais un artefact tronqué sous le nom final.
  std::random_device random;
  const std::filesystem::path work =
      artifact.path.parent_path() / std::format(".construction-{}-{}", random(), random());
  std::filesystem::create_directories(work, ec);
  if (ec) return refuse_backend(RefusalCode::CacheIo, std::format("{} : {}", work.string(), ec.message()));
  const std::filesystem::path staged = work / artifact.path.filename();
  auto built = runtime.build(model, inputs, staged, progress);
  if (built && invalid_source) {
    const auto previous = net::read_file(artifact.path);
    const auto fresh = net::read_file(staged);
    if (previous && fresh && *previous == *fresh) built = std::unexpected(*invalid_source);
  }
  if (built && compile_binary) built = runtime.prepare_compiled_artifact(staged, names, progress);
  if (built && compile_binary && progress.stop.stop_requested())
    built = refuse_backend(RefusalCode::Cancelled, "préparation du réseau annulée");
  if (built) built = net::write_io_names(staged, names.inputs, names.outputs);
  if (built) built = net::publish_artifact(staged, artifact.path);
  std::filesystem::remove_all(work, ec);
  return built;
}

Done bind_runtime_tensors(HostViews& views, std::string_view runtime_name, std::span<const Tensor> inputs,
                          std::span<const Tensor> outputs, std::vector<net::HostBinding>& bindings,
                          std::vector<VkBuffer>& external) {
  bindings.reserve(inputs.size() + outputs.size());
  {
    const std::scoped_lock lock(views.mutex);
    const auto bind = [&](const Tensor& t) -> std::expected<void, Refusal> {
      const auto found = views.views.find(t.handle);
      auto bytes = detail::tensor_bytes(t.type, t.shape);
      if (!bytes) return std::unexpected(bytes.error());
      if (found == views.views.end() || (!found->second.data && !found->second.device) ||
          t.offset > found->second.bytes || *bytes > found->second.bytes - t.offset) {
        return refuse_backend(RefusalCode::InputMismatch,
                              std::format("tenseur hors des tampons alloués pour {}", runtime_name));
      }
      const HostViews::View& v = found->second;
      bindings.push_back({.data = v.data ? v.data + t.offset : nullptr,
                          .bytes = *bytes,
                          .type = t.type,
                          .shape = t.shape,
                          .device = v.device ? v.device + t.offset : 0});
      if (v.device) {
        auto buffer = detail::from_bits<VkBuffer>(t.handle);
        if (!std::ranges::contains(external, buffer)) external.push_back(buffer);
      }
      return {};
    };
    for (const Tensor& t : inputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
    for (const Tensor& t : outputs) {
      if (auto ok = bind(t); !ok) return ok;
    }
  }
  return {};
}

struct DeviceBindings {
  RifeFeatureRequest features;
  bool times_only;
  std::optional<RuntimeEngine::RegionJob> job;
  std::vector<net::HostBinding> inputs{};
  net::RuntimeNet* model = nullptr;

  std::expected<std::shared_ptr<void>, Refusal> run(net::RuntimeNet& dense,
                                                    std::span<const net::HostBinding> ins,
                                                    std::span<const net::HostBinding> outs,
                                                    std::uint64_t release, std::uint64_t acquire) const {
    if (dense.rife_features) {
      if (features != RifeFeatureRequest::Disabled)
        return dense.rife_features->run_device(features, ins, outs, release, acquire);
      dense.rife_features->forget();
    }
    if (dense.time_reuse) return dense.time_reuse->run_device(times_only, ins, outs, release, acquire);
    if (model) return model->run_device(inputs, outs, release, acquire);
    return dense.run_device(ins, outs, release, acquire);
  }

  void bind(RuntimeEngine& engine, HostViews& views, std::string_view name, std::span<const Tensor> image,
            std::span<const Tensor> output, std::span<const net::HostBinding> dense,
            std::vector<VkBuffer>& external) {
    if (!job || image.size() != 1 || output.size() != 1 || dense.size() != 2 || !dense[0].device ||
        !dense[1].device)
      return;
    const auto& previous = job->previous.value_or(output.front());
    if (previous.shape != output.front().shape || previous.type != output.front().type) return;
    std::vector<net::HostBinding> old;
    std::vector<VkBuffer> owners;
    if (!bind_runtime_tensors(views, name, std::span(&previous, 1), {}, old, owners) || !old.front().device)
      return;
    for (auto buffer : owners)
      if (!std::ranges::contains(external, buffer)) external.push_back(buffer);
    auto& controls = job->metadata;
    // Un seul petit transfert PJRT. Les deux RGB sont des vues externes sur la même timeline,
    // y compris si « previous » désigne la sortie. La D2D de retour attend le résultat PJRT.
    inputs = {dense.front(),
              old.front(),
              {.data = reinterpret_cast<std::byte*>(controls.data()),
               .bytes = controls.size() * sizeof(controls.front()),
               .type = TensorType::F16,
               .shape = {{1, 1, 1, static_cast<std::int64_t>(controls.size())}}}};
    model = &engine.net().span_regions->network();
  }
};
}  // namespace

std::expected<Artifact, Refusal> VulkanBackend::build_runtime(const ModelSpec& model, const ShapeSet& shapes,
                                                              Progress& progress) {
  if (model.precision != TensorType::F16 && model.precision != TensorType::F32) {
    return refuse_backend(RefusalCode::FormatUnsupported,
                          std::format("{} : précision entière non prise en charge", model.id));
  }
  std::vector<net::InputShape> declared;
  auto names = net::onnx_io_names(model.onnx_path, &declared);
  if (!names) return std::unexpected(names.error());
  auto inputs = runtime_inputs(*names, std::move(declared), shapes);
  if (!inputs) return std::unexpected(inputs.error());
  std::string key_text = std::format("{}|{}|{}|{}|{}|{}", model.id, model.sha256,
                                     model.precision == TensorType::F16 ? "f16" : "f32", runtime_->version(),
                                     runtime_->device_name(), runtime_->name());
  if (const std::string_view recipe = runtime_->recipe(); !recipe.empty())
    key_text += std::format("|{}", recipe);
  for (const auto& input : *inputs) {
    key_text += '|' + input.name;
    for (const std::int64_t d : input.dims) key_text += std::format(",{}", d);
  }
  const std::uint64_t key = net::fnv1a(std::as_bytes(std::span(key_text)));
  Artifact artifact{
      .cache_key = std::format("{:016x}", key),
      .path = cache_dir_ / std::string(runtime_->name()) /
              std::format("{}-{:016x}{}", detail::file_stem(model.id), key, runtime_->extension())};
  std::error_code ec;
  const auto source_ready = [&] {
    return std::filesystem::exists(artifact.path, ec) && net::read_io_names(artifact.path);
  };
  const auto compiled = runtime_->compiled_artifact_ready(artifact.path);
  if (compiled.value_or(true) && source_ready()) return artifact;
  std::optional<prepare::KeyLock> held;
  std::optional<Refusal> invalid_source;
  if (compiled.has_value()) {
    std::filesystem::create_directories(artifact.path.parent_path(), ec);
    if (ec) return refuse_backend(RefusalCode::CacheIo, ec.message());
    auto lock = prepare::KeyLock::acquire(net::artifact_lock_path(artifact.path), progress.stop);
    if (!lock) return std::unexpected(lock.error());
    held.emplace(std::move(*lock));
    if (runtime_->compiled_artifact_ready(artifact.path).value_or(false)) {
      // L'autre processus a fini, ou une publication interrompue a déjà fourni le binaire et
      // sa source. Seuls les noms manquants restent à publier.
      if (!source_ready())
        if (auto ok = net::write_io_names(artifact.path, names->inputs, names->outputs); !ok)
          return std::unexpected(ok.error());
      return artifact;
    }
    if (source_ready()) {
      auto ok = runtime_->prepare_compiled_artifact(artifact.path, *names, progress);
      if (ok) return artifact;
      if (ok.error().code != RefusalCode::VendorCallFailed && ok.error().code != RefusalCode::InputMismatch)
        return std::unexpected(ok.error());
      // Une source ancienne ou endommagée peut être illisible pour PJRT. La régénérer à
      // part, sans effacer le cache et sans soumettre deux fois la même source refusée.
      invalid_source.emplace(std::move(ok.error()));
    }
  }
  if (auto ok = stage_runtime_artifact(*runtime_, model, *inputs, *names, artifact, progress,
                                       compiled.has_value(), invalid_source);
      !ok)
    return std::unexpected(ok.error());
  log(BackendLogLevel::Info, std::format("{} compilé par {}", model.id, runtime_->name()));
  return artifact;
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::load_runtime(const Artifact& artifact) {
  auto names = net::read_io_names(artifact.path);
  if (!names) return std::unexpected(names.error());
  auto net = runtime_->load(artifact.path);
  if (!net) {
    if (net.error().code == RefusalCode::CacheBusy || net.error().code == RefusalCode::Cancelled)
      return std::unexpected(net.error());
    // Un artefact que le runtime ne relit plus (disque abîmé, pilote ou runtime changés sans
    // changer la clé) est effacé : la prochaine préparation le reconstruit au lieu d'échouer
    // ici à chaque lancement.
    net::discard_artifact(artifact.path);
    log(BackendLogLevel::Warn, std::format("{} illisible ({}) : effacé pour être reconstruit",
                                           artifact.path.filename().string(), net.error().detail));
    return std::unexpected(net.error());
  }
  if (!runtime_timeline_) (*net)->span_regions.reset();
  auto label = runtime_label(runtime_->name(), artifact.path);
  if ((*net)->pair12_rgb_only && runtime_timeline_ && kernels_)
    return std::make_unique<DrbaRgbRuntimeEngine>(std::move(*net), std::move(*names), std::move(label), log_,
                                                  *kernels_);
  return std::make_unique<RuntimeEngine>(std::move(*net), std::move(*names), std::move(label), log_);
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> VulkanBackend::alloc_runtime_tensor(
    TensorType type, const Shape& shape) {
  auto dense = detail::tensor_bytes(type, shape);
  if (!dense) return std::unexpected(dense.error());
  if (device_import_) {
    auto shared = device_shared_tensor(*context_, *runtime_, type, shape, *dense, host_views_);
    if (shared) return shared;
    log(BackendLogLevel::Warn,
        std::format("partage GPU indisponible, tampons suivants par l’hôte : {}", shared.error().detail));
    device_import_ = false;
  }
  // GPU intégré : la mémoire hôte du runtime, importée par Vulkan : IOSurface de Core ML, que le réseau lit
  // et écrit sans copie. Au premier refus, la mémoire hôte du backend pour la
  // suite ; un refus pour ce tenseur seul (FormatUnsupported), pour lui seulement.
  if (host_alignment_ != 0) {
    auto imported = host_imported_tensor(*context_, *runtime_, type, shape, (*dense + 3) / 4 * 4,
                                         host_alignment_, host_views_, !host_import_checked_);
    if (imported) {
      host_import_checked_ = true;
      return imported;
    }
    if (imported.error().code == RefusalCode::FormatUnsupported) {
      log(BackendLogLevel::Debug, std::format("un tenseur de {} par la mémoire du backend ({})",
                                              runtime_->name(), imported.error().detail));
    } else {
      host_alignment_ = 0;
      log(BackendLogLevel::Info, std::format("tenseurs de {} par la mémoire du backend ({})",
                                             runtime_->name(), imported.error().detail));
    }
  }
  // En cache côté hôte : le runtime y lit les entrées et y écrit les sorties ; les noyaux de la
  // chaîne y accèdent comme à tout tampon de stockage.
  auto buffer = Buffer::create(*context_, *memory_, (*dense + 3) / 4 * 4, MemoryUse::Readback);
  if (!buffer) return std::unexpected(buffer.error());
  return std::make_unique<HostTensorBuffer>(*context_, std::move(*buffer), type, shape, host_views_);
}

Done VulkanBackend::retire_runtime_loans() {
  // Borne aussi les prêts pour un client qui soumet des lots sans attendre les images.
  constexpr std::size_t max_pending = 32;
  if (runtime_loans_.size() >= max_pending) {
    if (auto ok = context_->wait_on_host(runtime_timeline_->at(runtime_loans_.front().completed)); !ok)
      return ok;
  }
  auto value = runtime_timeline_->value();
  if (!value) return std::unexpected(value.error());
  while (!runtime_loans_.empty() && runtime_loans_.front().completed <= *value) runtime_loans_.pop_front();
  return {};
}

std::expected<void, Refusal> VulkanBackend::enqueue_runtime(Engine& engine, std::span<const Tensor> inputs,
                                                            std::span<Tensor> outputs) {
  auto* runtime_engine = dynamic_cast<RuntimeEngine*>(&engine);
  if (runtime_engine == nullptr)
    return refuse_backend(RefusalCode::InputMismatch, "moteur d'un autre backend");
  DeviceBindings device{.features = runtime_engine->take_features(),
                        .times_only = runtime_engine->take_times_only(),
                        .job = runtime_engine->take_region()};
  if (auto ok = runtime_engine->check_bindings(inputs.size(), outputs.size()); !ok) return ok;
  std::vector<net::HostBinding> bindings;
  std::vector<VkBuffer> external;
  if (auto ok = bind_runtime_tensors(*host_views_, runtime_->name(), inputs, outputs, bindings, external);
      !ok)
    return ok;
  if (!queue_) return refuse_backend(RefusalCode::DeviceUnavailable, "file du backend absente");
  {
    const std::scoped_lock lock(runtime_mutex_);
    if (runtime_failure_) return std::unexpected(*runtime_failure_);
  }
  const std::span<const net::HostBinding> all(bindings);
  const bool shared = std::ranges::all_of(all, [](const auto& b) { return b.device != 0; });
  const auto ins = all.first(inputs.size());
  const auto outs = all.subspan(inputs.size());
  // Par l'hôte, sans bloquer ce fil : le lot part, puis le runtime attend sur son propre fil que
  // les noyaux qui ont écrit les entrées soient finis (leurs écritures visibles de l'hôte par la
  // barrière de fin de chaque enregistrement), calcule, et signale l'étape ; ce fil enregistre
  // déjà la suite, que le GPU attend sur la timeline. Les sorties écrites par le runtime le sont
  // des soumissions suivantes, qui commencent après le signal. Pas sur une file partagée avec
  // l'affichage : la suite y attendrait le signal de l'hôte et retiendrait chaque image affichée
  // pendant tout le calcul (les pilotes Mesa retiennent même la soumission sur l'hôte).
  if (shared && runtime_timeline_) {
    device.bind(*runtime_engine, *host_views_, runtime_->name(), inputs, outputs, all, external);
    if (auto ok = retire_runtime_loans(); !ok) return ok;
    const auto release = runtime_timeline_->at(++runtime_step_);
    const auto acquire = runtime_timeline_->at(++runtime_step_);
    if (auto ok = kernels_->transfer_ownership(external, true, std::span(&release, 1)); !ok) return ok;
    auto loan = device.run(runtime_engine->net(), ins, outs, release.value, acquire.value);
    if (!loan) {
      // Aucun signal CUDA à attendre lors d'un refus. run_device a drainé les calculs déjà soumis.
      if (auto ok = kernels_->transfer_ownership(external, false); !ok) return ok;
      return std::unexpected(loan.error());
    }
    runtime_loans_.push_back({acquire.value, std::move(*loan)});
    runtime_engine->timing()->submitted(device.model != nullptr);
    return kernels_->transfer_ownership(external, false, std::span(&acquire, 1));
  }
  runtime_engine->forget_features();
  if (auto ok = kernels_->transfer_ownership(external, true); !ok) return ok;
  if (auto ok = queue_->flush(); !ok) return std::unexpected(ok.error());
  const BackendQueue::Step step = queue_->host_step();
  if (!context_->shares_queue()) {
    // ready puis done, sur le fil du runtime, l'un après l'autre : la mesure va de l'un à l'autre.
    auto ready_at = std::make_shared<RuntimeTiming::Clock::time_point>();
    auto deferred = runtime_engine->net().run_deferred(
        ins, outs,
        [this, wait = step.wait, ready_at] {
          auto ready = queue_->wait_elsewhere(wait);
          *ready_at = RuntimeTiming::Clock::now();
          return ready;
        },
        [this, &queue = *queue_, signal = step.signal, ready_at, shared,
         timing = runtime_engine->timing()](std::expected<void, Refusal> ran) {
          if (ran) timing->add(RuntimeTiming::Clock::now() - *ready_at, shared);
          finish_host_step(queue, signal, std::move(ran));
        });
    if (deferred) return kernels_->transfer_ownership(external, false);
    // File d'étapes pleine (ThreadUnavailable) : rien n'est lancé, et l'étape se fait ici, après
    // celles en file ; le fil du GPU attend le runtime au lieu de faire échouer l'image.
    const RefusalCode code = deferred.error().code;
    if (code != RefusalCode::FeatureUnavailable && code != RefusalCode::ThreadUnavailable) {
      // Rien n'a été lancé : l'étape se signale ici, sans quoi la file attendrait sans fin.
      if (auto ok = queue_->signal_on_host(step.signal); !ok) return std::unexpected(ok.error());
      if (auto ok = kernels_->transfer_ownership(external, false); !ok) return ok;
      return deferred;
    }
  }
  // Par l'hôte, sur ce fil : l'attente des entrées, le calcul, le signal de l'étape.
  if (auto ok = context_->wait_on_host(queue_->at(step.wait)); !ok) return std::unexpected(ok.error());
  const auto started = RuntimeTiming::Clock::now();
  auto ran = runtime_engine->net().run(ins, outs);
  if (ran) runtime_engine->timing()->add(RuntimeTiming::Clock::now() - started, shared);
  if (auto ok = queue_->signal_on_host(step.signal); !ok) return std::unexpected(ok.error());
  if (auto ok = kernels_->transfer_ownership(external, false); !ok) return ok;
  return ran;
}

void VulkanBackend::finish_host_step(BackendQueue& queue, std::uint64_t signal,
                                     std::expected<void, Refusal> ran) {
  {
    // Un échec isolé (délai de VideoToolbox, exception d'un runtime) coûte l'image de cette étape,
    // dont les sorties restent celles de l'image précédente ; seuls des échecs de suite ferment le
    // backend, comme le chemin synchrone, qui n'échoue que pour l'étape concernée.
    const std::scoped_lock lock(runtime_mutex_);
    if (ran) {
      step_failures_ = 0;
    } else {
      log(BackendLogLevel::Error, std::format("réseau par {} : {}", runtime_->name(), ran.error().detail));
      if (++step_failures_ >= kMaxStepFailures && !runtime_failure_)
        runtime_failure_ = std::move(ran.error());
    }
  }
  if (auto ok = queue.signal_on_host(signal); !ok) {
    log(BackendLogLevel::Error, std::format("signal de l'étape {} : {}", signal, ok.error().detail));
  }
}

}  // namespace kirari::gpu::vk
