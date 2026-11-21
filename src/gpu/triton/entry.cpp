#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/dynamic_library.h"
#include "core/file_sync.h"
#include "gpu/triton/device.h"
#include "gpu/vulkan/backend/deferred_steps.h"
#include "gpu/vulkan/backend/entry_points.h"
#include "prepare/sha256.h"

namespace {
using namespace kirari;
using namespace kirari::gpu;
using namespace kirari::gpu::triton;
namespace net = vk::net;
namespace fs = std::filesystem;
struct TensorPort {
  const char* name;
  Shape shape;
};
enum class ModelKind { Span, Distil, DistilMotion, RifeMotion, RifeLiteMotion };
struct Package {
  std::string_view arch, folder, onnx, manifest;
  ModelKind kind;
  int width, height;
  std::array<TensorPort, 2> input_ports;
  std::array<TensorPort, 2> output_ports;

  constexpr Package(std::string_view arch_, std::string_view folder_, std::string_view onnx_,
                    std::string_view manifest_, ModelKind kind_, int width_, int height_)
      : arch(arch_),
        folder(folder_),
        onnx(onnx_),
        manifest(manifest_),
        kind(kind_),
        width(width_),
        height(height_),
        input_ports{{first_input(), {"pair12", {{1, 7, height, width}}}}},
        output_ports{{first_output(), {"/Sigmoid_output_0", {{1, 1, height, width}}}}} {}

  std::span<const TensorPort> inputs() const { return std::span(input_ports).first(is_distil() ? 2 : 1); }
  std::span<const TensorPort> outputs() const {
    return std::span(output_ports).first(is_rife() || kind == ModelKind::DistilMotion ? 2 : 1);
  }
  constexpr bool is_distil() const { return kind == ModelKind::Distil || kind == ModelKind::DistilMotion; }
  constexpr bool is_rife() const {
    return kind == ModelKind::RifeMotion || kind == ModelKind::RifeLiteMotion;
  }
  std::string header() const { return std::format("{}\t{}\t{}\t{}", plan_format(), width, height, onnx); }

 private:
  constexpr std::string_view plan_format() const {
    switch (kind) {
      case ModelKind::Span:
        return "span-triton-v1";
      case ModelKind::Distil:
        return "drba-triton-v1";
      case ModelKind::DistilMotion:
        return "drba-triton-motion-v1";
      case ModelKind::RifeMotion:
        return "rife-triton-motion-v1";
      case ModelKind::RifeLiteMotion:
        return "rife-lite-triton-motion-v1";
    }
    return {};
  }
  constexpr TensorPort first_input() const {
    if (kind == ModelKind::Span) return {"X", {{1, 3, height, width}}};
    if (is_distil()) return {"pair01", {{1, 7, height, width}}};
    return {"input", {{1, 11, height, width}}};
  }
  constexpr TensorPort first_output() const {
    if (kind == ModelKind::Span) return {"Y", {{1, 3, std::int64_t{2} * height, std::int64_t{2} * width}}};
    if (kind == ModelKind::Distil) return {"out", {{1, 3, height, width}}};
    if (kind == ModelKind::DistilMotion) return {"/Mul_59_output_0", {{1, 4, height, width}}};
    if (kind == ModelKind::RifeLiteMotion) return {"/Add_5_output_0", {{1, 4, height, width}}};
    return {"/Add_7_output_0", {{1, 4, height, width}}};
  }
};
constexpr Package kPackages[] = {
#include "triton_packages.inc"
};

constexpr std::size_t tensor_bytes(const Shape& shape) {
  return static_cast<std::size_t>(shape.dims[1] * shape.dims[2] * shape.dims[3] * 2);
}

auto error(const std::string& msg) {
  return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "Triton : " + msg});
}
std::string hash(const fs::path& path) {
  auto output = prepare::sha256_file(path);
  if (!output) throw std::runtime_error(output.error().detail);
  return *output;
}
unsigned number(const std::string& field) {
  unsigned value = 0;
  auto [end, ec] = std::from_chars(field.data(), field.data() + field.size(), value);
  if (ec != std::errc{} || end != field.data() + field.size()) throw std::runtime_error("nombre invalide");
  return value;
}
bool span_topology(const std::vector<std::string>& launches, const std::map<std::string, Kernel>& kernels) {
  const bool fused = kernels.contains("stencil_gate");
  const bool fused_project = kernels.contains("stencil_gate_project");
  if (kernels.size() != (fused ? 6 : 7) || (fused_project && !fused)) return false;
  const std::size_t ends = fused_project ? 3 : 4;
  const std::size_t per_block = fused ? 3 : 4;
  if (launches.size() < ends || (launches.size() - ends) % per_block != 0) return false;
  const auto blocks = (launches.size() - ends) / per_block;
  if (blocks != 5 && blocks != 8) return false;
  std::vector<std::string_view> expected{"unshuffle", "stem"};
  for (std::size_t b = 0; b < blocks; ++b) {
    expected.insert(expected.end(), {"relu", "relu"});
    if (fused)
      expected.push_back(fused_project && b + 1 == blocks ? "stencil_gate_project" : "stencil_gate");
    else
      expected.insert(expected.end(), {"plain", "gate"});
  }
  if (!fused_project) expected.push_back("project");
  expected.push_back("pixel");
  return std::ranges::equal(launches, expected);
}

bool drba_topology(const std::vector<std::string>& launches, const std::map<std::string, Kernel>& kernels,
                   bool motion) {
  const bool separate_time = kernels.contains("distil_time");
  std::vector<std::string> expected{"images0",  "images1", "images2",  "enc_image", "enc_same",
                                    "enc_same", "enc_up",  "encoded0", "encoded1"};
  if (separate_time) expected.push_back("distil_time");
  for (int block = 0; block < 3; ++block) {
    if (block != 0) expected.push_back(std::format("drba_glue{}", block));
    const auto prefix = std::format("b{}_", block);
    expected.push_back(prefix + "down0");
    expected.push_back(prefix + "down1");
    expected.insert(expected.end(), 8, prefix + "residual");
    expected.push_back(prefix + "up");
  }
  expected.push_back(motion ? "distil_motion" : "distil_rgb");
  return kernels.size() == (separate_time ? 24 : 23) && launches == expected;
}

bool rife_topology(const std::vector<std::string>& launches, const std::map<std::string, Kernel>& kernels,
                   bool lite) {
  const bool packed_d1 = kernels.contains("b4_warp");
  const bool banded_d1 = kernels.contains("b4_warp_band");
  if (lite && !banded_d1) return false;
  std::vector<std::string> expected{"enc_rgb",   "enc_mid", "enc_mid", "enc_head",
                                    "enc_rgb_b", "enc_mid", "enc_mid", "enc_head"};
  const int blocks = lite ? 4 : 5;
  for (int block = 0; block < blocks; ++block) {
    const bool last = block + 1 == blocks;
    const auto prefix = std::format("b{}_", last ? 4 : block);
    expected.push_back(prefix + (last ? "flow" : "entry"));
    if (last && banded_d1) {
      for (int band = 0; band < 8; ++band) expected.insert(expected.end(), {"b4_warp_band", "b4_first_band"});
    } else {
      if (last && packed_d1) expected.push_back("b4_warp");
      expected.push_back(prefix + "first");
    }
    expected.push_back(prefix + "second");
    expected.insert(expected.end(), 8, prefix + "residual");
    expected.push_back(prefix + "head");
  }
  const std::size_t base_kernels = lite ? 24 : 29;
  return kernels.size() == base_kernels + (packed_d1 || banded_d1 ? 1 : 0) && launches == expected;
}

struct Program {
  // Jusqu'à deux entrées et deux sorties. Le plan garde leurs adresses stables.
  using Addresses = std::array<Pointer, 4>;
  std::shared_ptr<Device> gpu;
  fs::path root;
  const Package& package;
  std::map<std::string, Pointer> variables{{"0", 0}};
  std::array<Pointer*, 4> bindings{};
  std::size_t binding_count = 0;
  Addresses host{};
  std::map<std::string, std::size_t> buffer_sizes;
  std::vector<Pointer> owned;
  std::vector<Pointer> modules;
  std::map<std::string, Kernel> kernels;
  std::vector<Launch> launches;
  std::map<std::pair<Addresses, std::size_t>, std::unique_ptr<RecordedLaunch>> recorded;
  bool recording_refused = false;
  std::set<std::string> files;
  Pointer* feature_a = nullptr;
  Pointer* feature_b = nullptr;
  bool features_ready = false;
  std::array<std::size_t, 3> feature_passes{};
  std::size_t distil_time_start = 0;
  std::optional<std::array<Pointer, 2>> distil_inputs;
  std::array<std::size_t, 2> distil_passes{};
  explicit Program(std::shared_ptr<Device> handle, const fs::path& base, const Package& asset)
      : gpu(std::move(handle)), root(base / asset.folder), package(asset) {
    for (const auto& port : package.inputs()) bindings.at(binding_count++) = &variables[port.name];
    for (const auto& port : package.outputs()) bindings.at(binding_count++) = &variables[port.name];
  }
  ~Program() {
    std::lock_guard lock(gpu->sequence);
    std::ignore = gpu->in_context([&] {
      // Même si le pilote a perdu l'appareil, tenter la libération de chaque objet.
      try {
        gpu->synchronize();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "Triton : synchronisation à la libération : %s\n", e.what());
      }
      recorded.clear();
      for (auto mod : modules) gpu->unload(mod);
      for (auto ptr : owned) gpu->release(ptr);
    });
  }
  void verify() {
    if (hash(root / "SHA256SUMS") != package.manifest) throw std::runtime_error("empreinte du paquet");
    std::ifstream checks(root / "SHA256SUMS");
    std::string digest, name;
    while (checks >> digest >> name) {
      if (fs::path(name).filename() != name || name == "." || name == ".." || !files.insert(name).second ||
          hash(root / name) != digest)
        throw std::runtime_error("empreinte de " + name);
    }
    if (!files.contains("plan.tsv")) throw std::runtime_error("plan absent du paquet");
  }
  fs::path asset(const std::string& file) const {
    if (!files.contains(file)) throw std::runtime_error("fichier absent du manifeste : " + file);
    return root / file;
  }
  Pointer allocate(std::size_t size, std::string_view label) {
    const auto ptr = gpu->allocate(size, label);
    owned.push_back(ptr);
    return ptr;
  }
  void buffer(const std::vector<std::string>& f) {
    const auto size = number(f[2]);
    if (!size || variables.contains(f[1])) throw std::runtime_error("tampon invalide");
    const auto ptr = allocate(size, f[1]);
    variables.emplace(f[1], ptr);
    buffer_sizes.emplace(f[1], size);
    if (f[3] == "-") return;
    auto path = asset(f[3]);
    if (fs::file_size(path) != size) throw std::runtime_error("taille de " + f[3]);
    std::vector<char> bytes(size);
    std::ifstream in(path, std::ios::binary);
    if (!in.read(bytes.data(), size)) throw std::runtime_error("lecture de " + f[3]);
    gpu->upload(ptr, bytes.data(), size, f[1]);
  }
  void view(const std::vector<std::string>& f) {
    const auto offset = number(f[3]);
    const auto bytes = number(f[4]);
    const auto base = buffer_sizes.find(f[2]);
    if (base == buffer_sizes.end() || !bytes || offset > base->second || bytes > base->second - offset ||
        !variables.emplace(f[1], variables.at(f[2]) + offset).second)
      throw std::runtime_error("sous-vue de tampon invalide : " + f[1]);
  }
  void kernel(const std::vector<std::string>& f) {
    if (kernels.contains(f[1])) throw std::runtime_error("noyau dupliqué");
    const auto file = asset(f[2]);
    const auto size = fs::file_size(file);
    if (size == 0 || size > std::uintmax_t{32} * 1024 * 1024)
      throw std::runtime_error("taille du binaire GPU");
    std::vector<char> bytes(size);
    std::ifstream in(file, std::ios::binary);
    if (!in.read(bytes.data(), static_cast<std::streamsize>(size)))
      throw std::runtime_error("lecture de " + f[2]);
    Kernel k{.grid = number(f[4]), .block = number(f[5]), .shared = number(f[6])};
    if (f.size() == 9) {
      k.grid_y = number(f[7]);
      k.grid_z = number(f[8]);
    }
    if (!k.grid || !k.grid_y || !k.grid_z || !k.block)
      throw std::runtime_error("géométrie du noyau invalide");
    const auto module = gpu->load(bytes, f[3], k.shared);
    modules.push_back(module.handle);
    k.function = module.function;
    kernels.emplace(f[1], k);
  }
  void load() {
    std::ifstream plan(asset("plan.tsv"));
    std::string line;
    if (!std::getline(plan, line) || line != package.header())
      throw std::runtime_error("entête du programme");
    std::vector<std::string> names;
    while (std::getline(plan, line)) {
      std::vector<std::string> f;
      std::istringstream parser(line);
      for (std::string field; std::getline(parser, field, '\t');) f.push_back(field);
      if (f.size() == 4 && f[0] == "buffer")
        buffer(f);
      else if (f.size() == 5 && f[0] == "view")
        view(f);
      else if ((f.size() == 7 || f.size() == 9) && f[0] == "kernel")
        kernel(f);
      else if (f.size() >= 5 && f[0] == "launch" && f.back() == "0" && f[f.size() - 2] == "0") {
        Launch cmd{kernels.at(f[1]), {}};
        for (std::size_t i = 2; i < f.size(); ++i) cmd.args.push_back(&variables.at(f[i]));
        launches.push_back(std::move(cmd));
        names.push_back(std::move(f[1]));
      } else
        throw std::runtime_error("instruction de plan invalide : " + line);
    }
    const auto valid = package.kind == ModelKind::Span ? span_topology(names, kernels)
                       : package.is_distil()
                           ? drba_topology(names, kernels, package.kind == ModelKind::DistilMotion)
                           : rife_topology(names, kernels, package.kind == ModelKind::RifeLiteMotion);
    if (!valid) throw std::runtime_error("ordre des lancements Triton inattendu");
    if (package.is_distil()) {
      const auto time = std::ranges::find(names, "distil_time");
      if (time != names.end()) distil_time_start = static_cast<std::size_t>(time - names.begin());
    }
    if (package.is_rife()) {
      const auto bytes = tensor_bytes(package.outputs().front().shape);
      if (buffer_sizes.at("feature_a") != bytes || buffer_sizes.at("feature_b") != bytes)
        throw std::runtime_error("taille de l'encodage RIFE");
      feature_a = &variables.at("feature_a");
      feature_b = &variables.at("feature_b");
    }
  }
  void execute_bound(std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys) {
    Addresses addresses{};
    for (std::size_t i = 0; i < xs.size(); ++i) {
      addresses[i] = xs[i].device;
      if (addresses[i]) continue;
      const auto bytes = tensor_bytes(package.inputs()[i].shape);
      if (!host[i]) host[i] = allocate(bytes, "entrée FP16");
      gpu->upload(host[i], xs[i].data, bytes, "entrée FP16");
      addresses[i] = host[i];
    }
    for (std::size_t i = 0; i < ys.size(); ++i) {
      const auto slot = xs.size() + i;
      if (!ys[i].device && !host[slot])
        host[slot] = allocate(tensor_bytes(package.outputs()[i].shape), "sortie FP16");
      addresses[slot] = ys[i].device ? ys[i].device : host[slot];
    }
    try {
      execute(addresses);
    } catch (...) {
      // Le lot a pu être lancé en partie. Les tampons retournent à Vulkan après drainage.
      try {
        gpu->synchronize();
      } catch (...) {
        std::fputs("Triton : synchronisation après un échec de lancement impossible\n", stderr);
      }
      throw;
    }
    gpu->synchronize();
    for (std::size_t i = 0; i < ys.size(); ++i)
      if (!ys[i].device)
        gpu->download(ys[i].data, addresses[xs.size() + i], tensor_bytes(package.outputs()[i].shape));
  }
  void execute(const Addresses& addresses, RifeFeatureRequest request = RifeFeatureRequest::Disabled,
               bool times_only = false) {
    for (std::size_t i = 0; i < binding_count; ++i) *bindings[i] = addresses[i];
    if (package.is_rife()) {
      // Quatre appels par encodage. Les consommateurs pointent vers les deux
      // emplacements, que la paire suivante échange sans copie GPU.
      constexpr std::size_t encoder_calls = 4;
      std::size_t reused = 0;
      if (features_ready && request == RifeFeatureRequest::Following) {
        std::swap(*feature_a, *feature_b);
        reused = 1;
      } else if (features_ready && request == RifeFeatureRequest::SamePair) {
        reused = 2;
      }
      features_ready = false;
      for (auto& cmd : std::span(launches).subspan(encoder_calls * reused)) gpu->launch(cmd.kernel, cmd.args);
      features_ready = request != RifeFeatureRequest::Disabled;
      ++feature_passes[reused];
      return;
    }
    // Le lecteur ne signale « temps seul » que pour les mêmes pixels. Les adresses
    // protègent en plus contre un changement d'allocation entre deux soumissions.
    const std::array pair{addresses[0], addresses[1]};
    const bool reuse = times_only && distil_inputs == pair;
    distil_inputs.reset();
    execute_graph(addresses, reuse ? distil_time_start : 0);
    if (distil_time_start) {
      distil_inputs = pair;
      ++distil_passes[reuse ? 1 : 0];
    }
  }
  void execute_graph(const Addresses& addresses, std::size_t start) {
    const auto selected = std::span(launches).subspan(start);
    if (gpu->can_record()) {
      constexpr std::size_t max_recorded = 16;
      const auto key = std::pair{addresses, start};
      const auto previous = recorded.find(key);
      if (previous != recorded.end()) {
        if (!previous->second && !recording_refused) {
          previous->second = gpu->record(selected);
          recording_refused = previous->second == nullptr;
        }
        if (previous->second) {
          previous->second->launch();
          return;
        }
      } else if (!recording_refused && recorded.size() < max_recorded) {
        // Seuls les couples réellement réutilisés justifient un graphe.
        recorded.try_emplace(key);
      }
    }
    for (auto& cmd : selected) gpu->launch(cmd.kernel, cmd.args);
  }
};

class Network final : public net::RuntimeNet {
  class FeatureReuse final : public net::RuntimeRifeFeatures {
   public:
    explicit FeatureReuse(Network& owner) : owner_(owner) {}
    void forget() override {
      std::lock_guard lock(owner_.p_->gpu->sequence);
      owner_.p_->features_ready = false;
    }
    std::string summary() const override {
      std::lock_guard lock(owner_.p_->gpu->sequence);
      const auto& passes = owner_.p_->feature_passes;
      return std::format("RIFE Triton encodages · deux nouveaux {}, un nouveau {}, deux repris {}", passes[0],
                         passes[1], passes[2]);
    }
    std::expected<std::shared_ptr<void>, Refusal> run_device(RifeFeatureRequest request,
                                                             std::span<const net::HostBinding> xs,
                                                             std::span<const net::HostBinding> ys,
                                                             std::uint64_t wait,
                                                             std::uint64_t done) override {
      return owner_.run_shared(xs, ys, wait, done, request);
    }

   private:
    Network& owner_;
  };

  class DistilReuse final : public net::RuntimeTimeReuse {
   public:
    explicit DistilReuse(Network& owner) : owner_(owner) {}
    void forget() override {
      std::lock_guard lock(owner_.p_->gpu->sequence);
      owner_.p_->distil_inputs.reset();
    }
    std::string summary() const override {
      std::lock_guard lock(owner_.p_->gpu->sequence);
      const auto& passes = owner_.p_->distil_passes;
      return std::format("DistilDRBA Triton encodages · trios nouveaux {}, repris {}", passes[0], passes[1]);
    }
    std::expected<std::shared_ptr<void>, Refusal> run_device(bool same_images,
                                                             std::span<const net::HostBinding> xs,
                                                             std::span<const net::HostBinding> ys,
                                                             std::uint64_t wait,
                                                             std::uint64_t done) override {
      return owner_.run_shared(xs, ys, wait, done, RifeFeatureRequest::Disabled, same_images);
    }

   private:
    Network& owner_;
  };

 public:
  Network(std::shared_ptr<Program> program, net::IoNames names)
      : p_(std::move(program)),
        names_(std::move(names)),
        steps_("triton", "Triton",
               [this](std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys) {
                 return run(xs, ys);
               }) {
    // Les plans Distil ne lisent que les trois plans RGB de la troisième image dans pair12.
    pair12_rgb_only = p_->package.is_distil();
    if (p_->package.is_rife()) rife_features = std::make_unique<FeatureReuse>(*this);
    if (p_->distil_time_start) time_reuse = std::make_unique<DistilReuse>(*this);
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
  vk::Done run(std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys) override {
    if (!matches(xs, ys))
      return refuse(RefusalCode::InputMismatch, "Triton : liaison FP16 de forme inattendue");
    std::lock_guard lock(p_->gpu->sequence);
    return p_->gpu->in_context([&] { p_->execute_bound(xs, ys); });
  }
  vk::Done run_deferred(std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys,
                        std::function<vk::Done()> ready, std::function<void(vk::Done)> done) override {
    if (!matches(xs, ys))
      return refuse(RefusalCode::InputMismatch, "Triton : liaison FP16 de forme inattendue");
    return steps_.push(xs, ys, std::move(ready), std::move(done));
  }
  static bool matches_port(const net::HostBinding& binding, const TensorPort& port) {
    return binding.type == TensorType::F16 && binding.shape == port.shape &&
           binding.bytes >= tensor_bytes(port.shape) && (binding.device != 0 || binding.data != nullptr);
  }
  bool matches(std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys) const {
    return std::ranges::equal(xs, p_->package.inputs(), matches_port) &&
           std::ranges::equal(ys, p_->package.outputs(), matches_port);
  }
  std::expected<std::shared_ptr<void>, Refusal> run_device(std::span<const net::HostBinding> xs,
                                                           std::span<const net::HostBinding> ys,
                                                           std::uint64_t wait, std::uint64_t done) override {
    return run_shared(xs, ys, wait, done, RifeFeatureRequest::Disabled);
  }

 private:
  std::expected<std::shared_ptr<void>, Refusal> run_shared(std::span<const net::HostBinding> xs,
                                                           std::span<const net::HostBinding> ys,
                                                           std::uint64_t wait, std::uint64_t done,
                                                           RifeFeatureRequest request,
                                                           bool times_only = false) {
    auto& gpu = *p_->gpu;
    std::lock_guard lock(gpu.sequence);
    const auto unshared = [](const auto& binding) { return binding.device == 0; };
    if (!matches(xs, ys) || std::ranges::any_of(xs, unshared) || std::ranges::any_of(ys, unshared))
      return error("tenseurs ou timeline incompatibles");
    Program::Addresses addresses{};
    std::ranges::transform(xs, addresses.begin(), &net::HostBinding::device);
    std::ranges::transform(ys, std::span(addresses).subspan(xs.size()).begin(), &net::HostBinding::device);
    if (auto ok = gpu.wait(wait); !ok) {
      p_->features_ready = false;
      p_->distil_inputs.reset();
      return std::unexpected(ok.error());
    }
    if (auto ok = gpu.in_context([&] { p_->execute(addresses, request, times_only); }); !ok) {
      p_->features_ready = false;
      p_->distil_inputs.reset();
      std::ignore = gpu.in_context([&] { gpu.synchronize(); });
      return std::unexpected(ok.error());
    }
    if (auto ok = gpu.signal(done); !ok) {
      p_->features_ready = false;
      p_->distil_inputs.reset();
      std::ignore = gpu.in_context([&] { gpu.synchronize(); });
      return std::unexpected(ok.error());
    }
    return p_;
  }

  std::shared_ptr<Program> p_;
  net::IoNames names_;
  net::DeferredSteps steps_;  // Drainé avant le programme et ses tampons.
};

class Runtime final : public net::Runtime {
 public:
  Runtime(std::shared_ptr<Device> gpu, std::string device, fs::path folder,
          std::vector<const Package*> packages)
      : gpu_(std::move(gpu)),
        device_(std::move(device)),
        folder_(std::move(folder)),
        packages_(std::move(packages)) {
    for (const auto* package : packages_) recipe_.append(package->manifest);
  }
  std::string_view name() const override { return "triton"; }
  std::string device_name() const override { return device_; }
  std::string version() const override { return "3.8.0"; }
  std::string_view recipe() const override { return recipe_; }
  std::string_view extension() const override { return ".triton"; }
  bool shares_device_memory() const override { return gpu_->shares_device_memory(); }
  vk::Done import_timeline(int fd) override { return gpu_->import_timeline(fd); }
  std::expected<net::DeviceMemory, Refusal> import_device(Pointer fd, std::size_t size,
                                                          std::size_t bytes) override {
    return gpu_->import_memory(fd, size, bytes);
  }
  vk::Done build(const ModelSpec& model, std::span<const net::InputShape> inputs, const fs::path& out,
                 Progress& progress) override {
    if (progress.stop.stop_requested()) return std::unexpected(Refusal{RefusalCode::Cancelled, "Triton"});
    const auto package = std::ranges::find(packages_, model.sha256, &Package::onnx);
    if (package == packages_.end())
      return refuse(RefusalCode::FormatUnsupported, "Triton : modèle ONNX absent du paquet précompilé");
    const auto& candidate = **package;
    // Le backend omet les entrées d'un ONNX à deux entrées déjà fixées. Son SHA les lie au paquet.
    const bool fixed_drba = candidate.is_distil() && inputs.empty();
    const auto shape_matches = [&](const net::InputShape& input, const TensorPort& expected) {
      return (candidate.kind == ModelKind::Span || input.name == expected.name) &&
             std::ranges::equal(input.dims, expected.shape.dims);
    };
    if (model.precision != TensorType::F16 ||
        (!fixed_drba && !std::ranges::equal(inputs, candidate.inputs(), shape_matches)))
      return refuse(RefusalCode::FormatUnsupported, "Triton : signature d'entrée différente du paquet");
    auto sha = prepare::sha256_file(model.onnx_path);
    if (!sha || *sha != (*package)->onnx)
      return refuse(RefusalCode::FormatUnsupported, "Triton : empreinte ONNX différente du paquet");
    return core::write_text_atomically(out, (*package)->manifest);
  }
  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(const fs::path& path) override {
    std::array<char, 64> digest{};
    std::ifstream marker(path, std::ios::binary);
    if (!marker.read(digest.data(), static_cast<std::streamsize>(digest.size())) ||
        marker.peek() != std::char_traits<char>::eof())
      return refuse(RefusalCode::FormatUnsupported, "Triton : artefact de cache incompatible avec le paquet");
    const auto package =
        std::ranges::find(packages_, std::string_view(digest.data(), digest.size()), &Package::manifest);
    if (package == packages_.end())
      return refuse(RefusalCode::FormatUnsupported, "Triton : artefact de cache incompatible avec le paquet");
    auto names = net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    const auto& candidate = **package;
    if (names->outputs.size() != candidate.outputs().size() ||
        names->inputs.size() != candidate.inputs().size() ||
        (candidate.kind != ModelKind::Span &&
         (!std::ranges::equal(names->outputs, candidate.outputs(), {}, std::identity{}, &TensorPort::name) ||
          !std::ranges::equal(names->inputs, candidate.inputs(), {}, std::identity{}, &TensorPort::name))))
      return refuse(RefusalCode::InputMismatch, "Triton : signature d'E/S différente du paquet");
    auto p = std::make_shared<Program>(gpu_, folder_, candidate);
    try {
      p->verify();
    } catch (const std::exception& e) {
      return error(e.what());
    }
    std::lock_guard lock(gpu_->sequence);
    if (auto ok = gpu_->in_context([&] { p->load(); }); !ok) return std::unexpected(ok.error());
    return std::make_unique<Network>(std::move(p), std::move(*names));
  }

 private:
  std::shared_ptr<Device> gpu_;
  std::string device_;
  fs::path folder_;
  std::vector<const Package*> packages_;
  std::string recipe_;
};
std::expected<std::shared_ptr<Device>, Refusal> select(const vk::DeviceCaps& caps) {
#ifdef KIRARI_TRITON_CUDA
  if (caps.vendor_id == 0x10de) return make_cuda(caps);
#endif
#ifdef KIRARI_TRITON_HIP
  if (caps.vendor_id == 0x1002) return make_hip(caps);
#endif
#ifdef KIRARI_TRITON_XPU
  if (caps.vendor_id == 0x8086) return make_xpu(caps);
#endif
  return refuse(RefusalCode::DeviceMismatch, "Triton : aucun paquet pour cet appareil");
}
std::expected<std::shared_ptr<net::Runtime>, Refusal> make(const vk::DeviceCaps& caps) {
  auto gpu = select(caps);
  if (!gpu) return std::unexpected(gpu.error());
  const auto folder = library_containing(reinterpret_cast<const void*>(&make)).parent_path() / "triton";
  std::vector<const Package*> packages;
  for (const auto& package : kPackages)
    if (package.arch == (*gpu)->architecture() && fs::is_regular_file(folder / package.folder / "SHA256SUMS"))
      packages.push_back(&package);
  if (packages.empty()) return refuse(RefusalCode::LibraryUnavailable, "Triton : paquet adjacent absent");
  return std::make_shared<Runtime>(std::move(*gpu), caps.device_name, folder, std::move(packages));
}
vk::BackendOptions options() {
  return {.runtime = &make, .partial_runtime = true};
}
}  // namespace
KIRARI_VULKAN_BACKEND_ENTRY_POINTS("Triton", options)
