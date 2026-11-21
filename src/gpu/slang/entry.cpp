// Point d’entrée Slang. X et Y référencent directement des allocations Vulkan du lecteur.
#undef VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "assets.h"
#include "interop.h"

// API du lecteur.
#include "core/dynamic_library.h"
#include "core/file_sync.h"
#include "gpu/vulkan/backend/entry_points.h"

#if defined(KIRARI_SLANG_RESIZE4)
#include "gpu/slang/resize4.h"
#endif

namespace {
using namespace kirari;
using namespace kirari::gpu;
namespace net = vk::net;
namespace fs = std::filesystem;
#ifdef KIRARI_SLANG_FD_TRANSPORT
using slang_reader::Imported;
using slang_reader::Recording;
#endif
using slang_reader::Session;
using namespace kirari::gpu::slang;

struct Program {
  std::shared_ptr<Session> session;
  c64::balanced::Plan plan;
  c64::balanced::Network network;
#ifdef KIRARI_SLANG_FD_TRANSPORT
  std::deque<std::shared_ptr<Recording>> cache;
#endif
  Program(std::shared_ptr<Session> s, const fs::path& root, const Package& package)
      : session(std::move(s)), plan(verified(root, package)), network(session->gpu, plan, 1) {
    std::scoped_lock lock(session->queue);
#ifdef KIRARI_SLANG_FD_TRANSPORT
    network.initialize();
    constexpr auto mode = "X/Y Vulkan sans copie";
#else
    network.initialize(c64::balanced::Endpoints::Host);
    constexpr auto mode = "transfert des tenseurs d’entrée et de sortie par l’hôte";
#endif
    std::fprintf(stderr, "Slang SPAN : %zu lancements, %zu SPIR-V, paquet %.*s, %s\n", plan.launches.size(),
                 plan.kernels.size(), static_cast<int>(package.manifest.size()), package.manifest.data(),
                 mode);
  }
#ifdef KIRARI_SLANG_FD_TRANSPORT
  std::shared_ptr<Recording> commands(Imported& x, Imported& y) {
    for (const auto& r : cache)
      if (r->tensors[0].get() == &x && r->tensors[1].get() == &y) return r;
    auto r = std::make_shared<Recording>(session, x, y);
    r->record(network);
    cache.push_back(r);
    if (cache.size() > 4) cache.pop_front();
    return r;
  }
#endif
};

class Network final : public net::RuntimeNet {
  std::shared_ptr<Program> program_;
  net::IoNames names_;
#ifdef KIRARI_SLANG_FD_TRANSPORT
  struct Loan {
    std::shared_ptr<Program> program;
    std::shared_ptr<Recording> record;
  };
#endif
  static bool matches(std::span<const net::HostBinding> binding, const Shape& shape) {
    return binding.size() == 1 && binding[0].type == TensorType::F16 && binding[0].shape == shape &&
           binding[0].bytes >= std::uint64_t(shape.dims[2] * shape.dims[3] * 6);
  }

 public:
  Network(std::shared_ptr<Program> p, net::IoNames n) : program_(std::move(p)), names_(std::move(n)) {}
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
#ifdef KIRARI_SLANG_FD_TRANSPORT
  vk::Done run(std::span<const net::HostBinding>, std::span<const net::HostBinding>) override {
    return error("interop Vulkan nécessaire");
  }
  std::expected<std::shared_ptr<void>, Refusal> run_device(std::span<const net::HostBinding> xs,
                                                           std::span<const net::HostBinding> ys,
                                                           std::uint64_t wait, std::uint64_t signal) override
      try {
    if (!matches(xs, kX) || !matches(ys, kY) || !xs[0].device || !ys[0].device)
      return error("les deux tenseurs FP16 NCHW partagés du film sont requis");
    // Le handle de la liaison est le pointeur opaque renvoyé par import_device.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    auto& x = *reinterpret_cast<Imported*>(xs[0].device);
    auto& y = *reinterpret_cast<Imported*>(ys[0].device);
    // NOLINTEND(performance-no-int-to-ptr)
    if (x.session != program_->session || y.session != program_->session || x.bytes < xs[0].bytes ||
        y.bytes < ys[0].bytes)
      return error("vue Vulkan incompatible");
    std::scoped_lock lock(program_->session->queue);
    auto r = program_->commands(x, y);
    auto loan = std::make_shared<Loan>(program_, r);
    r->submit(wait, signal);
    return loan;
  } catch (const std::exception& e) {
    return error(e.what());
  }
#else
  vk::Done run(std::span<const net::HostBinding> xs, std::span<const net::HostBinding> ys) override try {
    if (!matches(xs, kX) || !matches(ys, kY) || !xs[0].data || !ys[0].data)
      return error("les deux tenseurs hôte FP16 NCHW du film sont requis");
    const auto& views = program_->plan.views;
    std::scoped_lock lock(program_->session->queue);
    program_->network.execute_host({xs[0].data, static_cast<std::size_t>(views.at("X").bytes)},
                                   {ys[0].data, static_cast<std::size_t>(views.at("Y").bytes)});
    return {};
  } catch (const std::exception& e) {
    return error(e.what());
  }
#endif
};

class Runtime final : public net::Runtime {
  std::shared_ptr<Session> session_;
  std::string device_;
  fs::path folder_;
  std::string recipe_ = recipe_signature();

 public:
  Runtime(std::shared_ptr<Session> s, std::string device, fs::path folder)
      : session_(std::move(s)), device_(std::move(device)), folder_(std::move(folder)) {}
  std::string_view name() const override { return "slang"; }
  std::string device_name() const override { return device_; }
  std::string version() const override { return "spirv-v1"; }
  std::string_view recipe() const override { return recipe_; }
  std::string_view extension() const override { return ".slang"; }
#ifdef KIRARI_SLANG_FD_TRANSPORT
  bool shares_device_memory() const override { return true; }
  vk::Done import_timeline(int fd) override try {
    session_->import_timeline(fd);
    return {};
  } catch (const std::exception& e) {
    return error(e.what());
  }
  std::expected<net::DeviceMemory, Refusal> import_device(std::uintptr_t fd, std::size_t allocation,
                                                          std::size_t bytes) override try {
    auto memory = std::make_shared<Imported>(session_);
    memory->load(static_cast<int>(fd), allocation, bytes);
    return net::DeviceMemory{reinterpret_cast<std::uintptr_t>(memory.get()), memory};
  } catch (const std::exception& e) {
    return error(e.what());
  }
#endif
  vk::Done build(const ModelSpec& model, std::span<const net::InputShape> inputs, const fs::path& out,
                 Progress& progress) override {
    return prepare_model(model, inputs, out, progress);
  }
  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(const fs::path& path) override try {
    auto selected = select_artifact(path);
    if (!selected) return std::unexpected(selected.error());
    return std::make_unique<Network>(std::make_shared<Program>(session_, folder_, *selected->package),
                                     std::move(selected->names));
  } catch (const std::exception& e) {
    return error(e.what());
  }
};

std::expected<std::shared_ptr<net::Runtime>, Refusal> make(const vk::DeviceCaps& caps) try {
  auto s = std::make_shared<Session>(caps.uuid);
  auto dir = library_containing(reinterpret_cast<const void*>(&make)).parent_path();
  return std::make_shared<Runtime>(std::move(s), caps.device_name, std::move(dir));
} catch (const std::exception& e) {
  return error(e.what());
}
vk::BackendOptions options() {
  vk::BackendOptions options{.runtime = &make, .partial_runtime = true};
#if defined(KIRARI_SLANG_RESIZE4)
  options.specialize_dispatch = &resize4_dispatch;
#endif
  return options;
}
}  // namespace
KIRARI_VULKAN_BACKEND_ENTRY_POINTS("Slang", options)
