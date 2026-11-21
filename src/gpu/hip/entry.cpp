#include <algorithm>
#include <array>
#include <format>
#include <stdexcept>

#include "core/dynamic_library.h"
#include "gpu/hip/geometry.h"
#include "gpu/hip/session.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/vulkan/backend/deferred_steps.h"
#include "gpu/vulkan/backend/entry_points.h"
#include "gpu/vulkan/backend/net_build.h"

namespace kirari::gpu::hip_backend {
std::unexpected<Refusal> refuse_hip(RefusalCode code, std::string_view detail) {
  return std::unexpected(Refusal{code, std::format("HIP SPAN : {}", detail)});
}

std::expected<vk::net::DeviceMemory, Refusal> Session::import_memory(std::uintptr_t, std::size_t,
                                                                     std::size_t) {
  return refuse_hip(RefusalCode::FeatureUnavailable, "partage de mémoire GPU indisponible");
}

namespace {
namespace net = vk::net;
namespace package = span_fp8;
namespace fs = std::filesystem;

bool input_shape(std::span<const std::int64_t> dimensions) {
  constexpr auto max_side = package::kMaxSide;
  return dimensions.size() == 4 && dimensions[0] == 1 && dimensions[1] == 3 && dimensions[2] > 0 &&
         dimensions[2] <= max_side && dimensions[3] > 0 && dimensions[3] <= max_side &&
         package::size_supported(static_cast<int>(dimensions[2]), static_cast<int>(dimensions[3]));
}

bool compatible(const package::Package& weights) {
  return weights.real_channels == kChannels && weights.blocks.size() == package::kBlocks;
}

struct Filters {
  void* weights{};
  void* bias{};
};

class Storage {
 public:
  explicit Storage(std::shared_ptr<Session> session) : session_(std::move(session)) {}
  ~Storage() {
    const std::scoped_lock lock(session_->sequence);
    try {
      static_cast<void>(session_->in_context([&] {
        for (void* p : owned_)
          if (p) session_->release(p);
        if (scratch_.pointer) session_->release(scratch_.pointer);
        for (auto& staged : staging_)
          if (staged.pointer) session_->release(staged.pointer);
      }));
    } catch (...) {
      // Le pilote peut déjà être indisponible au déchargement.
    }
  }
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Filters upload(const package::Conv& layer) {
    return {.weights = upload_data(std::as_bytes(std::span(layer.weights))),
            .bias = upload_data(std::as_bytes(std::span(layer.bias)))};
  }

  // Résidu distinct de c2 et de la porte. On le réutilise avant le stem pour l'unshuffle.
  std::array<void*, 6> views(int pixels) {
    constexpr std::size_t view_count = 6;
    const auto size = static_cast<std::size_t>(pixels) * kChannels * sizeof(std::uint16_t);
    auto* base = static_cast<std::byte*>(reuse(scratch_, size * view_count, "arène SPAN"));
    return {base, base + size, base + size * 2, base + size * 3, base + size * 4, base + size * 5};
  }

  std::array<void*, 2> stage(const net::HostBinding& in, const net::HostBinding& out) {
    const auto in_bytes = static_cast<std::size_t>(dense_tensor_bytes(in.type, in.shape));
    const auto out_bytes = static_cast<std::size_t>(dense_tensor_bytes(out.type, out.shape));
    auto* input = reuse(staging_[0], in_bytes, "tampon d'entrée HIP");
    auto* output = reuse(staging_[1], out_bytes, "tampon de sortie HIP");
    session_->upload(input, in.data, in_bytes, "copie de l'entrée vers HIP");
    return {input, output};
  }

 private:
  struct Buffer {
    void* pointer{};
    std::size_t bytes{};
  };

  void* reuse(Buffer& buffer, std::size_t bytes, std::string_view name) {
    if (bytes <= buffer.bytes) return buffer.pointer;
    auto* fresh = session_->allocate(bytes, name);
    if (buffer.pointer) session_->release(buffer.pointer);
    buffer = {fresh, bytes};
    return fresh;
  }

  void* upload_data(std::span<const std::byte> bytes) {
    // Enregistrer la case avant l'allocation pour que son nettoyage survive à tout échec.
    auto& dest = owned_.emplace_back(nullptr);
    dest = session_->allocate(bytes.size(), "allocation de poids");
    session_->upload(dest, bytes.data(), bytes.size(), "transfert de poids");
    return dest;
  }

  std::shared_ptr<Session> session_;
  std::vector<void*> owned_;
  Buffer scratch_;
  std::array<Buffer, 2> staging_{};
};

class Network final : public net::RuntimeNet {
 public:
  Network(std::shared_ptr<Session> session, net::IoNames names, const package::Package& weights)
      : session_(std::move(session)),
        storage_(session_),
        names_(std::move(names)),
        steps_("hip-span", "HIP",
               [this](std::span<const net::HostBinding> in, std::span<const net::HostBinding> out) {
                 return run(in, out);
               }) {
    const std::scoped_lock lock(session_->sequence);
    auto loaded = session_->in_context([&] {
      stem_ = storage_.upload(weights.first);
      for (std::size_t i = 0; i < weights.blocks.size(); ++i) {
        const auto& block = weights.blocks[i];
        blocks_[i] = {storage_.upload(block.conv[0]), storage_.upload(block.conv[1]),
                      storage_.upload(block.conv[2]), storage_.upload(block.attention)};
      }
      projection_ = storage_.upload(weights.concat);
      pixel_ = storage_.upload(weights.pixel);
    });
    if (!loaded) throw std::runtime_error(loaded.error().detail);
  }

  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }

  vk::Done run(std::span<const net::HostBinding> in, std::span<const net::HostBinding> out) override {
    if (!bindings(in, out))
      return refuse_hip(RefusalCode::InputMismatch, "tenseurs FP16 NCHW du SPAN incompatibles");
    const std::scoped_lock lock(session_->sequence);
    try {
      return session_->in_context([&] {
        try {
          calculate(in.front(), out.front());
        } catch (...) {
          // Rendre les tampons seulement une fois le flux drainé.
          try {
            session_->synchronize();
          } catch (...) {
          }
          throw;
        }
      });
    } catch (const std::exception& e) {
      return refuse_hip(RefusalCode::VendorCallFailed, e.what());
    }
  }

  vk::Done run_deferred(std::span<const net::HostBinding> in, std::span<const net::HostBinding> out,
                        std::function<vk::Done()> ready, std::function<void(vk::Done)> done) override {
    if (!bindings(in, out))
      return refuse_hip(RefusalCode::InputMismatch, "tenseurs FP16 NCHW du SPAN incompatibles");
    return steps_.push(in, out, std::move(ready), std::move(done));
  }

 private:
  void calculate(const net::HostBinding& in, const net::HostBinding& out) {
    void* src = reinterpret_cast<void*>(in.device);
    void* dst = reinterpret_cast<void*>(out.device);
    const bool staged = !session_->shares_memory();
    if (staged) {
      const auto pointers = storage_.stage(in, out);
      src = pointers[0];
      dst = pointers[1];
    }
    execute(src, dst, static_cast<int>(in.shape.dims[3] / 2), static_cast<int>(in.shape.dims[2] / 2));
    session_->synchronize();
    if (staged)
      session_->download(out.data, dst, static_cast<std::size_t>(dense_tensor_bytes(out.type, out.shape)));
  }

  bool usable(const net::HostBinding& b) const {
    return (session_->shares_memory() ? b.device != 0 : b.data != nullptr) && b.type == TensorType::F16 &&
           b.bytes >= dense_tensor_bytes(b.type, b.shape);
  }

  bool bindings(std::span<const net::HostBinding> in, std::span<const net::HostBinding> out) const {
    if (in.size() != 1 || out.size() != 1 || !input_shape(in.front().shape.dims)) return false;
    const auto& dims = in.front().shape.dims;
    return out.front().shape == Shape{{1, 3, dims[2] * 2, dims[3] * 2}} && usable(in.front()) &&
           usable(out.front());
  }

  template <class... A>
  void launch(Session::Kernel which, unsigned groups_x, unsigned groups_y, A... args) {
    std::array<void*, sizeof...(A)> argv{&args...};
    session_->launch(which, groups_x, groups_y, argv);
  }

  void execute(void* input, void* output, int width, int height) {
    const int pixels = width * height;
    const auto groups = [](int count, int size) { return static_cast<unsigned>((count + size - 1) / size); };
    const auto x = groups(width, kPixels), y = groups(height, kRows);
    const auto points = groups(pixels, kPointPixels);
    auto [stem, first, state, a, b, res] = storage_.views(pixels);
    const auto conv = [&](Session::Kernel mode, void* src, Filters f, void* dst) {
      launch(mode, x, y, src, f.weights, f.bias, dst, width, height);
    };
    launch(Session::Unshuffle, groups(pixels * 16, kWorkers), 1, input, res, width, height);
    conv(Session::Stem, res, stem_, stem);
    void* skip = stem;
    for (const auto& block : blocks_) {
      conv(Session::Relu, skip, block[0], a);
      conv(Session::Relu, a, block[1], b);
      conv(Session::Plain, b, block[2], res);
      void* next = skip == stem ? first : state;
      // Chaque pixel lit uniquement le skip du même pixel, même pour state en place.
      launch(Session::Gate, points, 1, skip, res, block[3].weights, block[3].bias, next, pixels);
      skip = next;
    }
    launch(Session::Project, points, 1, stem, state, first, a, projection_.weights, projection_.bias, b,
           pixels);
    conv(Session::Pixel, b, pixel_, output);
  }

  std::shared_ptr<Session> session_;
  Storage storage_;
  net::IoNames names_;
  std::array<std::array<Filters, 4>, package::kBlocks> blocks_{};
  Filters stem_, projection_, pixel_;
  // Détruit avant les poids et l'arène. Le worker peut encore les lire au déchargement.
  net::DeferredSteps steps_;
};

class Runtime final : public net::Runtime {
 public:
  Runtime(std::shared_ptr<Session> session, std::string name)
      : session_(std::move(session)), device_(std::move(name)) {}
  std::string_view name() const override { return "hip"; }
  std::string device_name() const override { return device_; }
  std::string version() const override { return "span-fp16-v1"; }
  std::string_view recipe() const override { return session_->architecture(); }
  std::string_view extension() const override { return ".hipspan"; }
  bool shares_device_memory() const override { return session_->shares_memory(); }
  std::expected<net::DeviceMemory, Refusal> import_device(std::uintptr_t fd, std::size_t allocation,
                                                          std::size_t bytes) override {
    return session_->import_memory(fd, allocation, bytes);
  }

  vk::Done build(const ModelSpec& model, std::span<const net::InputShape> inputs, const fs::path& dest,
                 Progress& progress) override {
    if (progress.stop.stop_requested()) return refuse_hip(RefusalCode::Cancelled, "préparation annulée");
    if (model.precision != TensorType::F16 || inputs.size() != 1 || !input_shape(inputs[0].dims))
      return refuse_hip(RefusalCode::FormatUnsupported,
                        "SPAN Balanced FP16 NCHW avec dimensions paires requis");
    auto weights = package::package_from_onnx_file(model.onnx_path);
    if (!weights) return std::unexpected(weights.error());
    if (!compatible(*weights)) return refuse_hip(RefusalCode::FormatUnsupported, "SPAN b8f64 requis");
    if (progress.stop.stop_requested()) return refuse_hip(RefusalCode::Cancelled, "préparation annulée");
    weights->fp16 = true;
    return net::write_file(dest, package::serialize(*weights));
  }

  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(const fs::path& path) override try {
    auto names = net::read_io_names(path);
    if (!names) return std::unexpected(names.error());
    auto bytes = net::read_file(path);
    if (!bytes) return std::unexpected(bytes.error());
    auto weights = package::parse(*bytes);
    if (!weights) return std::unexpected(weights.error());
    if (!compatible(*weights) || !weights->fp16 || weights->acc32 || names->inputs.size() != 1 ||
        names->outputs.size() != 1)
      return refuse_hip(RefusalCode::FormatUnsupported, "paquet HIP SPAN b8f64 FP16 attendu");
    return std::make_unique<Network>(session_, std::move(*names), *weights);
  } catch (const std::exception& e) {
    return refuse_hip(RefusalCode::VendorCallFailed, e.what());
  }

 private:
  std::shared_ptr<Session> session_;
  std::string device_;
};

std::expected<std::shared_ptr<net::Runtime>, Refusal> make_runtime(const vk::DeviceCaps& caps) try {
  const auto folder = library_containing(reinterpret_cast<const void*>(&make_runtime)).parent_path() / "hip";
  std::expected<std::shared_ptr<Session>, Refusal> session =
      refuse_hip(RefusalCode::DeviceMismatch, "une carte AMD ou NVIDIA avec son binaire HIP est requise");
#ifdef KIRARI_HIP_ROCM
  if (caps.vendor_id == 0x1002) session = make_rocm(caps, folder);
#endif
#ifdef KIRARI_HIP_CUDA
  if (caps.vendor_id == 0x10de) session = make_cuda(caps, folder);
#endif
  if (!session) return std::unexpected(session.error());
  return std::make_shared<Runtime>(std::move(*session), caps.device_name);
} catch (const std::exception& e) {
  return refuse_hip(RefusalCode::VendorCallFailed, e.what());
}

vk::BackendOptions options() {
  return {.runtime = &make_runtime, .partial_runtime = true};
}
}  // namespace
}  // namespace kirari::gpu::hip_backend

KIRARI_VULKAN_BACKEND_ENTRY_POINTS("HIP", kirari::gpu::hip_backend::options)
