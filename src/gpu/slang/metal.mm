// Exécute les sept shaders Slang 8×8 sur le GPU Apple Silicon.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstring>
#include <mutex>

#include "assets.h"
#include "core/dynamic_library.h"
#include "gpu/vulkan/backend/entry_points.h"

namespace kirari::gpu::slang {
namespace {
namespace net = vk::net;

void check(bool success, NSError* reason, const char* message) {
  if (success) return;
  if (const char* text = reason.localizedDescription.UTF8String)
    throw std::runtime_error(std::string(message) + " : " + text);
  throw std::runtime_error(message);
}

class MetalNetwork final : public net::RuntimeNet {
  c64::balanced::Plan plan_;
  net::IoNames names_;
  id<MTLCommandQueue> queue_;
  std::map<std::string, id<MTLBuffer>> buffers_;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
  std::mutex mutex_;

  void initialize(id<MTLDevice> device) {
    for (const auto& [name, view] : plan_.views) {
      id<MTLBuffer> memory = nullptr;
      if (!view.file.empty()) {
        auto bytes = c64::read(view.file);
        c64::require(bytes.size() == view.bytes, "taille des poids Metal incorrecte");
        memory = [device newBufferWithBytes:bytes.data()
                                     length:bytes.size()
                                    options:MTLResourceStorageModeShared];
      } else {
        auto storage =
            name == "X" || name == "Y" ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        memory = [device newBufferWithLength:view.bytes options:storage];
      }
      c64::require(memory != nullptr, "allocation du tenseur Metal impossible");
      buffers_.emplace(name, memory);
    }
    MTLCompileOptions* options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = false;
#pragma clang diagnostic pop
    for (const auto& [name, kernel] : plan_.kernels) {
      auto bytes = c64::read(kernel.file);
      NSString* source = [[NSString alloc] initWithBytes:bytes.data()
                                                  length:bytes.size()
                                                encoding:NSUTF8StringEncoding];
      c64::require(source != nullptr, "source MSL illisible");
      NSError* reason = nullptr;
      id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&reason];
      check(library != nullptr, reason, "compilation du shader MSL");
      id<MTLFunction> function = [library newFunctionWithName:@"main_0"];
      c64::require(function != nullptr, "point d’entrée MSL main_0 absent");
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                   error:&reason];
      check(pipeline != nullptr, reason, "création du pipeline Metal");
      c64::require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 128,
                   "les shaders Metal demandent quatre groupes SIMD de 32 fils");
      pipelines_.emplace(name, pipeline);
    }
  }

  void dispatch() {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    c64::require(command != nullptr, "commande de calcul Metal indisponible");
    constexpr std::array<std::uint32_t, 2> size{960, 540};
    for (const auto& step : plan_.launches) {
      const auto& kernel = plan_.kernels.at(step.name);
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      c64::require(encoder != nullptr, "encodeur Metal indisponible");
      [encoder setComputePipelineState:pipelines_.at(step.name)];
      [encoder setBytes:size.data() length:sizeof(size) atIndex:kernel.constants];
      for (std::size_t i = 0; i < step.args.size(); ++i)
        [encoder setBuffer:buffers_.at(step.args[i]) offset:0 atIndex:i + (i >= kernel.constants ? 1 : 0)];
      [encoder dispatchThreadgroups:MTLSizeMake(kernel.x, kernel.y, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      // Les buffers Metal suivis propagent leurs dépendances entre encodeurs.
      [encoder endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted, command.error, "inférence Metal");
  }

  bool matches(std::span<const net::HostBinding> binding, const Shape& shape, const char* name) const {
    return binding.size() == 1 && binding[0].data && binding[0].type == TensorType::F16 &&
           binding[0].shape == shape && binding[0].bytes >= plan_.views.at(name).bytes;
  }

 public:
  MetalNetwork(id<MTLDevice> device, c64::balanced::Plan plan, net::IoNames names)
      : plan_(std::move(plan)), names_(std::move(names)), queue_([device newCommandQueue]) {
    c64::require(queue_ != nullptr, "file de calcul Metal indisponible");
    initialize(device);
  }
  std::span<const std::string> input_names() const override { return names_.inputs; }
  std::span<const std::string> output_names() const override { return names_.outputs; }
  vk::Done run(std::span<const net::HostBinding> input, std::span<const net::HostBinding> output) override
      try {
    if (!matches(input, kX, "X") || !matches(output, kY, "Y"))
      return error("les deux tenseurs hôte FP16 NCHW du film sont requis");
    @autoreleasepool {
      const std::scoped_lock lock(mutex_);
      id<MTLBuffer> x = buffers_.at("X");
      id<MTLBuffer> y = buffers_.at("Y");
      std::memcpy(x.contents, input.front().data, x.length);
      dispatch();
      std::memcpy(output.front().data, y.contents, y.length);
    }
    return {};
  } catch (const std::exception& e) {
    return error(e.what());
  }
};

class Runtime final : public net::Runtime {
  id<MTLDevice> device_;
  std::string name_;
  fs::path folder_;

 public:
  Runtime(id<MTLDevice> device, fs::path folder)
      : device_(device),
        name_(device.name.UTF8String ? device.name.UTF8String : "GPU Apple"),
        folder_(std::move(folder)) {}
  std::string_view name() const override { return "slang"; }
  std::string device_name() const override { return name_; }
  std::string version() const override { return "metal-v1"; }
  std::string_view recipe() const override { return kPackages[0].manifest; }
  std::string_view extension() const override { return ".slang"; }
  vk::Done build(const ModelSpec& model, std::span<const net::InputShape> inputs, const fs::path& out,
                 Progress& progress) override {
    return prepare_model(model, inputs, out, progress);
  }
  std::expected<std::unique_ptr<net::RuntimeNet>, Refusal> load(const fs::path& artifact) override try {
    auto selected = select_artifact(artifact);
    if (!selected) return std::unexpected(selected.error());
    @autoreleasepool {
      return std::make_unique<MetalNetwork>(device_, verified(folder_, *selected->package, true),
                                            std::move(selected->names));
    }
  } catch (const std::exception& e) {
    return error(e.what());
  }
};

std::expected<std::shared_ptr<net::Runtime>, Refusal> make(const vk::DeviceCaps&) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyApple7])
      return refuse(RefusalCode::FeatureUnavailable,
                    "Slang Metal demande un GPU Apple Silicon avec matrices SIMD");
    auto dir = library_containing(reinterpret_cast<const void*>(&make)).parent_path();
    return std::make_shared<Runtime>(device, std::move(dir));
  }
}

vk::BackendOptions options() {
  return {.runtime = &make, .partial_runtime = true};
}
}  // namespace
}  // namespace kirari::gpu::slang
KIRARI_VULKAN_BACKEND_ENTRY_POINTS("Slang", kirari::gpu::slang::options)
