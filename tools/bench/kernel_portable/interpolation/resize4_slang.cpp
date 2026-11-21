#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/slang/resize4.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "kernels/ref/half.h"

namespace {
using namespace kirari;
using namespace kirari::gpu;
namespace net = vk::net;

template <class T>
T checked(std::expected<T, Refusal> result) {
  if (!result) throw std::runtime_error(result.error().detail);
  if constexpr (!std::is_void_v<T>) return std::move(*result);
}

ir::Graph graph(int height, int width, ir::Layout order) {
  constexpr auto fp16 = ir::DataType::F16;
  return {.tensors = {{.name = "x",
                       .kind = ir::TensorKind::Input,
                       .type = fp16,
                       .dims = {1, 5, height, width},
                       .known = true,
                       .layout = order},
                      {.name = "scales",
                       .kind = ir::TensorKind::Constant,
                       .dims = {4},
                       .known = true,
                       .value = {.dims = {4}, .f = {1, 1, 4, 4}}},
                      {.name = "y",
                       .type = fp16,
                       .dims = {1, 5, height * 4, width * 4},
                       .known = true,
                       .layout = order}},
          .nodes = {{.name = "resize",
                     .op = ir::OpKind::Resize,
                     .attrs = ir::ResizeAttrs{.mode = ir::ResizeMode::Linear},
                     .inputs = {0, ir::kNoTensor, 1},
                     .outputs = {2}}},
          .inputs = {0},
          .outputs = {2},
          .opset = 18};
}

std::vector<std::uint16_t> run(vk::VkContext& ctx, vk::MemoryAllocator& memory, vk::PipelineCache& pipelines,
                               const ir::Graph& g, const vk::Buffer& input,
                               net::SpecializeDispatch specialize) {
  const auto target = vk::profile_device(ctx).target;
  const auto choices = checked(net::default_choices(g, target));
  auto plan = checked(net::lower(g, choices, target));
  auto engine = checked(
      net::NetEngine::create(ctx, memory, pipelines, g, std::move(plan),
                             {.profile_dispatches = true, .specialize = specialize, .target = target}));
  const auto& name = engine->program().dispatches.front().shader;
  if (specialize && name != "slang_resize4_f16")
    throw std::runtime_error("le dispatch Slang n'a pas été sélectionné");
  const auto bytes = engine->output_bytes(0);
  auto output = checked(vk::Buffer::create(ctx, memory, bytes, vk::MemoryUse::Device));
  std::vector<std::uint16_t> result(bytes / sizeof(std::uint16_t));
  // Initialise aussi le bourrage NHWC. Les shaders écrivent uniquement les canaux logiques.
  checked(vk::upload(ctx, memory, output, 0, std::as_bytes(std::span(result))));
  auto* record = checked(engine->recording(std::array{input.view()}, std::array{output.view()}));
  for (int i = 0; i < 55; ++i) {
    checked(ctx.submit(*record));
    checked(ctx.wait_on_host(record->last_run()));
    if (i >= 5) checked(engine->collect_profile(*record));
  }
  checked(vk::download(ctx, memory, output, 0, std::as_writable_bytes(std::span(result))));
  const auto profile = engine->profile_report();
  std::printf("  %s %.2f µs, %zu mesures\n", name.c_str(), profile.total_ms * 1000.0, profile.runs);
  return result;
}

void compare(vk::VkContext& ctx, vk::MemoryAllocator& memory, vk::PipelineCache& pipelines, int height,
             int width, ir::Layout order) {
  const auto g = graph(height, width, order);
  const auto bytes = ir::storage_bytes(g.tensors[0]);
  std::vector<std::uint16_t> pixels(bytes / sizeof(std::uint16_t));
  std::mt19937 random(430);
  std::uniform_real_distribution<float> distribution(-32, 32);
  for (auto& p : pixels) p = kernels::ref::float_to_half(distribution(random));
  auto input = checked(vk::Buffer::create(ctx, memory, bytes, vk::MemoryUse::Device));
  checked(vk::upload(ctx, memory, input, 0, std::as_bytes(std::span(pixels))));
  std::printf("%d×%d, %s\n", height, width, order == ir::Layout::Nchw ? "NCHW" : "NHWC16");
  const auto baseline = run(ctx, memory, pipelines, g, input, nullptr);
  const auto candidate = run(ctx, memory, pipelines, g, input, slang::resize4_dispatch);
  std::size_t differences = 0;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    if (baseline[i] == candidate[i]) continue;
    if (differences < 8)
      std::fprintf(stderr, "  %zu · Vulkan %04x · Slang %04x\n", i, unsigned(baseline[i]),
                   unsigned(candidate[i]));
    ++differences;
  }
  const auto address = [order](int h, int w, int c, int y, int x) {
    return static_cast<std::size_t>(order == ir::Layout::Nchw ? (c * h + y) * w + x : (y * w + x) * 16 + c);
  };
  const auto tap = [](int out, int size) {
    const float pos =
        std::clamp((static_cast<float>(out) + 0.5F) * 0.25F - 0.5F, 0.0F, static_cast<float>(size - 1));
    const int lo = static_cast<int>(pos);
    return std::tuple{lo, std::min(lo + 1, size - 1), pos - static_cast<float>(lo)};
  };
  const auto sample = [&](int c, int y, int x) {
    return kernels::ref::half_to_float(pixels[address(height, width, c, y, x)]);
  };
  std::vector<std::uint16_t> precise_output(candidate.size());
  for (int y = 0; y < height * 4; ++y) {
    const auto [top, bottom, wy] = tap(y, height);
    for (int x = 0; x < width * 4; ++x) {
      const auto [left, right, wx] = tap(x, width);
      for (int c = 0; c < 5; ++c) {
        const float a = (1.F - wx) * sample(c, top, left) + wx * sample(c, top, right);
        const float b = (1.F - wx) * sample(c, bottom, left) + wx * sample(c, bottom, right);
        precise_output[address(height * 4, width * 4, c, y, x)] =
            kernels::ref::float_to_half((1.F - wy) * a + wy * b);
      }
    }
  }
  if (precise_output != candidate) throw std::runtime_error("Slang diffère de la référence CPU sans FMA");
  std::printf("  %zu codes égaux au CPU sans FMA, %zu écarts face à Vulkan\n", candidate.size(), differences);
}
}  // namespace

int main() try {
  auto ctx = checked(vk::VkContext::create({}));
  auto memory = checked(vk::MemoryAllocator::create(*ctx));
  auto pipelines = checked(vk::PipelineCache::create(*ctx));
  std::printf("%s\n", ctx->caps().device_name.c_str());
  for (auto order : {ir::Layout::Nchw, ir::Layout::Nhwc16}) {
    for (const auto& [h, w] : {std::pair{1, 3}, std::pair{7, 11}, std::pair{180, 320}, std::pair{270, 480}})
      compare(*ctx, *memory, *pipelines, h, w, order);
  }
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
