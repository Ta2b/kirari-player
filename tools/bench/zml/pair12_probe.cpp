// Deux géométries pour l'écriture des seuls plans RGB lus par DistilDRBA. Les deux
// enregistrements ont les mêmes tampons, requêtes Vulkan et dépendances. Aucun réseau n'est lancé.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu/vulkan/backend/net_query.h"
#include "gpu/vulkan/compute/kernel_dispatch.h"
#include "gpu/vulkan/compute/kernel_runner.h"

namespace vk = kirari::gpu::vk;
using kirari::Refusal;

namespace {
constexpr unsigned width = 1280, height = 720, padded = 768, top = 24;
constexpr std::size_t plane_bytes = width * padded * 2U;
constexpr unsigned samples = 96, buffers = 12;

template <class T>
T require(std::expected<T, Refusal> result) {
  if (!result) throw std::runtime_error(result.error().detail);
  return std::move(result).value();
}

template <class T>
std::vector<T> load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamoff>(in.tellg());
  if (bytes <= 0 || static_cast<std::size_t>(bytes) % sizeof(T))
    throw std::runtime_error("fichier vide ou mal formé");
  std::vector<T> result(static_cast<std::size_t>(bytes) / sizeof(T));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(result.data()), bytes);
  if (!in) throw std::runtime_error("lecture incomplète");
  return result;
}

struct Fixture {
  vk::Buffer src, dst;
  vk::TensorView image() const { return {src.view(), width, height, 3, vk::Precision::F16}; }
  vk::TensorView pair12() const { return {dst.view(), width, padded, 7, vk::Precision::F16}; }
};

struct Probe {
  std::unique_ptr<vk::VkContext> ctx = require(vk::VkContext::create({}));
  std::unique_ptr<vk::MemoryAllocator> memory = require(vk::MemoryAllocator::create(*ctx));
  std::unique_ptr<vk::PipelineCache> pipes = require(vk::PipelineCache::create(*ctx));
  vk::KernelRunner kernels{*ctx, *pipes};
  vk::net::TimestampSupport support = vk::net::timestamp_support(*ctx);
  vk::net::QueryPool pool{*ctx, vk::net::query_functions(*ctx), samples + 1};
  const vk::Pipeline* xy = nullptr;
  std::vector<Fixture> frames;
  std::vector<std::byte> expected;
  explicit Probe(std::span<const std::uint32_t> spirv, const std::vector<std::byte>& input)
      : expected(7 * plane_bytes, std::byte{0x71}) {
    if (!support.valid_bits || !ctx->caps().enabled.storage_buffer_16bit)
      throw std::runtime_error("estampilles ou FP16 indisponibles");
    if (input.size() != expected.size())
      throw std::runtime_error("pair12 de la capture doit avoir sept plans");
    require(pool.create());
    xy = require(pipes->get({.shader = "pair12_xy_f16",
                             .buffers = 3,
                             .push_bytes = sizeof(vk::detail::RifeFramesPush),
                             .spirv = spirv}));
    std::vector<std::byte> image;
    for (unsigned c = 3; c < 6; ++c) {
      auto channel = std::span(input).subspan(c * plane_bytes, plane_bytes);
      auto visible = channel.subspan(top * width * 2U, height * width * 2U);
      image.insert(image.end(), visible.begin(), visible.end());
      auto plane = std::span(expected).subspan(c * plane_bytes, plane_bytes);
      std::ranges::fill(plane, std::byte{0});
      std::ranges::copy(visible, plane.subspan(top * width * 2U).begin());
    }
    const std::vector sentinel(expected.size(), std::byte{0x71});
    for (unsigned i = 0; i < buffers; ++i) {
      auto src = require(vk::Buffer::create(*ctx, *memory, image.size(), vk::MemoryUse::Device));
      auto dst = require(vk::Buffer::create(*ctx, *memory, expected.size(), vk::MemoryUse::Device));
      require(vk::upload(*ctx, *memory, src, 0, image));
      require(vk::upload(*ctx, *memory, dst, 0, sentinel));
      frames.push_back({std::move(src), std::move(dst)});
    }
  }

  void dispatch(vk::Recorder& rec, const Fixture& frame, bool xy_geometry) {
    if (!xy_geometry) {
      require(kernels.drba_third_image(rec, frame.image(), frame.pair12(), 0.0F));
      return;
    }
    const vk::detail::RifeFramesPush push{width, height, width, padded, 0, top, 0, 0.0F};
    const std::array views{frame.src.view(), frame.src.view(), frame.dst.view()};
    require(rec.dispatch(*xy, views, {}, vk::detail::bytes_of(push), {width / 16, padded / 16, 1}));
  }

  void stamp(vk::Recorder& rec, unsigned i) {
    require(rec.commands([&](VkCommandBuffer cb) {
      if (!i) pool.fn().reset(cb, pool.handle(), 0, pool.count());
      pool.fn().write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool.handle(), i);
    }));
  }

  std::unique_ptr<vk::Recording> record(bool xy_geometry, unsigned rotation) {
    auto rec =
        require(vk::Recorder::begin(*ctx, *memory, {.sets = samples, .buffers = 3 * samples, .images = 1}));
    stamp(rec, 0);
    for (unsigned i = 0; i < samples; ++i) {
      dispatch(rec, frames[i % rotation], xy_geometry);
      stamp(rec, i + 1);
    }
    auto recorded = require(rec.finish());
    if (recorded->dispatches() != samples) throw std::runtime_error("nombre de dispatches invalide");
    return recorded;
  }

  void timed(vk::Recording& recorded, std::string_view geometry, unsigned rotation, unsigned pass) {
    require(ctx->submit(recorded));
    require(ctx->wait_on_host(recorded.last_run()));
    std::array<std::uint64_t, samples + 1> ticks{};
    require(vk::check(
        pool.fn().results(ctx->device(), pool.handle(), 0, pool.count(), sizeof(ticks), ticks.data(),
                          sizeof(ticks[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults"));
    for (unsigned i = 0; i < samples; ++i)
      std::cout << geometry << '\t' << rotation << '\t' << pass << '\t' << i << '\t'
                << vk::net::ticks_to_ns(ticks[i], ticks[i + 1], support) << '\n';
  }

  void quality(bool xy_geometry) {
    const std::vector sentinel(expected.size(), std::byte{0x71});
    auto& frame = frames.front();
    require(vk::upload(*ctx, *memory, frame.dst, 0, sentinel));
    auto rec = require(vk::Recorder::begin(*ctx, *memory));
    dispatch(rec, frame, xy_geometry);
    auto cmd = require(rec.finish());
    require(ctx->submit(*cmd));
    require(ctx->wait_on_host(cmd->last_run()));
    std::vector<std::byte> got(expected.size());
    require(vk::download(*ctx, *memory, frame.dst, 0, got));
    if (got != expected) throw std::runtime_error("écart FP16, bord ou canal hors tranche altéré");
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const auto code = load<std::uint32_t>(argv[1]);
    Probe probe(code, load<std::byte>(argv[2]));
    std::cerr << probe.ctx->caps().device_name << '\n';
    probe.quality(false);
    probe.quality(true);
    for (const unsigned rotation : {1U, buffers}) {
      auto z = probe.record(false, rotation), xy = probe.record(true, rotation);
      // Un échauffement complet par géométrie, puis ABBA répété.
      for (unsigned pass = 0; pass < 10; ++pass) {
        const bool candidate = pass < 2 ? pass == 1 : (pass - 2) % 4 == 1 || (pass - 2) % 4 == 2;
        probe.timed(candidate ? *xy : *z, candidate ? "xy" : "z", rotation, pass);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
