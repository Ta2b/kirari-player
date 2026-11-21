#include "gpu/zml/span_regions.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <format>

#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/sdk_process.h"
#include "gpu/zml/span_region_hlo.h"

namespace kirari::gpu::zml {
namespace {
constexpr int kRows = 6, kCols = 4, kCapacity = 24;
constexpr int kTW = 64, kTH = 16, kWH = kRows * kTH + 104, kWW = kCols * kTW + 104;
constexpr std::string_view kTag = "// kirari-span-regions-v1 ";
vk::net::IoNames region_names() {
  return {{"image", "previous", "metadata"}, {"output"}};
}

struct Geometry {
  int h, w;
  kernels::SpanTileGrid grid() const {
    // Même décision que le lecteur CUDA de référence, y compris son halo vertical Winograd prudent.
    return {.tile_w = kTW,
            .tile_h = kTH,
            .cols = kernels::span_tiles_along(w, kTW),
            .rows = kernels::span_tiles_along(h, kTH),
            .reach_cols = 1,
            .reach_rows = 7,
            .shift_step = 2,
            .scale = 2,
            .shift_step_y = 4};
  }
  bool valid() const {
    return h >= kWH && w >= kWW && h <= 16384 && w <= 16384 && h % 2 == 0 && w % 2 == 0 && grid().usable();
  }
};
using Point = std::pair<int, int>;

Point densest_cover(std::span<const std::uint8_t> todo, const kernels::SpanTileGrid& g, Point anchor) {
  const auto [y, x] = anchor;
  int best = -1;
  Point origin{};
  for (int r = std::max(0, y - kRows + 1); r <= std::min(y, g.rows - kRows); ++r) {
    for (int c = std::max(0, x - kCols + 1); c <= std::min(x, g.cols - kCols); ++c) {
      int count = 0;
      for (int i = r; i < r + kRows; ++i) {
        const auto cells = todo.subspan(static_cast<std::size_t>(i * g.cols + c), kCols);
        count += static_cast<int>(std::ranges::count(cells, std::uint8_t{1}));
      }
      if (count > best) {
        best = count;
        origin = {r, c};
      }
    }
  }
  return origin;
}
class Regions final : public vk::net::RuntimeSpanRegions {
 public:
  Regions(Geometry shape, std::unique_ptr<vk::net::RuntimeNet> narrow)
      : shape_(shape), net_(std::move(narrow)) {}
  kernels::SpanTileGrid grid() const override { return shape_.grid(); }
  Shape previous_shape() const override { return {{1, 3, 2 * shape_.h, 2 * shape_.w}}; }
  vk::net::RuntimeNet& network() override { return *net_; }
  std::vector<std::uint16_t> plan(std::span<const std::uint8_t> core) const override {
    const auto g = grid();
    if (core.size() != static_cast<std::size_t>(g.count())) return {};
    std::vector<std::uint8_t> todo(core.size());
    std::vector<std::uint16_t> metadata(core.size() + 2 * kCapacity + 1, 0);
    for (std::size_t i = 0; i < core.size(); ++i)
      if (core[i]) {
        todo[i] = 1;
        metadata[i] = 0x3c00;
      }
    // Au-delà du budget ou de la capacité, le lecteur est informé avant toute soumission GPU.
    const auto limit = std::min<std::int64_t>(kCapacity, 7LL * shape_.h * shape_.w / (10LL * kWH * kWW));
    int n = 0;
    for (;;) {
      const auto first = std::ranges::find(todo, std::uint8_t{1});
      if (first == todo.end()) break;
      if (n >= limit) return {};
      const auto index = static_cast<int>(std::distance(todo.begin(), first));
      const auto [r, c] = densest_cover(todo, g, {index / g.cols, index % g.cols});
      const auto offset = core.size() + 2 * static_cast<std::size_t>(n++);
      metadata[offset] = static_cast<std::uint16_t>(std::min(r * kTH, shape_.h - kRows * kTH));
      metadata[offset + 1] = static_cast<std::uint16_t>(std::min(c * kTW, shape_.w - kCols * kTW));
      for (int y = r; y < r + kRows; ++y) std::fill_n(todo.begin() + y * g.cols + c, kCols, std::uint8_t{0});
    }
    metadata.back() = static_cast<std::uint16_t>(n);
    return metadata;  // n=0 conserve toute la sortie d'avant.
  }

 private:
  Geometry shape_;
  std::unique_ptr<vk::net::RuntimeNet> net_;
};

struct Marker {
  Geometry shape;
  std::uint64_t hash;
};
std::optional<Marker> marker(std::span<const std::byte> source) {
  std::string_view s(reinterpret_cast<const char*>(source.data()), source.size());
  if (!s.starts_with(kTag)) return {};
  s.remove_prefix(kTag.size());
  auto take = [&](auto& value, char sep) {
    const auto p = s.find(sep);
    if (p == s.npos || p == 0) return false;
    const auto part = s.substr(0, p);
    const auto [end, ec] = std::from_chars(part.data(), part.data() + part.size(), value);
    s.remove_prefix(p + 1);
    return ec == std::errc{} && end == part.data() + part.size();
  };
  Marker m{};
  if (!take(m.shape.h, ' ') || !take(m.shape.w, ' ') || !take(m.hash, '\n') || !m.shape.valid()) return {};
  return m;
}
std::filesystem::path sibling(const std::filesystem::path& path, std::uint64_t key) {
  auto dest = path;
  dest += std::format(".region-{:016x}.mlir", key);
  return dest;
}
}  // namespace

std::expected<void, Refusal> prepare_span_regions(std::shared_ptr<Client> client,
                                                  const std::filesystem::path& dense,
                                                  const std::vector<std::int64_t>& d, Progress& progress) {
  if (d.size() != 4 || d[0] != 1 || d[1] != 3 || d[2] <= 0 || d[3] <= 0 || d[2] > 16384 || d[3] > 16384)
    return {};
  const Geometry shape{static_cast<int>(d[2]), static_cast<int>(d[3])};
  if (!shape.valid()) return {};
  auto weights = dense;
  weights += ".poids";
  const auto package = vk::net::read_file(weights);
  if (!package || package->size() < 40 || std::memcmp(package->data(), "AJPSPAN8", 8) != 0 ||
      (*package)[12] != std::byte{8} || (*package)[20] != std::byte{64})
    return {};
  // Le SDK officiel trace exactement les mêmes poids que le dense pour cette fenêtre.
  const auto tile = dense.parent_path() / "fenetre-regionale.mlir";
  auto emitted = run_sdk_frontend(
      {core::to_utf8(weights), std::to_string(kWH), std::to_string(kWW), core::to_utf8(tile)}, progress);
  if (!emitted) return std::unexpected(emitted.error());
  auto source = vk::net::read_file(tile);
  if (!source) return std::unexpected(source.error());
  auto module = wrap_span_regions(std::string(reinterpret_cast<const char*>(source->data()), source->size()),
                                  shape.h, shape.w);
  if (!module)
    return std::unexpected(
        Refusal{RefusalCode::BuilderFailed, "SPAN régional : signature de fenêtre non reconnue"});
  const auto bytes = std::as_bytes(std::span(*module));
  const auto key = vk::net::fnv1a(bytes);
  const auto regional = sibling(dense, key);
  auto exe = compile(client, bytes, region_names(), regional, &progress);
  if (!exe) return std::unexpected(exe.error());
  if (progress.stop.stop_requested())
    return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation SPAN régional annulée"});
  if (auto saved = vk::net::write_file(regional, bytes); !saved) return saved;
  auto full = vk::net::read_file(dense);
  if (!full) return std::unexpected(full.error());
  const auto tag = std::format("{}{} {} {}\n", kTag, shape.h, shape.w, key);
  const auto prefix = std::as_bytes(std::span(tag));
  full->insert(full->begin(), prefix.begin(), prefix.end());
  return vk::net::write_file(dense, *full);
}
void load_span_regions(std::shared_ptr<Client> client, const std::filesystem::path& dense,
                       std::span<const std::byte> source, vk::net::RuntimeNet& full) {
  const auto m = marker(source);
  if (!m) return;
  const auto regional = sibling(dense, m->hash);
  const auto bytes = vk::net::read_file(regional);
  if (!bytes || vk::net::fnv1a(*bytes) != m->hash) return;
  auto network = compile(client, *bytes, region_names(), regional, nullptr, true);
  if (network) full.span_regions = std::make_unique<Regions>(m->shape, std::move(*network));
}
}  // namespace kirari::gpu::zml
