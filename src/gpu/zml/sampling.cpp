#include <algorithm>
#include <format>
#include <limits>

#include "gpu/ir/graph.h"
#include "gpu/ir/sample_coordinates.h"
#include "gpu/zml/drba_resize.h"
#include "gpu/zml/hlo.h"
#include "gpu/zml/rife_sampling_plan.h"

namespace kirari::gpu::zml {
namespace {
// Un gather sur un axe spatial, pour les voisins d'un Resize séparable.
Value take(Hlo& h, const Value& x, std::size_t axis, const Dims& indices) {
  // Les réductions 1/2, 1/4, 1/8 et 1/16 du flot lisent toujours le même décalage
  // dans chaque groupe. Une slice stridée donne directement ces adresses à la fusion.
  const auto stride = indices.size() > 1 ? indices[1] - indices[0] : 1;
  if (stride > 0 && std::adjacent_find(indices.begin(), indices.end(), [stride](auto a, auto b) {
                      return b - a != stride;
                    }) == indices.end()) {
    Dims start(x.dims.size(), 0), end = x.dims, strides(x.dims.size(), 1);
    start[axis] = indices.front();
    end[axis] = indices.back() + 1;
    strides[axis] = stride;
    return h.slice(x, start, end, strides);
  }
  Dims offsets = axes(x.dims.size());
  offsets.erase(offsets.begin() + static_cast<std::ptrdiff_t>(axis));
  std::string list;
  for (auto k : offsets) list += (list.empty() ? "" : ", ") + std::to_string(k);
  Dims dims = x.dims, sizes = x.dims;
  dims[axis] = static_cast<std::int64_t>(indices.size());
  sizes[axis] = 1;
  auto ids = h.constant({.type = ir::DataType::I64, .dims = {dims[axis]}, .i = indices});
  return h.op("gather", {x, ids}, dims, x.element,
              std::format("dimension_numbers = #stablehlo.gather<offset_dims = [{}], collapsed_slice_dims = "
                          "[{}], start_index_map = [{}], index_vector_dim = 1>, slice_sizes = {}",
                          list, axis, axis, array(sizes)));
}

Value reduced_warp_planes(Hlo&, const ir::Graph&, const ir::Node&, const std::vector<Value>&,
                          const CompactWarpPlan&);

#ifdef KIRARI_ZML_RESIZE_PHASES
// Pour un agrandissement half_pixel pair de rapport r, entre deux pixels adjacents
// s'intercalent r combinaisons aux poids (j+1/2)/r. Les r/2 valeurs de chaque bord
// reprennent simplement le premier ou le dernier pixel. Aucun adressage indirect.
Value upsample_phases(Hlo& h, const Value& x, std::size_t axis, std::int64_t factor) {
  const auto size = x.dims[axis];
  const auto rank = x.dims.size();
  const auto span = [&](std::int64_t begin, std::int64_t limit) {
    Dims starts(rank, 0), ends = x.dims;
    starts[axis] = begin;
    ends[axis] = limit;
    return h.slice(x, starts, ends, Dims(rank, 1));
  };
  Dims out = x.dims;
  out[axis] *= factor;
  if (size == 1) return h.broadcast(x, out, axes(rank));
  const auto left = span(0, size - 1), right = span(1, size);
  Dims phases = left.dims;
  const auto phase = static_cast<std::int64_t>(axis + 1);
  phases.insert(phases.begin() + phase, factor);
  auto mapping = axes(rank + 1);
  mapping.erase(mapping.begin() + phase);
  std::vector<float> weights;
  for (std::int64_t j = 0; j < factor; ++j)
    weights.push_back((static_cast<float>(j) + 0.5F) / static_cast<float>(factor));
  const auto w =
      h.broadcast(h.constant({.type = ir::DataType::F32, .dims = {factor}, .f = weights}), phases, {phase});
  const auto a = h.binary("multiply", h.broadcast(left, phases, mapping),
                          h.binary("subtract", h.number(1, "f32"), w, phases), phases);
  const auto b = h.binary("multiply", h.broadcast(right, phases, mapping), w, phases);
  Dims interior = x.dims, ends = x.dims;
  interior[axis] = (size - 1) * factor;
  ends[axis] = factor / 2;
  const auto mid = h.reshape(h.binary("add", a, b, phases), interior);
  return h.op("concatenate", {h.expand(span(0, 1), ends), mid, h.expand(span(size - 1, size), ends)}, out,
              x.element, std::format("dimension = {} : i64", axis));
}
#endif

Value resize_axis(Hlo& h, const Value& x, const ir::ResizeAttrs& a, std::size_t i, std::int64_t size,
                  float scale) {
#ifdef KIRARI_ZML_RESIZE_PHASES
  const auto factor = size / x.dims[i];
  if (a.mode == ir::ResizeMode::Linear &&
      (a.coord == ir::CoordMode::HalfPixel || a.coord == ir::CoordMode::PytorchHalfPixel) &&
      (factor == 2 || factor == 4 || factor == 8 || factor == 16) && scale == static_cast<float>(factor) &&
      size == x.dims[i] * factor && size <= (1 << 23)) {
    return upsample_phases(h, x, i, factor);
  }
#endif
  Dims lo, hi;
  std::vector<float> weights;
  for (std::int64_t j = 0; j < size; ++j) {
    const auto coord = ir::detail::original_coordinate(a.coord, j, scale, x.dims[i], size);
    if (a.mode == ir::ResizeMode::Nearest)
      lo.push_back(ir::detail::nearest_index(a.nearest, coord, x.dims[i]));
    else {
      const auto taps = ir::detail::linear_taps(coord, x.dims[i]);
      lo.push_back(taps.lo);
      hi.push_back(taps.hi);
      weights.push_back(taps.t);
    }
  }
  auto left = take(h, x, i, lo);
  if (hi.empty()) return left;
  auto right = take(h, x, i, hi);
  const auto weight = weights.front();
  auto w = std::ranges::all_of(weights, [weight](float v) { return v == weight; })
               ? h.expand(h.number(weight, "f32"), left.dims)
               : h.broadcast(h.constant({.type = ir::DataType::F32, .dims = {size}, .f = weights}), left.dims,
                             {static_cast<std::int64_t>(i)});
  auto first = h.binary("subtract", h.number(1, "f32"), w, w.dims);
  return h.binary("add", h.binary("multiply", first, left, left.dims),
                  h.binary("multiply", w, right, left.dims), left.dims);
}

Value resize(Hlo& h, const ir::Graph& g, const ir::Node& n, const std::vector<Value>& values,
             const Dims& out) {
  auto x = values[n.inputs[0]];
  const auto& a = std::get<ir::ResizeAttrs>(n.attrs);
  const auto rank = out.size();
  auto mapping = a.axes.empty() ? axes(rank) : a.axes;
  std::vector<float> scales(rank);
  for (std::size_t i = 0; i < rank; ++i)
    scales[i] = static_cast<float>(out[i]) / static_cast<float>(x.dims[i]);
  if (n.inputs.size() > 2 && n.inputs[2] != ir::kNoTensor) {
    const auto& given = g.tensors[n.inputs[2]].value.f;
    for (std::size_t j = 0; j < given.size(); ++j) {
      auto axis = mapping[j];
      if (axis < 0) axis += static_cast<std::int64_t>(rank);
      scales[static_cast<std::size_t>(axis)] = given[j];
    }
  }
  const auto original = x.element;
#ifdef KIRARI_ZML_DRBA_DIRECT_RESIZE
  if (const int factor = drba_resize_factor(g, n, scales)) {
    const int first = factor / 2 - 1;
    const auto tap = [&](int row, int col) {
      return h.convert(h.slice(x, {0, 0, first + row, first + col}, x.dims, {1, 1, factor, factor}), "f32");
    };
    const auto half = h.number(0.5, "f32");
    const auto mix = [&](const Value& left, const Value& right) {
      return h.binary("add", h.binary("multiply", left, half, out), h.binary("multiply", right, half, out),
                      out);
    };
    const auto top = mix(tap(0, 0), tap(0, 1));
    const auto bottom = mix(tap(1, 0), tap(1, 1));
    return h.convert(mix(top, bottom), original);
  }
#endif
  if (rank == 4) {
    if (auto plan = compact_warp_plan(g, n, scales[2], scales[3]);
        plan && !values[plan->warp.inputs[0]].id.empty()) {
      x = reduced_warp_planes(h, g, n, values, *plan);
      scales[2] = scales[3] = 0.5F;
    }
  }
  if (a.mode == ir::ResizeMode::Linear) x = h.convert(x, "f32");
  for (std::size_t i = rank; i-- > rank - 2;) {
    if (x.dims[i] == out[i] && scales[i] == 1) continue;
    x = resize_axis(h, x, a, i, out[i], scales[i]);
  }
  return h.convert(x, original);
}

// Une seule fenêtre de 2 × 2 pixels par coordonnée, partagée par les canaux du lot.
// N est un axe de lot du gather et C un axe de la fenêtre. La sortie reste planaire.
Value border_window(Hlo& h, const Value& x, const Value& px, const Value& py, const Dims& dims) {
  const auto& plane = px.dims;  // [N,Hs,Ws,1]
  const auto origin = [&](const Value& pixel, std::int64_t size) {
    auto cell = h.op("floor", {pixel}, plane, "f32");
    // Au bord supérieur, gather ramène le début de la fenêtre à taille - 2.
    // Le poids vaut alors 1 : le pixel du bord est le second de la fenêtre.
    return h.binary("minimum", cell, h.number(static_cast<double>(size - 2), "f32"), plane);
  };
  const auto left = origin(px, x.dims[3]), top = origin(py, x.dims[2]);
  const auto indices = h.op("concatenate", {h.convert(top, "i32"), h.convert(left, "i32")},
                            {plane[0], plane[1], plane[2], 2}, "i32", "dimension = 3 : i64");
  Dims window = dims;
  window.insert(window.end(), {2, 2});
  const auto pixels = h.op(
      "gather", {x, indices}, window, x.element,
      std::format("dimension_numbers = #stablehlo.gather<offset_dims = [1, 4, 5], "
                  "operand_batching_dims = [0], start_indices_batching_dims = [0], "
                  "start_index_map = [2, 3], index_vector_dim = 3>, slice_sizes = array<i64: 1, {}, 2, 2>",
                  dims[1]));
  const auto tap = [&](std::int64_t dy, std::int64_t dx) {
    Dims end = dims;
    end.insert(end.end(), {dy + 1, dx + 1});
    return h.convert(h.reshape(h.slice(pixels, {0, 0, 0, 0, dy, dx}, end, {1, 1, 1, 1, 1, 1}), dims), "f32");
  };
  const auto mix = [&](const Value& a, const Value& b, const Value& pixel, const Value& base) {
    const auto w = h.reshape(h.binary("subtract", pixel, base, plane), {plane[0], 1, plane[1], plane[2]});
    const auto complement = h.binary("subtract", h.number(1, "f32"), w, w.dims);
    return h.binary("add", h.binary("multiply", a, complement, dims), h.binary("multiply", b, w, dims), dims);
  };
  return mix(mix(tap(0, 0), tap(0, 1), px, left), mix(tap(1, 0), tap(1, 1), px, left), py, top);
}

// Image source [H,W,N,C] : les sept plans d'une image sont contigus dans le cache
// PJRT de RIFE 4.22 Lite. Un seul indice désigne le pixel, les quatre voisins ont
// des décalages constants. Le gather garde les pixels en FP16 jusqu'au mélange.
Value linear_border(Hlo& h, const Value& pixels, const Value& px, const Value& py) {
  const auto& plane = px.dims;  // [N,H,W,1]
  const auto n = plane[0], height = plane[1], width = plane[2], channels = pixels.dims[1];
  const auto origin = [&](const Value& pixel, std::int64_t size) {
    return h.binary("minimum", h.op("floor", {pixel}, plane, "f32"),
                    h.number(static_cast<double>(size - 2), "f32"), plane);
  };
  const auto top = origin(py, height), left = origin(px, width);
  const auto linear = h.binary(
      "add", h.binary("multiply", h.convert(top, "i32"), h.number(static_cast<double>(width), "i32"), plane),
      h.convert(left, "i32"), plane);
  const auto batch = h.op("iota", {}, plane, "i32", "iota_dimension = 0 : i64");
  const auto base = h.binary(
      "add", h.binary("multiply", linear, h.number(static_cast<double>(n), "i32"), plane), batch, plane);
  const auto table = h.reshape(h.transpose(pixels, {2, 3, 0, 1}), {height * width * n, channels});
  const auto step = width * n;
  const auto offsets =
      h.convert(h.constant({.type = ir::DataType::I64, .dims = {4}, .i = {0, n, step, step + n}}), "i32");
  const Dims result{height, width, n, channels}, window{height, width, n, 4, channels};
  const auto ids = h.binary("add", h.transpose(base, {1, 2, 0, 3}), offsets, {height, width, n, 4});
  const auto gathered = h.op("gather", {table, ids}, window, table.element,
                             std::format("dimension_numbers = #stablehlo.gather<offset_dims = [4], "
                                         "collapsed_slice_dims = [0], start_index_map = [0], "
                                         "index_vector_dim = 4>, slice_sizes = array<i64: 1, {}>",
                                         channels));
  const auto tap = [&](std::int64_t i) {
    const auto sample =
        h.slice(gathered, {0, 0, 0, i, 0}, {height, width, n, i + 1, channels}, {1, 1, 1, 1, 1});
    return h.convert(h.reshape(sample, result), "f32");
  };
  const auto mix = [&](const Value& a, const Value& b, const Value& w) {
    return h.binary("add",
                    h.binary("multiply", a, h.binary("subtract", h.number(1, "f32"), w, w.dims), result),
                    h.binary("multiply", b, w, result), result);
  };
  const auto wx = h.transpose(h.binary("subtract", px, left, plane), {1, 2, 0, 3});
  const auto wy = h.transpose(h.binary("subtract", py, top, plane), {1, 2, 0, 3});
  return h.transpose(mix(mix(tap(0), tap(1), wx), mix(tap(2), tap(3), wx), wy), {2, 3, 0, 1});
}

// Coordonnées ONNX normalisées et poids bilinéaires calculés en float, y compris pour un réseau FP16.
Value grid_sample(Hlo& h, const Value& x, const Value& grid, const ir::GridSampleAttrs& a, const Dims& dims,
                  const Value* linear_source = nullptr) {
  const auto coords = h.convert(a.grid_nchw ? h.transpose(grid, {0, 2, 3, 1}) : grid, "f32");
  const auto& shape = coords.dims;
  Dims plane{shape[0], shape[1], shape[2], 1};
  const auto pixel = [&](int axis, std::int64_t size) {
    auto u = h.slice(coords, {0, 0, 0, axis}, {shape[0], shape[1], shape[2], axis + 1}, {1, 1, 1, 1});
    auto scalar = [&](double k) { return h.number(k, "f32"); };
    u = h.binary("add", u, scalar(1), plane);
    u = h.binary("multiply", u, scalar(static_cast<double>(a.align_corners ? size - 1 : size)), plane);
    if (!a.align_corners) u = h.binary("subtract", u, scalar(1), plane);
    u = h.binary("multiply", u, scalar(0.5), plane);
    const bool border = a.padding == ir::GridPadding::Border;
    return h.op("clamp",
                {scalar(border ? 0 : -2), u, scalar(static_cast<double>(border ? size - 1 : size + 1))},
                plane, "f32");
  };
  auto px = pixel(0, x.dims[3]), py = pixel(1, x.dims[2]);
  if (linear_source) return h.convert(linear_border(h, *linear_source, px, py), x.element);
  const bool nearest = a.mode == ir::GridMode::Nearest;
  if (!nearest && a.padding == ir::GridPadding::Border && x.dims[2] >= 2 && x.dims[3] >= 2)
    return h.convert(border_window(h, x, px, py, dims), x.element);
  const auto img = h.transpose(x, {0, 2, 3, 1});
  const auto round = nearest ? "round_nearest_even" : "floor";
  auto fx = h.op(round, {px}, plane, "f32"), fy = h.op(round, {py}, plane, "f32");
  auto ix = h.convert(fx, "i32"), iy = h.convert(fy, "i32");
  auto batch = h.op("iota", {}, plane, "i32", "iota_dimension = 0 : i64");
  Dims dst{dims[0], dims[2], dims[3], dims[1]};
  const auto tap = [&](const Value& row, const Value& col) {
    auto indices = h.op("concatenate", {batch, row, col}, {plane[0], plane[1], plane[2], 3}, "i32",
                        "dimension = 3 : i64");
    auto y = h.convert(
        h.op("gather", {img, indices}, dst, img.element,
             std::format(
                 "dimension_numbers = #stablehlo.gather<offset_dims = [3], collapsed_slice_dims = [0, 1, 2], "
                 "start_index_map = [0, 1, 2], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 1, {}>",
                 dims[1])),
        "f32");
    if (a.padding == ir::GridPadding::Border) return y;
    const auto compare = [&](const Value& v, std::int64_t k, std::string_view cmp) {
      return h.op("compare", {v, h.expand(h.number(static_cast<double>(k), "i32"), plane)}, plane, "i1",
                  std::format("comparison_direction = #stablehlo<comparison_direction {}>, compare_type = "
                              "#stablehlo<comparison_type SIGNED>",
                              cmp));
    };
    auto inside = [&](const Value& v, std::int64_t size) {
      return h.binary("and", compare(v, 0, "GE"), compare(v, size, "LT"), plane);
    };
    auto valid = h.binary("and", inside(row, x.dims[2]), inside(col, x.dims[3]), plane);
    return h.op("select", {h.expand(valid, dst), y, h.expand(h.number(0, "f32"), dst)}, dst, "f32");
  };
  auto y = tap(iy, ix);
  if (!nearest) {
    auto ix1 = h.binary("add", ix, h.number(1, "i32"), plane),
         iy1 = h.binary("add", iy, h.number(1, "i32"), plane);
    auto lerp = [&](const Value& lo, const Value& hi, const Value& w) {
      auto first = h.binary("subtract", h.number(1, "f32"), w, plane);
      return h.binary("add", h.binary("multiply", lo, first, dst), h.binary("multiply", hi, w, dst), dst);
    };
    auto wx = h.binary("subtract", px, fx, plane), wy = h.binary("subtract", py, fy, plane);
    y = lerp(lerp(y, tap(iy, ix1), wx), lerp(tap(iy1, ix), tap(iy1, ix1), wx), wy);
  }
  return h.convert(h.transpose(y, {0, 3, 1, 2}), x.element);
}

}  // namespace
#ifdef KIRARI_ZML_LINEAR_WARP
namespace {
bool is_full_resolution_downconv(const ir::Graph& g, const ir::Node* conv) {
  if (!conv || conv->op != ir::OpKind::Conv || std::get<ir::ConvAttrs>(conv->attrs).strides != ir::Pair{2, 2})
    return false;
  const auto& kernel = g.tensors[conv->inputs[1]].dims;
  if (kernel.size() != 4 || kernel[1] != 28 || kernel[2] != 3 || kernel[3] != 3) return false;
  return true;
}

bool feeds_full_resolution_block(const ir::Graph& g, ir::GraphIndex& index, const ir::Node& warp,
                                 std::int64_t h, std::int64_t w) {
  using enum ir::OpKind;
  const auto next = [&](ir::TensorId id) -> const ir::Node* {
    const auto reader = index.sole_reader(id);
    return reader ? &g.nodes[*reader] : nullptr;
  };
  const auto* split = next(warp.outputs[0]);
  if (split && split->op == Cast) split = next(split->outputs[0]);
  if (!split || split->op != Split || split->outputs.size() != 2 ||
      std::get<ir::SplitAttrs>(split->attrs).axis != 1)
    return false;
  if (g.tensors[split->outputs[0]].dims != Dims{2, 3, h, w} ||
      g.tensors[split->outputs[1]].dims != Dims{2, 4, h, w})
    return false;
  const auto* rgb = next(split->outputs[0]);
  const auto* features = next(split->outputs[1]);
  if (!rgb || !features || rgb->op != Reshape || features->op != Reshape) return false;
  const auto* join = next(rgb->outputs[0]);
  if (!join || join != next(features->outputs[0]) || join->op != Concat ||
      std::get<ir::AxisAttrs>(join->attrs).axis != 1 || g.tensors[join->outputs[0]].dims != Dims{1, 28, h, w})
    return false;
  return is_full_resolution_downconv(g, next(join->outputs[0]));
}
}  // namespace

std::optional<ir::TensorId> linear_warp_source(const ir::Graph& g, const ir::Node& warp) {
  using enum ir::OpKind;
  const auto& a = std::get<ir::GridSampleAttrs>(warp.attrs);
  const auto& shape = g.tensors[warp.inputs[0]].dims;
  if (a.mode != ir::GridMode::Bilinear || a.padding != ir::GridPadding::Border || !a.align_corners ||
      shape.size() != 4 || shape[0] != 2 || shape[1] != 7 || g.tensors[warp.outputs[0]].dims != shape)
    return {};
  const auto h = shape[2], w = shape[3];
  // Les origines des fenêtres sont représentables exactement en FP32, tous les indices en i32.
  if (h < 2 || w < 2 || h > (1 << 24) || w > (1 << 24) ||
      h > std::numeric_limits<std::int32_t>::max() / shape[0] / w)
    return {};
  ir::GraphIndex index(g);
  auto source = warp.inputs[0];
  if (g.tensors[source].type == ir::DataType::F32) {
    const auto parent = index.producer(source);
    if (parent == ir::kNoNode || g.nodes[parent].op != Cast) return {};
    source = g.nodes[parent].inputs[0];
  }
  if (g.tensors[source].type != ir::DataType::F16) return {};
  if (!feeds_full_resolution_block(g, index, warp, h, w)) return {};
  return source;
}
#endif

namespace {
std::optional<std::int64_t> compact_resize_factor(const ir::ResizeAttrs& attrs, const Dims& d,
                                                  const Dims& out, float scale_h, float scale_w,
                                                  bool include_half) {
  if (d.size() != 4 || out.size() != 4 || out[2] <= 0 || out[3] <= 0 ||
      attrs.mode != ir::ResizeMode::Linear || attrs.coord != ir::CoordMode::HalfPixel)
    return {};
  const auto factor = d[2] / out[2];
  if (factor < (include_half ? 2 : 4) || factor % 2 != 0 || d[2] != factor * out[2] ||
      d[3] != factor * out[3] || scale_h != 1.F / static_cast<float>(factor) || scale_h != scale_w)
    return {};
  return factor;
}
}  // namespace

std::optional<CompactWarpPlan> compact_warp_plan(const ir::Graph& g, const ir::Node& resize, float scale_h,
                                                 float scale_w, bool include_half) {
  const auto dest = resize.inputs[0];
  const auto& d = g.tensors[dest].dims;
  const auto& out = g.tensors[resize.outputs[0]].dims;
  const auto factor =
      compact_resize_factor(std::get<ir::ResizeAttrs>(resize.attrs), d, out, scale_h, scale_w, include_half);
  if (!factor) return {};
  const auto producers = ir::producers(g);
  const auto parent = [&](ir::TensorId t, ir::OpKind op) -> const ir::Node* {
    const auto i = producers[t];
    return i != ir::kNoNode && g.nodes[i].op == op ? &g.nodes[i] : nullptr;
  };
  const auto uncast = [&](ir::TensorId t) {
    const auto* cast = parent(t, ir::OpKind::Cast);
    return cast ? cast->inputs[0] : t;
  };
  using enum ir::OpKind;
  const auto* join = parent(uncast(dest), Concat);
  if (!join || std::get<ir::AxisAttrs>(join->attrs).axis != 1 || join->inputs.size() < 2) return {};
  const auto* rgb = parent(join->inputs[0], Reshape);
  const auto* features = parent(join->inputs[1], Reshape);
  if (!rgb || !features) return {};
  const auto* split = parent(rgb->inputs[0], Split);
  if (!split || split->outputs.size() != 2 || split->outputs[0] != rgb->inputs[0] ||
      split->outputs[1] != features->inputs[0] || std::get<ir::SplitAttrs>(split->attrs).axis != 1)
    return {};
  const auto* warp = parent(uncast(split->inputs[0]), GridSample);
  if (!warp) return {};
  for (auto id : split->outputs) {
    const auto& shape = g.tensors[id].dims;
    if (shape.size() != 4 || shape[2] != d[2] || shape[3] != d[3]) return {};
  }
  return CompactWarpPlan{
      *warp, *split, *join, *factor, uncast(split->inputs[0]) != split->inputs[0], uncast(dest) != dest};
}

namespace {
// Chaque cellule r × r du Resize half_pixel lit seulement ses quatre pixels centraux.
// En resserrant la grille de RIFE avant GridSample, XLA calcule précisément ces warps.
Value reduced_warp_planes(Hlo& h, const ir::Graph& g, const ir::Node& resize, const std::vector<Value>& v,
                          const CompactWarpPlan& plan) {
  const auto& [warp, split, join, factor, cast_warp, cast_planes] = plan;
  const auto dest = resize.inputs[0];
  const auto& d = v[dest].dims;
  const auto replay_cast = [&](Value a, ir::TensorId t, bool explicit_cast) {
    // Même un Cast FP16 immédiatement réélargi en FP32 arrondit le signal.
    if (explicit_cast) a = h.convert(a, element_type(g.tensors[t].type));
    return h.convert(a, v[t].element);
  };
  const auto centers = [&](Value a, std::size_t spatial) {
    // [H] -> [H/r,r] -> [H/r,2] -> [2H/r] ; même chose sur W.
    for (auto i : {spatial, spatial + 1}) {
      auto shape = a.dims;
      shape[i] /= factor;
      shape.insert(shape.begin() + static_cast<std::ptrdiff_t>(i + 1), factor);
      a = h.reshape(a, shape);
      Dims start(shape.size(), 0), end = shape;
      start[i + 1] = factor / 2 - 1;
      end[i + 1] = start[i + 1] + 2;
      a = h.slice(a, start, end, Dims(shape.size(), 1));
      shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(i + 1));
      shape[i] *= 2;
      a = h.reshape(a, shape);
    }
    return a;
  };
  const auto& attrs = std::get<ir::GridSampleAttrs>(warp.attrs);
  auto grid = centers(v[warp.inputs[1]], attrs.grid_nchw ? 2 : 1);
  Dims reduced = v[warp.outputs[0]].dims;
  reduced[2] = 2 * d[2] / factor;
  reduced[3] = 2 * d[3] / factor;
  auto sampled = grid_sample(h, v[warp.inputs[0]], grid, attrs, reduced);
  sampled = replay_cast(sampled, split.inputs[0], cast_warp);
  std::vector<Value> parts;
  std::int64_t channel = 0;
  for (std::size_t i = 0; i < join.inputs.size(); ++i) {
    const auto id = join.inputs[i];
    if (i < 2) {
      Dims end = reduced, shape = v[id].dims;
      end[1] = channel + v[split.outputs[i]].dims[1];
      auto part = h.slice(sampled, {0, channel, 0, 0}, end, {1, 1, 1, 1});
      shape[2] = reduced[2];
      shape[3] = reduced[3];
      parts.push_back(h.reshape(part, shape));
      channel = end[1];
    } else {
      parts.push_back(centers(v[id], 2));
    }
  }
  return replay_cast(h.op("concatenate", parts, {d[0], d[1], reduced[2], reduced[3]}, parts.front().element,
                          "dimension = 1 : i64"),
                     dest, cast_planes);
}
}  // namespace
std::expected<Value, Refusal> sample(Hlo& h, const ir::Graph& g, const ir::Node& n,
                                     const std::vector<Value>& v) {
  const auto& x = v[n.inputs[0]];
  const auto& dims = g.tensors[n.outputs[0]].dims;
  if (n.op == ir::OpKind::Resize) return resize(h, g, n, v, dims);
#ifdef KIRARI_ZML_LINEAR_WARP
  if (const auto raw = linear_warp_source(g, n); raw && !v[*raw].id.empty() && v[*raw].element == "f16")
    return grid_sample(h, x, v[n.inputs[1]], std::get<ir::GridSampleAttrs>(n.attrs), dims, &v[*raw]);
#endif
  return grid_sample(h, x, v[n.inputs[1]], std::get<ir::GridSampleAttrs>(n.attrs), dims);
}
}  // namespace kirari::gpu::zml
