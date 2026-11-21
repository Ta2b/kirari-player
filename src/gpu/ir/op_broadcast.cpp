#include <algorithm>
#include <numeric>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace kirari::gpu::ir::detail {
namespace {
std::expected<Dims, Refusal> sizes(const Node& n, std::span<const Operand> in, std::size_t k) {
  if (in[k].dims->size() != 1) return shape_error(n, "la forme doit être un vecteur");
  auto dims = const_ints(n, in, k, "la forme");
  if (!dims) return std::unexpected(dims.error());
  if (std::ranges::any_of(*dims, [](auto x) { return x < 0 || x > kMaxDim; }))
    return shape_error(n, "dimension invalide");
  return dims;
}
Dims broadcast_strides(const Dims& src, const Dims& dst) {
  Dims strides(dst.size());
  const auto local = row_strides(src);
  for (std::size_t i = 0; i < src.size(); ++i)
    if (src[i] != 1) strides[dst.size() - src.size() + i] = static_cast<std::int64_t>(local[i]);
  return strides;
}
}  // namespace
Inferred infer_fill(const Node& n, std::span<const Operand> in) {
  auto dims = sizes(n, in, 0);
  if (!dims) return std::unexpected(dims.error());
  auto a = attrs_of<FillAttrs>(n);
  if (!a) return std::unexpected(a.error());
  if (value_size((*a)->scalar) != 1) return graph_error(n, "valeur scalaire attendue");
  return std::vector<OutputInfo>{{(*a)->scalar.type, std::move(*dims)}};
}
Evaluated eval_fill(const Node& n, std::span<const Value* const>, std::span<const OutputInfo> outs) {
  auto out = make_output(outs[0]);
  const auto& value = std::get<FillAttrs>(n.attrs).scalar;
  if (out.type == DataType::I64)
    std::ranges::fill(out.i, value.i.front());
  else
    std::ranges::fill(out.f, value.f.front());
  return std::vector<Value>{std::move(out)};
}
Inferred infer_expand(const Node& n, std::span<const Operand> in) {
  auto dims = sizes(n, in, 1);
  if (!dims) return std::unexpected(dims.error());
  auto result = broadcast_dims(n, *in[0].dims, *dims);
  if (!result) return std::unexpected(result.error());
  return std::vector<OutputInfo>{{in[0].type, std::move(*result)}};
}
Inferred infer_where(const Node& n, std::span<const Operand> in) {
  if (in[0].type != DataType::I64 || in[1].type != in[2].type)
    return graph_error(n, "types incompatibles dans Where");
  auto xy = broadcast_dims(n, *in[1].dims, *in[2].dims);
  if (!xy) return std::unexpected(xy.error());
  auto out = broadcast_dims(n, *in[0].dims, *xy);
  if (!out) return std::unexpected(out.error());
  return std::vector<OutputInfo>{{in[1].type, std::move(*out)}};
}
Evaluated eval_expand(const Node&, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto out = make_output(outs[0]);
  const auto steps = broadcast_strides(in[0]->dims, out.dims);
  for_each_index(out.dims, [&](auto index, std::size_t pos) {
    const auto at = std::inner_product(index.begin(), index.end(), steps.begin(), std::size_t{0});
    if (out.type == DataType::I64)
      out.i[pos] = in[0]->i[at];
    else
      out.f[pos] = in[0]->f[at];
  });
  return std::vector<Value>{std::move(out)};
}
Evaluated eval_where(const Node&, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto out = make_output(outs[0]);
  const std::array steps{broadcast_strides(in[0]->dims, out.dims), broadcast_strides(in[1]->dims, out.dims),
                         broadcast_strides(in[2]->dims, out.dims)};
  for_each_index(out.dims, [&](auto index, std::size_t pos) {
    const auto at = [&](std::size_t k) {
      return std::inner_product(index.begin(), index.end(), steps[k].begin(), std::size_t{0});
    };
    std::size_t k = in[0]->i[at(0)] ? 1 : 2;
    if (out.type == DataType::I64)
      out.i[pos] = in[k]->i[at(k)];
    else
      out.f[pos] = in[k]->f[at(k)];
  });
  return std::vector<Value>{std::move(out)};
}
Inferred infer_scan(const Node& n, std::span<const Operand> in) {
  if (auto ok = expect_float(n, in[0], "l’entrée"); !ok) return std::unexpected(ok.error());
  auto axis = const_ints(n, in, 1, "l’axe de CumSum");
  if (!axis) return std::unexpected(axis.error());
  if (axis->size() != 1) return graph_error(n, "un seul axe attendu");
  if (auto a = normalize_axis(n, axis->front(), in[0].dims->size()); !a) return std::unexpected(a.error());
  if (auto a = attrs_of<ScanAttrs>(n); !a) return std::unexpected(a.error());
  return std::vector<OutputInfo>{{in[0].type, *in[0].dims}};
}
Evaluated eval_scan(const Node& n, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  auto out = make_output(outs[0]);
  const auto attrs = std::get<ScanAttrs>(n.attrs);
  const auto axis = *normalize_axis(n, in[1]->i.front(), out.dims.size());
  const auto size = static_cast<std::size_t>(out.dims[axis]);
  const auto step = row_strides(out.dims)[axis];
  if (out.f.empty()) return std::vector<Value>{std::move(out)};
  for (std::size_t block = 0; block < out.f.size(); block += size * step) {
    for (std::size_t i = 0; i < step; ++i) {
      float sum = 0;
      for (std::size_t j = 0; j < size; ++j) {
        const auto at = block + i + (attrs.reverse ? size - 1 - j : j) * step;
        out.f[at] = attrs.exclusive ? sum : sum + in[0]->f[at];
        sum += in[0]->f[at];
      }
    }
  }
  return std::vector<Value>{std::move(out)};
}
}  // namespace kirari::gpu::ir::detail
