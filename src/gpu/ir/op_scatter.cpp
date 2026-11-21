#include <algorithm>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace kirari::gpu::ir::detail {
Inferred infer_scatter_nd(const Node& n, std::span<const Operand> in) {
  const auto& data = *in[0].dims;
  const auto& indices = *in[1].dims;
  if (data.empty() || indices.empty() || indices.back() < 1 ||
      indices.back() > static_cast<std::int64_t>(data.size()))
    return shape_error(n, "rang de l'entrée ou profondeur des indices invalide pour ScatterND");
  if (in[1].type != DataType::I64 || in[0].type != in[2].type)
    return graph_error(n, "types d'indices ou de mises à jour incompatibles dans ScatterND");
  Dims expected(indices.begin(), indices.end() - 1);
  expected.insert(expected.end(), data.begin() + indices.back(), data.end());
  if (expected != *in[2].dims) return shape_error(n, "forme des mises à jour incompatible dans ScatterND");
  return std::vector<OutputInfo>{{in[0].type, data}};
}

Evaluated eval_scatter_nd(const Node& n, std::span<const Value* const> in, std::span<const OutputInfo>) {
  Value out = *in[0];
  const auto& indices = *in[1];
  const auto depth = static_cast<std::size_t>(indices.dims.back());
  const auto strides = row_strides(out.dims);
  const auto length = strides[depth - 1];
  for (std::size_t tuple = 0; tuple < indices.i.size() / depth; ++tuple) {
    std::size_t start = 0;
    for (std::size_t axis = 0; axis < depth; ++axis) {
      auto pos = indices.i[tuple * depth + axis];
      if (pos < 0) pos += out.dims[axis];
      if (pos < 0 || pos >= out.dims[axis]) return shape_error(n, "indice hors du tenseur dans ScatterND");
      start += static_cast<std::size_t>(pos) * strides[axis];
    }
    const auto copy = [&](const auto& updates, auto& data) {
      std::copy_n(updates.begin() + static_cast<std::ptrdiff_t>(tuple * length), length,
                  data.begin() + static_cast<std::ptrdiff_t>(start));
    };
    if (out.type == DataType::I64)
      copy(in[2]->i, out.i);
    else
      copy(in[2]->f, out.f);
  }
  return std::vector<Value>{std::move(out)};
}
}  // namespace kirari::gpu::ir::detail
