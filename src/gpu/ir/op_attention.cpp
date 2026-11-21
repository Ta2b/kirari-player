#include <algorithm>
#include <cmath>
#include <limits>

#include "gpu/ir/graph.h"
#include "gpu/ir/op_impl.h"
#include "gpu/ir/op_util.h"

namespace kirari::gpu::ir::detail {
Inferred infer_tile(const Node& n, std::span<const Operand> in) {
  auto rep = const_ints(n, in, 1, "les répétitions");
  if (!rep) return std::unexpected(rep.error());
  auto d = *in[0].dims;
  if (in[1].dims->size() != 1 || d.size() != rep->size()) return shape_error(n, "rang de Tile incompatible");
  for (std::size_t j = 0; j < d.size(); ++j) {
    if ((*rep)[j] < 0 || (*rep)[j] > kMaxDim / std::max<std::int64_t>(1, d[j]))
      return shape_error(n, "répétition hors borne");
    d[j] *= (*rep)[j];
  }
  return std::vector<OutputInfo>{{in[0].type, std::move(d)}};
}
Evaluated eval_tile(const Node&, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const auto& x = *in[0];
  auto out = make_output(outs[0]);
  const auto strides = row_strides(x.dims);
  for_each_index(out.dims, [&](auto index, std::size_t pos) {
    std::size_t at = 0;
    for (std::size_t i = 0; i < index.size(); ++i)
      at += static_cast<std::size_t>(index[i] % x.dims[i]) * strides[i];
    if (out.type == DataType::I64)
      out.i[pos] = x.i[at];
    else
      out.f[pos] = x.f[at];
  });
  return std::vector<Value>{std::move(out)};
}
Inferred infer_matmul(const Node& n, std::span<const Operand> in) {
  const Dims& a = *in[0].dims;
  const Dims& b = *in[1].dims;
  if (a.size() < 2 || b.size() < 2) return unsupported_error(n, "MatMul attend deux matrices");
  if (in[0].type != in[1].type || in[0].type == DataType::I64 || a.back() != b[b.size() - 2])
    return shape_error(n, "produit matriciel incompatible");
  auto batch = broadcast_dims(n, Dims(a.begin(), a.end() - 2), Dims(b.begin(), b.end() - 2));
  if (!batch) return std::unexpected(batch.error());
  batch->push_back(a[a.size() - 2]);
  batch->push_back(b.back());
  return std::vector<OutputInfo>{{in[0].type, std::move(*batch)}};
}
Evaluated eval_matmul(const Node&, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const auto& a = *in[0];
  const auto& b = *in[1];
  auto out = make_output(outs[0]);
  const auto sa = row_strides(a.dims), sb = row_strides(b.dims);
  for_each_index(out.dims, [&](auto idx, std::size_t pos) {
    auto start = [&](const Dims& d, const auto& strides) {
      std::size_t offset = 0;
      for (std::size_t j = 0; j + 2 < d.size(); ++j)
        if (d[j] > 1) offset += static_cast<std::size_t>(idx[idx.size() - d.size() + j]) * strides[j];
      return offset;
    };
    const auto ia = start(a.dims, sa) + static_cast<std::size_t>(idx[idx.size() - 2]) * sa[sa.size() - 2];
    const auto ib = start(b.dims, sb) + static_cast<std::size_t>(idx.back());
    float sum = 0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(a.dims.back()); ++k)
      sum += a.f[ia + k] * b.f[ib + k * sb[sb.size() - 2]];
    out.f[pos] = sum;
  });
  return std::vector<Value>{std::move(out)};
}
Inferred infer_norm(const Node& n, std::span<const Operand> in) {
  if (auto ok = expect_float(n, in[0], "l’entrée"); !ok) return std::unexpected(ok.error());
  auto a = attrs_of<NormAttrs>(n);
  if (!a) return std::unexpected(a.error());
  auto d = *in[0].dims;
  auto axis = normalize_axis(n, (*a)->axis, d.size());
  if (!axis) return std::unexpected(axis.error());
  if (n.op == OpKind::InstanceNorm && d.size() < 3)
    return shape_error(n, "InstanceNormalization exige un rang de trois ou plus");
  const auto suffix = n.op == OpKind::InstanceNorm
                          ? Dims{d[1]}
                          : Dims(d.begin() + static_cast<std::ptrdiff_t>(*axis), d.end());
  for (std::size_t j = 1; j < in.size(); ++j)
    if (in[j].type != in[0].type || *in[j].dims != suffix)
      return unsupported_error(n, "échelle ou biais de normalisation incompatible");
  return std::vector<OutputInfo>{{in[0].type, std::move(d)}};
}
Evaluated eval_norm(const Node& n, std::span<const Value* const> in, std::span<const OutputInfo> outs) {
  const auto& x = *in[0];
  auto out = make_output(outs[0]);
  if (out.f.empty()) return std::vector<Value>{std::move(out)};
  const auto attr = std::get<NormAttrs>(n.attrs);
  const auto axis =
      n.op == OpKind::InstanceNorm ? std::size_t{2} : *normalize_axis(n, attr.axis, x.dims.size());
  const auto steps = row_strides(x.dims);
  const auto block = steps[axis] * static_cast<std::size_t>(x.dims[axis]);
  const auto inner = n.op == OpKind::Softmax ? steps[axis] : std::size_t{1};
  const auto count = block / inner;
  for (std::size_t base = 0; base < x.f.size(); base += block)
    for (std::size_t j = 0; j < inner; ++j) {
      const auto at = [&](std::size_t k) { return base + j + k * inner; };
      if (n.op == OpKind::Softmax) {
        float high = -std::numeric_limits<float>::infinity(), sum = 0;
        for (std::size_t k = 0; k < count; ++k) high = std::max(high, x.f[at(k)]);
        for (std::size_t k = 0; k < count; ++k) sum += (out.f[at(k)] = std::exp(x.f[at(k)] - high));
        for (std::size_t k = 0; k < count; ++k) out.f[at(k)] /= sum;
      } else {
        float mean = 0, var = 0;
        for (std::size_t k = 0; k < count; ++k) mean += x.f[at(k)];
        mean /= static_cast<float>(count);
        for (std::size_t k = 0; k < count; ++k) {
          const auto diff = x.f[at(k)] - mean;
          var += diff * diff;
        }
        auto inv = 1.0F / std::sqrt(var / static_cast<float>(count) + attr.epsilon);
        for (std::size_t k = 0; k < count; ++k) {
          auto param =
              n.op == OpKind::InstanceNorm ? (base / block) % static_cast<std::size_t>(x.dims[1]) : k;
          out.f[at(k)] = (x.f[at(k)] - mean) * inv * in[1]->f[param] + (in.size() > 2 ? in[2]->f[param] : 0);
        }
      }
    }
  return std::vector<Value>{std::move(out)};
}
}  // namespace kirari::gpu::ir::detail
