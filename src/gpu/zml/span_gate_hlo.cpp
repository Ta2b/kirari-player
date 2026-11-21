#include "gpu/zml/span_gate_hlo.h"

#include <algorithm>
#include <bitset>
#include <optional>
#include <string_view>

#include "gpu/ir/protobuf.h"
#include "gpu/zml/span_gate_profile.h"

namespace kirari::gpu::zml {
namespace {
namespace pb = ir::proto;
using pb::Bytes;
using pb::Writer;
using Result = std::expected<Writer, Refusal>;
constexpr std::string_view kConfig = R"({"fusion_backend_config":{"kind":"__triton_gemm"}})";
constexpr std::size_t kProjectionSlot = 8;

std::optional<Bytes> payload(Bytes source, std::uint32_t key) {
  pb::Reader reader(source);
  for (;;) {
    auto field = reader.next();
    if (!field || !*field) return {};
    if ((**field).number == key && (**field).wire == pb::WireType::Length) return (**field).bytes;
  }
}

std::string_view text(Bytes message, std::uint32_t key) {
  const auto data = payload(message, key);
  if (!data) return {};
  return {reinterpret_cast<const char*>(data->data()), data->size()};
}

bool gate_shape(Bytes proto, int channels) {
  std::vector<std::int64_t> dims;
  std::uint64_t element = 0;
  auto read = pb::for_each_field(proto, [&](const pb::Field& f) -> std::expected<void, Refusal> {
    if (f.number == 2 && f.wire == pb::WireType::Varint) element = f.scalar;
    if (f.number == 3) return pb::append_int64s(f, dims);
    return {};
  });
  return read && element == 10 && dims == std::vector<std::int64_t>{518400, channels};  // F16, NHW × C
}

template <class Rewrite>
Result rewrite_children(Bytes message, std::uint32_t child, Rewrite rewrite) {
  Writer out;
  auto read = pb::for_each_field(message, [&](const pb::Field& field) -> std::expected<void, Refusal> {
    if (field.number == child && field.wire == pb::WireType::Length) return rewrite(out, field);
    out.field(field);
    return {};
  });
  if (!read) return std::unexpected(read.error());
  return out;
}

struct Gates {
  const SpanGateProfile& profile;
  bool projection;
  std::bitset<kProjectionSlot + 1> seen;

  std::expected<std::optional<std::size_t>, Refusal> slot(std::string_view name) const {
    if (projection && name == kSpanProjectionLocation) return kProjectionSlot;
    if (!name.starts_with(profile.location)) return std::nullopt;
    name.remove_prefix(profile.location.size());
    if (name.size() != 1 || name[0] < '1' || name[0] > '8')
      return pb::malformed("identité de porte SPAN invalide");
    const auto index = static_cast<std::size_t>(name[0] - '1');
    if (index >= profile.blocks) return pb::malformed("porte SPAN hors du réseau");
    return index;
  }

  std::expected<void, Refusal> instruction(Writer& out, const pb::Field& field) {
    const auto data = field.bytes;
    const auto meta = payload(data, 7);  // HloInstructionProto.metadata : OpMetadata.op_name
    const auto index = slot(meta ? text(*meta, 2) : std::string_view{});
    if (!index) return std::unexpected(index.error());
    if (!*index || text(data, 2) != "fusion") {
      out.field(field);
      return {};
    }
    const auto shape = payload(data, 3);
    if (seen.test(**index) || text(data, 11) != "kCustom" || !shape || !gate_shape(*shape, profile.channels))
      return pb::malformed("signature ou nombre de portes SPAN incompatible");
    seen.set(**index);
    // Le sélecteur d'émetteur appartient au backend. XLA choisit la tuile et produit le code GPU.
    const auto config = text(data, 43);
    if ((!config.empty() && config != kConfig) || payload(data, 99))
      return pb::malformed("porte SPAN avec une configuration XLA préexistante");
    Writer op;
    auto parsed = pb::for_each_field(data, [&](const pb::Field& f) -> std::expected<void, Refusal> {
      if (f.number != 43) op.field(f);
      return {};
    });
    if (!parsed) return parsed;
    op.string(43, kConfig);
    out.message(field.number, op);
    return {};
  }
};
}  // namespace

std::expected<std::vector<std::byte>, std::string> rewrite_span_triton_gates(Bytes module) {
  const auto name = text(module, 1);
  const auto* profile = span_gate_profile(name);
  if (!profile) return std::vector<std::byte>{};
  const auto entry = text(module, 2);
  Gates gates{*profile, name == kSpanProjectionModule, {}};
  auto result =
      rewrite_children(module, 3, [&](Writer& out, const pb::Field& comp) -> std::expected<void, Refusal> {
        if (text(comp.bytes, 1) != entry) {
          out.field(comp);
          return {};
        }
        auto body = rewrite_children(
            comp.bytes, 2, [&](Writer& dst, const pb::Field& f) { return gates.instruction(dst, f); });
        if (!body) return std::unexpected(body.error());
        out.message(comp.number, *body);
        return {};
      });
  if (!result) return std::unexpected(result.error().detail);
  if (gates.seen.count() != profile->blocks + static_cast<std::size_t>(gates.projection))
    return std::unexpected("nombre de portes SPAN émises incompatible");
  return std::move(*result).take();
}
}  // namespace kirari::gpu::zml
