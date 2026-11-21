#include "gpu/zml/span_gate_hlo.h"

#include <algorithm>
#include <format>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/protobuf.h"

namespace kirari::gpu::zml {
namespace {
using ir::proto::Writer;
constexpr std::string_view kMarked = "kirari_span_balanced_gate8";
constexpr std::string_view kTriton = R"({"fusion_backend_config":{"kind":"__triton_gemm"}})";
Writer gate(int index, int channels, std::string_view config, std::string_view family = "Balanced",
            bool projection = false) {
  Writer instruction, shape, meta;
  shape.varint(2, 10);
  shape.varint(3, 518400);
  shape.varint(3, static_cast<unsigned>(channels));
  meta.string(2, projection ? "Kirari_SPAN_Balanced_projection"
                            : std::format("Kirari_SPAN_{}_porte_{}", family, index));
  instruction.string(2, "fusion");
  instruction.message(3, shape);
  instruction.message(7, meta);
  instruction.string(11, "kCustom");
  if (!config.empty()) instruction.string(43, config);
  return instruction;
}
Writer module(std::string_view name, int last = 8, int channels = 64, std::string_view config = "",
              int blocks = 8, std::string_view family = "Balanced") {
  Writer mod, entry, nested;
  nested.string(1, "other");
  nested.message(2, gate(8, 32, "other config"));
  mod.string(1, name);
  mod.string(2, "main.11");
  mod.message(3, nested);
  entry.string(1, "main.11");
  for (int i = 1; i < blocks; ++i) entry.message(2, gate(i, channels, "", family));
  entry.message(2, gate(last, channels, config, family));
  if (name == "kirari_span_balanced_gate8_projection" && last == 8)
    entry.message(2, gate(0, channels, "", family, true));
  mod.message(3, entry);
  mod.string(12345, "champ inconnu");
  return mod;
}
}  // namespace

TEST_CASE("XlaTransform ne marque que les huit portes de l’entrée de SPAN Balanced", "[zml][span]") {
  const auto original = module(kMarked);
  const auto rewritten = rewrite_span_triton_gates(original.data());
  REQUIRE(rewritten);
  const std::string_view bytes(reinterpret_cast<const char*>(rewritten->data()), rewritten->size());
  auto tail = bytes;
  for (int i = 0; i < 8; ++i) {
    auto offset = tail.find(kTriton);
    REQUIRE(offset != tail.npos);
    tail.remove_prefix(offset + kTriton.size());
  }
  CHECK_FALSE(tail.contains(kTriton));
  CHECK(bytes.contains("champ inconnu"));
  CHECK(bytes.contains("other config"));
  REQUIRE(rewrite_span_triton_gates(*rewritten) == rewritten);
  CHECK(rewrite_span_triton_gates(module("kirari_span").data())->empty());
  CHECK_FALSE(rewrite_span_triton_gates(module(kMarked, 7).data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(kMarked, 9).data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(kMarked, 8, 48).data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(kMarked, 8, 64, "autre émetteur").data()));
  REQUIRE(rewrite_span_triton_gates(module("kirari_span_balanced_gate8_projection").data()));
  CHECK_FALSE(rewrite_span_triton_gates(module("kirari_span_balanced_gate8_projection", 7).data()));
  constexpr std::string_view performance = "kirari_span_performance_gate5";
  REQUIRE(rewrite_span_triton_gates(module(performance, 5, 48, "", 5, "Performance").data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(performance, 5, 48, "", 5, "Balanced").data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(performance, 6, 48, "", 6, "Performance").data()));
  CHECK_FALSE(rewrite_span_triton_gates(module(performance, 5, 64, "", 5, "Performance").data()));
  auto truncated = std::move(module(kMarked)).take();
  truncated.pop_back();
  CHECK_FALSE(rewrite_span_triton_gates(truncated));
}
}  // namespace kirari::gpu::zml
