#pragma once

#include <charconv>
#include <format>
#include <string_view>

namespace kirari::gpu::zml {
namespace pair12_detail {
inline bool dimension(std::string_view& text, int& value) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || value <= 0 || value % 64 || end == text.data() + text.size() || *end != 'x')
    return false;
  text.remove_prefix(static_cast<std::size_t>(end - text.data()) + 1);
  return true;
}
}  // namespace pair12_detail

// Reconnaissance fermée de la forme canonique produite par le SDK. Le même texte sert à la
// clé du binaire PJRT. Tout emploi supplémentaire de l'argument, toute autre signature ou
// syntaxe ancienne/inconnue conserve le remplissage intégral. Aucune annotation annexe à périmer.
inline bool reads_only_pair12_rgb(std::string_view program) {
  using namespace std::string_view_literals;
  constexpr auto entry = "\nmodule @kirari_motion {\n  func.func @main(%arg0: tensor<1x7x"sv;
  const auto start = program.find(entry);
  if (start == program.npos) return false;
  // Une unique fonction : même les corps/arguments d'une deuxième fonction sont hors contrat.
  const auto func = program.find("func.func");
  if (func != start + entry.find("func.func") || program.find("func.func", func + 1) != program.npos)
    return false;
  auto body = program.substr(start + entry.size());
  int height = 0, width = 0;
  if (!pair12_detail::dimension(body, height) || !pair12_detail::dimension(body, width)) return false;
  const auto signature = std::format(
      "f16> loc(unknown), %arg1: tensor<1x7x{}x{}xf16> loc(unknown)) -> tensor<1x3x{}x{}xf16> {{\n", height,
      width, height, width);
  if (!body.starts_with(signature)) return false;
  body.remove_prefix(signature.size());
  constexpr auto arg = "%arg1"sv;
  const auto use = body.find(arg);
  if (use == body.npos || body.find(arg, use + arg.size()) != body.npos) return false;
  const auto previous_line = body.rfind('\n', use);
  const auto line_start = previous_line == body.npos ? 0 : previous_line + 1;
  auto line = body.substr(line_start, use - line_start);
  if (!line.starts_with("    %")) return false;
  line.remove_prefix(5);
  const auto result_end = line.find_first_not_of("0123456789");
  if (result_end == 0 || result_end == line.npos || line.substr(result_end) != " = stablehlo.slice ")
    return false;
  const auto slice =
      std::format("%arg1 [0:1, 3:6, 0:{}, 0:{}] : (tensor<1x7x{}x{}xf16>) -> tensor<1x3x{}x{}xf16>", height,
                  width, height, width, height, width);
  const auto operand = body.substr(use);
  if (!operand.starts_with(slice)) return false;
  const auto tail = operand.substr(slice.size());
  return tail.starts_with("\n") || tail.starts_with(" loc(");
}
}  // namespace kirari::gpu::zml
