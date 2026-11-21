#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace kirari::gpu::zml {
struct SpanGateProfile {
  std::string_view module;
  std::string_view location;
  std::string_view option;
  int channels;
  std::size_t blocks;
  std::array<std::string_view, 2> weights;
};

// Les deux empreintes de chaque famille correspondent au modèle normal et à Sharp1.
inline constexpr std::array kSpanGateProfiles = {
    SpanGateProfile{"kirari_span_balanced_gate8",
                    "Kirari_SPAN_Balanced_porte_",
                    "--span-balanced-triton-gates",
                    64,
                    8,
                    {"b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c",
                     "dde763c8202cfebac5ec030ea6644f3ee339ea35bbda8a76e257405399552f0f"}},
    SpanGateProfile{"kirari_span_performance_gate5",
                    "Kirari_SPAN_Performance_porte_",
                    "--span-performance-triton-gates",
                    48,
                    5,
                    {"7b804d87320f37b7269d6e27796046fbd975e95e2aa1f65d5557f52a36282716",
                     "72246eded3aab52f8d1a71613c0a1d900734b358d1820b62eb36f1ed376e1cf4"}},
};
inline constexpr std::string_view kSpanProjectionModule = "kirari_span_balanced_gate8_projection";
inline constexpr std::string_view kSpanProjectionLocation = "Kirari_SPAN_Balanced_projection";

inline std::string_view span_gate_module(std::string_view mlir) {
  constexpr std::string_view declaration = "module @";
  // Le frontal peut imprimer ses alias de localisation avant le module.
  while (!mlir.empty()) {
    const auto end = mlir.find('\n');
    const auto line = mlir.substr(0, end);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != line.npos) {
      const auto token = line.substr(first);
      if (!token.starts_with('#') && !token.starts_with("//")) {
        if (!token.starts_with(declaration)) return {};
        const auto suffix = token.substr(declaration.size());
        const auto separator = suffix.find(' ');
        if (separator == suffix.npos || !suffix.substr(separator).starts_with(" {")) return {};
        return suffix.substr(0, separator);
      }
    }
    if (end == mlir.npos) break;
    mlir.remove_prefix(end + 1);
  }
  return {};
}

inline const SpanGateProfile* span_gate_profile(std::string_view module) {
  if (module == kSpanProjectionModule) return &kSpanGateProfiles.front();
  auto found = std::ranges::find(kSpanGateProfiles, module, &SpanGateProfile::module);
  return found == kSpanGateProfiles.end() ? nullptr : &*found;
}
}  // namespace kirari::gpu::zml
