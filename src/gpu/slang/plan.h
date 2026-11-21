#pragma once
#include <map>

#include <vulkan/vulkan.h>

#include "data.h"

namespace c64::balanced {
struct View {
  VkDeviceSize bytes{}, offset{}, host{};
  fs::path file;
};
struct Kernel {
  fs::path file;
  U x{}, y{};
  VkPipeline pipeline{};
  U constants{};
};
struct Launch {
  std::string name;
  std::vector<std::string> args;
  VkDescriptorSet set{};
};
struct Plan {
  std::map<std::string, View> views;
  std::map<std::string, Kernel> kernels;
  std::vector<Launch> launches;
};

inline std::size_t span_blocks(std::string_view model) {
  if (model == "b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c" ||
      model == "dde763c8202cfebac5ec030ea6644f3ee339ea35bbda8a76e257405399552f0f")
    return 8;
  if (model == "7b804d87320f37b7269d6e27796046fbd975e95e2aa1f65d5557f52a36282716" ||
      model == "72246eded3aab52f8d1a71613c0a1d900734b358d1820b62eb36f1ed376e1cf4")
    return 5;
  return 0;
}

inline std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> row;
  std::size_t pos = 0;
  while (true) {
    const auto next = line.find('\t', pos);
    row.emplace_back(line.substr(pos, next == line.npos ? next : next - pos));
    if (next == line.npos) return row;
    pos = next + 1;
  }
}

inline Plan read_plan_rows(std::ifstream& in, const fs::path& root, bool metal) {
  Plan plan;
  std::string line;
  while (std::getline(in, line)) {
    if (line.ends_with('\r')) line.pop_back();
    if (line.empty()) continue;
    const auto row = fields(line);
    if (row.size() == 4 && row[0] == "buffer") {
      const auto bytes = dimension(row[2].c_str(), UINT32_MAX);
      require(bytes % 4 == 0, "taille de tampon non alignée");
      require(
          plan.views.emplace(row[1], View{bytes, 0, 0, row[3] == "-" ? fs::path{} : root / row[3]}).second,
          "tampon dupliqué");
    } else if (row.size() == (metal ? 6 : 5) && row[0] == "kernel") {
      Kernel k{root / row[2], dimension(row[3].c_str(), UINT32_MAX), dimension(row[4].c_str(), UINT32_MAX)};
      // Le générateur enregistre le binding des push constants selon le MSL émis.
      if (metal) k.constants = dimension(row[5], 8, 0);
      require(plan.kernels.emplace(row[1], k).second, "noyau dupliqué");
    } else if (row.size() >= 3 && row.size() <= 10 && row[0] == "launch") {
      plan.launches.push_back({row[1], {row.begin() + 2, row.end()}});
    } else
      throw std::runtime_error("enregistrement TSV invalide : " + line);
  }
  require(in.eof(), "lecture incomplète du plan Slang");
  return plan;
}

inline bool expected_steps(const Plan& plan, std::size_t blocks) {
  const bool separate = plan.launches.size() == 4 * blocks + 4 && !plan.kernels.contains("fused_gate");
  const bool fused_gates = plan.kernels.contains("fused_gate") && !plan.kernels.contains("plain") &&
                           !plan.kernels.contains("gate");
  const bool projected = plan.kernels.contains("fused_project");
  const bool fused = fused_gates && ((plan.launches.size() == 3 * blocks + 4 && !projected) ||
                                     (plan.launches.size() == 3 * blocks + 3 && projected &&
                                      !plan.kernels.contains("project") &&
                                      plan.launches[plan.launches.size() - 2].name == "fused_project"));
  return separate || fused;
}

inline void validate_plan(const Plan& plan, std::size_t blocks, bool metal) {
  require(expected_steps(plan, blocks), "étapes de SPAN inattendues");
  const auto& x = plan.views.at("X");
  const auto& y = plan.views.at("Y");
  require(x.bytes == 1920ull * 1080 * 3 * 2 && y.bytes == 3840ull * 2160 * 3 * 2,
          "géométrie X ou Y incompatible");
  if (blocks == 5) {
    for (const auto* name : {"stem", "first", "state", "a", "b", "res"})
      require(plan.views.at(name).bytes == 960ull * 540 * 48 * 2, "activation SPAN C48 non compacte");
  }
  for (const auto& l : plan.launches) {
    require(plan.kernels.contains(l.name), "noyau de l'étape inconnu");
    require(!metal || plan.kernels.at(l.name).constants <= l.args.size(),
            "binding des constantes Metal hors plage");
    for (const auto& a : l.args) require(plan.views.contains(a), "tampon de l'étape inconnu");
  }
  require(plan.launches.front().args.front() == "X" && plan.launches.back().args.back() == "Y",
          "entrées et sorties du plan inattendues");
  for (std::size_t i = 0; i < plan.launches.size(); ++i)
    for (const auto& name : plan.launches[i].args)
      require((name != "X" || i == 0) && (name != "Y" || i + 1 == plan.launches.size()),
              "accès externe dans un dispatch intermédiaire");
}

inline Plan load_plan(fs::path path, bool metal = false, std::string_view expected_model = {}) {
  if (fs::is_directory(path)) path /= "plan.tsv";
  std::ifstream in(path);
  std::string line;
  require(bool(std::getline(in, line)), "plan TSV absent");
  if (line.ends_with('\r')) line.pop_back();
  const auto meta = fields(line);
  require(meta.size() == 4 && meta[0] == (metal ? "span-slang-metal-v1" : "span-slang-v1") &&
              meta[1] == "1920" && meta[2] == "1080" && (expected_model.empty() || meta[3] == expected_model),
          "cible, modèle ou dimension du plan inconnu");
  const auto blocks = span_blocks(meta[3]);
  require(blocks != 0 && (!metal || blocks == 8), "modèle Slang incompatible");
  Plan plan = read_plan_rows(in, path.parent_path(), metal);
  validate_plan(plan, blocks, metal);
  return plan;
}

}  // namespace c64::balanced
