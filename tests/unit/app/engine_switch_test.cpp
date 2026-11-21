#include "app/play/engine_switch.h"

#include <array>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/backend_library.h"
#include "core/settings.h"
#include "gpu/cuda_mode.h"

using namespace kirari;
using namespace kirari::app::play;

namespace {

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

}  // namespace

TEST_CASE("Le moteur d'une bibliothèque se lit à son nom", "[app][engine_switch]") {
  CHECK(engine_of_library("base/kirari_backend_tensorrt.so") == core::EngineId::TensorRt);
  CHECK(engine_of_library("C:/kirari/kirari_backend_vulkan.dll") == core::EngineId::Vulkan);
  CHECK(engine_of_library("Kirari.app/Contents/PlugIns/kirari_backend_zml.so") == core::EngineId::Zml);
  CHECK(engine_of_library("base/kirari_backend_hip.so") == core::EngineId::Hip);
  CHECK(engine_of_library("C:/kirari/kirari_backend_slang.dll") == core::EngineId::Slang);
  CHECK(engine_of_library("base/kirari_backend_triton.so") == core::EngineId::Triton);
  // MIGraphX et OpenVINO, retirés : une ancienne bibliothèque n'est d'aucun moteur, et reconnue retirée.
  CHECK_FALSE(engine_of_library("build/libkirari_backend_migraphx.so").has_value());
  CHECK(retired_library("build/libkirari_backend_migraphx.so"));
  CHECK(retired_library("base/kirari_backend_openvino.dll"));
  CHECK_FALSE(retired_library("base/kirari_backend_tensorrt.so"));
  CHECK_FALSE(retired_library("base/openvino.dll"));
  CHECK(engine_of_library("Kirari.app/Contents/PlugIns/kirari_backend_coreml.so") == core::EngineId::Metal);
  CHECK_FALSE(engine_of_library("base/libvulkan.so.1").has_value());
}

TEST_CASE("L'onglet propose les moteurs présents, Vulkan d'abord, sans doublon", "[app][engine_switch]") {
  const std::array<std::filesystem::path, 8> candidates{
      "data/backends/tensorrt/10/kirari_backend_tensorrt.so",
      "base/kirari_backend_tensorrt.so",
      "base/kirari_backend_openvino.so",
      "base/kirari_backend_vulkan.so",
      "base/kirari_backend_zml.so",
      "base/kirari_backend_hip.so",
      "base/kirari_backend_slang.so",
      "base/kirari_backend_triton.so"};
  const std::vector<core::EngineId> expected{
      core::EngineId::Vulkan, core::EngineId::TensorRt, core::EngineId::Cuda,  core::EngineId::Zml,
      core::EngineId::Hip,    core::EngineId::Slang,    core::EngineId::Triton};
  CHECK(switchable_engines(candidates) == expected);
}

TEST_CASE("Changer de moteur le choisit, l'installe et le préfère pour SPAN et RIFE",
          "[app][engine_switch]") {
  core::Settings settings;
  settings.first_run_done = true;
  const core::Settings tensorrt = using_engine(settings, core::EngineId::TensorRt);
  CHECK(tensorrt.preferred[usage(core::Usage::Upscale)] == core::EngineId::TensorRt);
  CHECK(tensorrt.preferred[usage(core::Usage::Interpolate)] == core::EngineId::TensorRt);
  CHECK(core::validate_settings(tensorrt).has_value());

  // Revenir à Vulkan ne retire pas TensorRT : il reste proposé à l'onglet.
  const core::Settings back = using_engine(tensorrt, core::EngineId::Vulkan);
  CHECK(back.preferred[usage(core::Usage::Upscale)] == core::EngineId::Vulkan);
  CHECK(back.chosen_engines.size() == 2);
  CHECK(core::validate_settings(back).has_value());
}

TEST_CASE("Les moteurs spécialisés survivent à l'enregistrement des préférences", "[app][engine_switch]") {
  for (const auto id :
       {core::EngineId::Hip, core::EngineId::Slang, core::EngineId::Triton, core::EngineId::Cuda}) {
    const auto settings = using_engine(core::Settings{}, id);
    const auto serialized = core::format_settings(settings);
    REQUIRE(serialized);
    const auto parsed = core::parse_settings(*serialized);
    REQUIRE(parsed);
    CHECK(parsed->preferred[usage(core::Usage::Upscale)] == id);
    CHECK(parsed->preferred[usage(core::Usage::Interpolate)] == id);
  }
}

TEST_CASE("La bibliothèque du moteur préféré s'essaie en premier", "[app][engine_switch]") {
  const std::vector<std::filesystem::path> candidates{
      "base/kirari_backend_tensorrt.so", "base/kirari_backend_migraphx.so", "base/kirari_backend_vulkan.so"};
  core::Settings settings = using_engine(core::Settings{}, core::EngineId::Vulkan);
  const auto vulkan = preferred_first(candidates, settings);
  REQUIRE(vulkan.size() == 3);
  CHECK(vulkan[0].filename() == "kirari_backend_vulkan.so");
  CHECK(vulkan[1].filename() == "kirari_backend_tensorrt.so");
  CHECK(vulkan[2].filename() == "kirari_backend_migraphx.so");

  settings = using_engine(settings, core::EngineId::TensorRt);
  CHECK(preferred_first(candidates, settings) == candidates);
  CHECK(cuda_inference_mode(settings) == gpu::kTensorRtInference);
  CHECK(selected_engine(candidates.front(), settings) == core::EngineId::TensorRt);

  settings = using_engine(settings, core::EngineId::Cuda);
  CHECK(preferred_first(candidates, settings) == candidates);
  CHECK(cuda_inference_mode(settings) == gpu::kCudaNativeInference);
  CHECK(selected_engine(candidates.front(), settings) == core::EngineId::Cuda);
}

TEST_CASE("BackendLibrary : une bibliothèque gardée se rend sans rechargement jusqu'au choix",
          "[app][engines]") {
  const std::filesystem::path path(KIRARI_BACKEND_VULKAN);
  if (path.empty() || !std::filesystem::is_regular_file(path)) SKIP("backend Vulkan non construit");
  auto first = BackendLibrary::open(std::span(&path, 1));
  REQUIRE(first);
  const BackendLibrary* kept = first->get();
  BackendLibrary::keep_open(std::move(*first));
  auto again = BackendLibrary::open(std::span(&path, 1));
  REQUIRE(again);
  CHECK(again->get() == kept);
  again->reset();
  CHECK_FALSE(BackendLibrary::choose({}, {}));  // le choix relâche les bibliothèques gardées
}
