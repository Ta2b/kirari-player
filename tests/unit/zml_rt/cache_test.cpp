#include <future>
#include <random>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/cuda_io.h"
#include "gpu/zml/drba_pair12.h"
#include "gpu/zml/pjrt.h"
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
#include "gpu/zml/span_gate_artifact.h"
#endif

namespace kirari::gpu::zml {
namespace {

TEST_CASE("DistilDRBA : écriture réduite certifiée par tous les usages de pair12", "[zml][drba_pair12]") {
  const std::string source =
      "#loc = loc(unknown)\nmodule @kirari_motion {\n"
      "  func.func @main(%arg0: tensor<1x7x768x1280xf16> loc(unknown), %arg1: tensor<1x7x768x1280xf16> "
      "loc(unknown)) -> tensor<1x3x768x1280xf16> {\n"
      "    %2 = stablehlo.slice %arg1 [0:1, 3:6, 0:768, 0:1280] : (tensor<1x7x768x1280xf16>) -> "
      "tensor<1x3x768x1280xf16> loc(#loc)\n"
      "    return %2 : tensor<1x3x768x1280xf16>\n  }\n}\n";
  REQUIRE(reads_only_pair12_rgb(source));
  auto changed = [&](std::string_view from, std::string_view to) {
    auto other = source;
    other.replace(other.find(from), from.size(), to);
    return reads_only_pair12_rgb(other);
  };
  CHECK_FALSE(changed("3:6", "0:3"));
  CHECK_FALSE(changed("0:768,", "0:767,"));
  CHECK_FALSE(changed("%arg1: tensor", "%other: tensor"));
  CHECK_FALSE(changed("7x768x1280xf16", "7x768x1280xf32"));
  CHECK_FALSE(changed("1280] :", "1280:2] :"));
  CHECK_FALSE(
      changed("    return", "    %3 = stablehlo.negate %arg1 : tensor<1x7x768x1280xf16>\n    return"));
  CHECK_FALSE(changed(" = stablehlo.slice ", " = stablehlo.reverse "));
  CHECK_FALSE(changed("    %2 =", "    // %2 ="));
  CHECK_FALSE(changed("    return", "    // func.func @second(...)\n    return"));
  CHECK_FALSE(reads_only_pair12_rgb("old cache"));
}

struct Calls {
  int compiled = 0, loaded = 0;
  std::vector<std::int64_t> protected_inputs;
  mutable int saved = 0;
  bool empty = false;
  std::stop_source* cancel = nullptr;
};

constexpr std::string_view kBinary = "binaire PJRT simulé";
constexpr std::string_view kSource = "stablehlo";
const vk::net::IoNames kNames{{"in"}, {"out"}};

PJRT_Api cache_api() {
  PJRT_Api api{};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.PJRT_Client_Compile = [](PJRT_Client_Compile_Args* a) -> PJRT_Error* {
    auto& calls = *reinterpret_cast<Calls*>(a->client);
    ++calls.compiled;
    if (calls.cancel) calls.cancel->request_stop();
    a->executable = reinterpret_cast<PJRT_LoadedExecutable*>(a->client);
    return nullptr;
  };
  api.PJRT_LoadedExecutable_GetExecutable = [](PJRT_LoadedExecutable_GetExecutable_Args* a) -> PJRT_Error* {
    a->executable = reinterpret_cast<PJRT_Executable*>(a->loaded_executable);
    return nullptr;
  };
  api.PJRT_Executable_NumOutputs = [](PJRT_Executable_NumOutputs_Args* a) -> PJRT_Error* {
    a->num_outputs = 1;
    return nullptr;
  };
  api.PJRT_Executable_Serialize = [](PJRT_Executable_Serialize_Args* a) -> PJRT_Error* {
    const auto& calls = *reinterpret_cast<const Calls*>(a->executable);
    ++calls.saved;
    a->serialized_bytes = kBinary.data();
    a->serialized_bytes_size = calls.empty ? 0 : kBinary.size();
    a->serialized_executable_deleter = [](PJRT_SerializedExecutable*) {};
    return nullptr;
  };
  api.PJRT_Executable_DeserializeAndLoad = [](PJRT_Executable_DeserializeAndLoad_Args* a) -> PJRT_Error* {
    if (std::string_view(a->serialized_executable, a->serialized_executable_size) != kBinary)
      return reinterpret_cast<PJRT_Error*>(a->client);
    ++reinterpret_cast<Calls*>(a->client)->loaded;
    a->loaded_executable = reinterpret_cast<PJRT_LoadedExecutable*>(a->client);
    return nullptr;
  };
  api.PJRT_Error_Destroy = [](PJRT_Error_Destroy_Args*) {};
  api.PJRT_Executable_Destroy = [](PJRT_Executable_Destroy_Args*) -> PJRT_Error* { return nullptr; };
  api.PJRT_LoadedExecutable_Destroy = [](PJRT_LoadedExecutable_Destroy_Args*) -> PJRT_Error* {
    return nullptr;
  };
  api.PJRT_Client_Destroy = [](PJRT_Client_Destroy_Args*) -> PJRT_Error* { return nullptr; };
  return api;
}

std::shared_ptr<Client> client(const PJRT_Api& api, Calls& calls) {
  // Le handle représente la durée de vie de la table. Aucune fonction PJRT n'en vient.
#if defined(_WIN32)
  constexpr auto system_library = "kernel32.dll";
#elif defined(__APPLE__)
  constexpr auto system_library = "libSystem.B.dylib";
#else
  constexpr auto system_library = "libc.so.6";
#endif
  auto lib = DynamicLibrary::open_system(system_library);
  REQUIRE(lib);
  auto c = std::make_shared<Client>(std::move(*lib), api);
  c->handle = reinterpret_cast<PJRT_Client*>(&calls);
  return c;
}

struct Cache {
  std::filesystem::path dir, path;
  Cache() {
    std::random_device random;
    dir = std::filesystem::temp_directory_path() / std::format("kirari-pjrt-{}-{}", random(), random());
    REQUIRE(std::filesystem::create_directory(dir));
    path = dir / "reseau.mlir";
  }
  ~Cache() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

#ifdef KIRARI_ZML_SPAN_TRITON_GATES
TEST_CASE("SPAN conserve un cache portable cohérent si le greffon refuse les portes", "[zml][cache][span]") {
  Cache cache;
  const auto api = cache_api();
  Calls calls;
  auto device = client(api, calls);
  constexpr std::string_view portable = "module @kirari_span {";
  const auto dense = std::as_bytes(std::span(portable));
  const auto tagged = std::format("// kirari-span-portable-v1 {:016x}\nmodule @kirari_span_balanced_gate8 {{",
                                  vk::net::fnv1a(dense));
  const auto candidate = std::as_bytes(std::span(tagged));
  REQUIRE(vk::net::write_file(cache.path, candidate));
  auto sidecar = cache.path;
  sidecar += ".portable";
  Progress progress{};
  SECTION("le secours correspond aux mêmes poids") {
    REQUIRE(vk::net::write_file(sidecar, dense));
    REQUIRE(compile_span_or_portable(device, candidate, kNames, cache.path, &progress));
    REQUIRE(calls.compiled == 1);
    auto active = vk::net::read_file(cache.path);
    REQUIRE(active);
    CHECK(std::ranges::equal(*active, dense));
    CHECK(std::filesystem::exists(executable_cache(cache.path, dense, 0)));
    CHECK_FALSE(std::filesystem::exists(executable_cache(cache.path, candidate, 0)));
    REQUIRE(compile_span_or_portable(device, *active, kNames, cache.path));
    CHECK(calls.compiled == 1);
    CHECK(calls.loaded == 1);
  }
  SECTION("un fichier voisin ancien ne peut pas remplacer le réseau") {
    REQUIRE(vk::net::write_file(sidecar, std::as_bytes(std::span(kSource))));
    auto network = compile_span_or_portable(device, candidate, kNames, cache.path, &progress);
    REQUIRE_FALSE(network);
    CHECK(network.error().code == RefusalCode::SdkRejected);
    CHECK(calls.compiled == 0);
    auto active = vk::net::read_file(cache.path);
    REQUIRE(active);
    CHECK(std::ranges::equal(*active, candidate));
  }
}
#endif

TEST_CASE("PJRT conserve les deux entrées Vulkan quand la sortie reprend le tampon précédent",
          "[zml][interop]") {
  struct Borrowed final : CudaIo {
    std::expected<vk::net::DeviceMemory, Refusal> import(std::uintptr_t, std::size_t, std::size_t) override {
      return vk::net::DeviceMemory{};
    }
    vk::Done attach(int) override { return {}; }
    vk::Done wait(std::uint64_t) override { return {}; }
    vk::Done signal(std::uint64_t) override { return {}; }
    vk::Done drain() override { return {}; }
    std::expected<PJRT_Buffer*, Refusal> view(Client&, const vk::net::HostBinding& b,
                                              PJRT_Buffer_Type) override {
      return reinterpret_cast<PJRT_Buffer*>(b.device);
    }
    vk::Done copy(Client&, PJRT_Buffer*, const vk::net::HostBinding&,
                  std::vector<std::shared_ptr<void>>*) override {
      return {};
    }
  };
  Cache cache;
  auto api = cache_api();
  api.PJRT_LoadedExecutable_Execute = [](PJRT_LoadedExecutable_Execute_Args* a) -> PJRT_Error* {
    auto& calls = *reinterpret_cast<Calls*>(a->executable);
    const auto& opt = *a->options;
    for (std::size_t i = 0; i < opt.num_non_donatable_input_indices; ++i)
      calls.protected_inputs.push_back(opt.non_donatable_input_indices[i]);
    a->output_lists[0][0] = a->argument_lists[0][1];
    return nullptr;
  };
  api.PJRT_Buffer_Destroy = [](PJRT_Buffer_Destroy_Args*) -> PJRT_Error* { return nullptr; };
  api.PJRT_Buffer_ElementType = [](PJRT_Buffer_ElementType_Args* a) -> PJRT_Error* {
    a->type = PJRT_Buffer_Type_F16;
    return nullptr;
  };
  api.PJRT_Buffer_Dimensions = [](PJRT_Buffer_Dimensions_Args* a) -> PJRT_Error* {
    static constexpr std::array<std::int64_t, 4> shape{1, 1, 1, 1};
    a->dims = shape.data();
    a->num_dims = shape.size();
    return nullptr;
  };
  Calls calls;
  auto c = client(api, calls);
  c->cuda = std::make_shared<Borrowed>();
  auto net = compile(c, std::as_bytes(std::span(kSource)), {{"image", "avant"}, {"sortie"}}, cache.path);
  REQUIRE(net);
  std::array<std::uint16_t, 2> data{};
  const auto binding = [](auto& value) {
    return vk::net::HostBinding{
        .bytes = sizeof(value), .shape = {{1, 1, 1, 1}}, .device = reinterpret_cast<std::uintptr_t>(&value)};
  };
  const std::array inputs{binding(data[0]), binding(data[1])};
  REQUIRE((*net)->run_device(inputs, std::span(inputs).last(1), 1, 2));
  CHECK(calls.protected_inputs == std::vector<std::int64_t>{0, 1});
}

TEST_CASE("DistilDRBA : un cache PJRT existant sans preuve garde le remplissage intégral",
          "[zml][cache][drba_pair12]") {
  Cache cache;
  auto api = cache_api();
  Calls calls;
  auto source = std::as_bytes(std::span(kSource));
  const vk::net::IoNames names{{"pair01", "pair12"}, {"out"}};
  const auto binary = executable_cache(cache.path, source, 0);
  REQUIRE(vk::net::write_file(binary, std::as_bytes(std::span(kBinary))));
  auto net = compile(client(api, calls), source, names, cache.path);
  REQUIRE(net);
  CHECK(calls.loaded == 1);
  CHECK_FALSE((*net)->pair12_rgb_only);
}

TEST_CASE("Le client de lecture reprend le PJRT du constructeur", "[zml][cache]") {
  Cache cache;
  auto api = cache_api();
  Calls builder, player;
  auto source = std::as_bytes(std::span(kSource));
  auto binary = executable_cache(cache.path, source, 0);
  Progress progress{};
  {
    auto prepared = compile(client(api, builder), source, kNames, cache.path, &progress);
    REQUIRE(prepared);
  }
  REQUIRE(std::filesystem::file_size(binary) == kBinary.size());
  REQUIRE(builder.compiled == 1);
  REQUIRE(builder.saved == 1);
  auto reader = client(api, player);
  SECTION("même carte et même programme") {
    REQUIRE(compile(reader, source, kNames, cache.path));
    CHECK(player.loaded == 1);
    CHECK(player.compiled == 0);
    CHECK(player.saved == 0);
  }
  SECTION("un autre ordinal ou une autre source utilise une autre clé") {
    reader->device_id = 1;
    REQUIRE(compile(reader, source, kNames, cache.path));
    reader->device_id = 0;
    REQUIRE(compile(reader, source.first(2), kNames, cache.path));
    CHECK(player.loaded == 0);
    CHECK(player.compiled == 2);
    CHECK(player.saved == 2);
  }
  SECTION("le chargement direct répare un ancien binaire illisible") {
    REQUIRE(vk::net::write_file(binary, source));
    REQUIRE(compile(reader, source, kNames, cache.path));
    REQUIRE(compile(reader, source, kNames, cache.path));
    CHECK(player.compiled == 1);
    CHECK(player.saved == 1);
    CHECK(player.loaded == 1);
  }
}

TEST_CASE("Une préparation PJRT interrompue garde la source", "[zml][cache]") {
  Cache cache;
  auto api = cache_api();
  Calls builder;
  auto source = std::as_bytes(std::span(kSource));
  REQUIRE(vk::net::write_file(cache.path, source));
  std::stop_source stop;
  Progress progress{.stop = stop.get_token()};
  RefusalCode reason = RefusalCode::CacheIo;
  SECTION("le JIT synchrone a rendu la main après la demande d'arrêt") {
    builder.cancel = &stop;
    reason = RefusalCode::Cancelled;
  }
  SECTION("le greffon a rendu une sérialisation vide") {
    builder.empty = true;
  }
  auto prepared = compile(client(api, builder), source, kNames, cache.path, &progress);
  REQUIRE_FALSE(prepared);
  CHECK(prepared.error().code == reason);
  CHECK(builder.compiled == 1);
  CHECK(builder.saved == (builder.empty ? 1 : 0));
  CHECK_FALSE(std::filesystem::exists(executable_cache(cache.path, source, 0)));
  auto old = vk::net::read_file(cache.path);
  REQUIRE(old);
  CHECK(std::ranges::equal(*old, source));
}

TEST_CASE("Deux lecteurs d'un cache sans PJRT partagent une compilation", "[zml][cache]") {
  Cache cache;
  auto api = cache_api();
  Calls a, b;
  auto first = client(api, a), second = client(api, b);
  auto run = [&](std::shared_ptr<Client> c) {
    return compile(std::move(c), std::as_bytes(std::span(kSource)), kNames, cache.path).has_value();
  };
  auto pending = std::async(std::launch::async, run, first);
  const bool other = run(second);
  REQUIRE(pending.get());
  REQUIRE(other);
  CHECK(a.compiled + b.compiled == 1);
  CHECK(a.saved + b.saved == 1);
  CHECK(a.loaded + b.loaded == 1);
}
}  // namespace
}  // namespace kirari::gpu::zml
