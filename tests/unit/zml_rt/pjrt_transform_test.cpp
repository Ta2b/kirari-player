#include "gpu/zml/pjrt_transform.h"

#include <string_view>

#include <catch2/catch_test_macros.hpp>

#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
#include <xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>
#endif

namespace kirari::gpu::zml {
namespace {
constexpr std::string_view kMarked = "#loc = loc(unknown)\nmodule @kirari_span_balanced_gate8 {";
constexpr std::string_view kPortable = "module @kirari_span_balanced_gate8_portable {";

struct FakeCompiler {
  unsigned compiled{}, destroyed{};
  bool fail{};
  PJRT_Api api{};
  FakeCompiler() {
    api.PJRT_Client_Compile = [](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
      auto* self = reinterpret_cast<FakeCompiler*>(args->client);
      ++self->compiled;
      if (self->fail) return reinterpret_cast<PJRT_Error*>(self);
      args->executable = reinterpret_cast<PJRT_LoadedExecutable*>(self);
      return nullptr;
    };
    api.PJRT_LoadedExecutable_Destroy = [](PJRT_LoadedExecutable_Destroy_Args* args) -> PJRT_Error* {
      ++reinterpret_cast<FakeCompiler*>(args->executable)->destroyed;
      return nullptr;
    };
    api.PJRT_Error_Message = [](PJRT_Error_Message_Args* args) {
      args->message = "refus du compilateur";
      args->message_size = std::string_view(args->message).size();
    };
    api.PJRT_Error_Destroy = [](PJRT_Error_Destroy_Args*) {};
  }
  std::expected<void, Refusal> compile(std::string_view mlir) {
    PJRT_Program program{.struct_size = PJRT_Program_STRUCT_SIZE,
                         .code = const_cast<char*>(mlir.data()),
                         .code_size = mlir.size(),
                         .format = "mlir",
                         .format_size = 4};
    PJRT_Client_Compile_Args args{.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                  .client = reinterpret_cast<PJRT_Client*>(this),
                                  .program = &program};
    return compile_with_span_transform(api, args);
  }
};

TEST_CASE("PJRT accepte le programme portable sur un greffon sans XlaTransform", "[zml][span_gates]") {
  FakeCompiler compiler;
  CHECK_FALSE(has_span_transform(compiler.api));
  CHECK_FALSE(compiler.compile(kMarked));
  CHECK_FALSE(compiler.compile("module @kirari_span_performance_gate5 {"));
  CHECK(compiler.compile(kPortable));
  CHECK(compiler.compile("// module @kirari_span_balanced_gate8 {\nmodule @kirari_motion {"));
  CHECK(compiler.compiled == 2);
}

#if __has_include(<xla/pjrt/c/pjrt_c_api_xla_transform_extension.h>)
TEST_CASE("PJRT retire uniquement sa transformation et refuse une compilation sans confirmation",
          "[zml][span_gates]") {
  FakeCompiler compiler;
  static std::string_view add_name, remove_name;
  add_name = {};
  remove_name = {};
  PJRT_Xla_Transform_Extension ext{
      .base = {.struct_size = PJRT_Xla_Transform_Extension_STRUCT_SIZE,
               .type = PJRT_Extension_Type_XlaTransform},
      .register_xla_transform = [](PJRT_Register_Xla_Transform_Args* args) -> PJRT_Error* {
        add_name = {args->name, args->name_size};
        return nullptr;
      },
      .clear_xla_transform = [](PJRT_Clear_Xla_Transform_Args* args) -> PJRT_Error* {
        remove_name = {args->name, args->name_size};
        args->cleared = true;
        return nullptr;
      }};
  compiler.api.extension_start = &ext.base;
  CHECK(has_span_transform(compiler.api));
  CHECK(compiler.compile(kPortable));
  CHECK(add_name.empty());
  SECTION("le greffon a omis la passe") {
    CHECK_FALSE(compiler.compile(kMarked));
    CHECK(compiler.destroyed == 1);
  }
  SECTION("le compilateur a échoué") {
    compiler.fail = true;
    CHECK_FALSE(compiler.compile(kMarked));
    CHECK(compiler.destroyed == 0);
  }
  CHECK_FALSE(add_name.empty());
  CHECK(remove_name == add_name);
  CHECK(compiler.compiled == 2);
}
#endif
}  // namespace
}  // namespace kirari::gpu::zml
