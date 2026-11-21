#pragma once

#include <expected>
#include <xla/pjrt/c/pjrt_c_api.h>

#include "core/types.h"

namespace kirari::gpu::zml {
bool has_span_transform(const PJRT_Api& api);
std::expected<void, Refusal> compile_with_span_transform(const PJRT_Api& api, PJRT_Client_Compile_Args& args);
}  // namespace kirari::gpu::zml
