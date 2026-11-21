#include "gpu/slang/resize4.h"

#include <bit>

namespace kirari::gpu::slang {
namespace {

constexpr std::uint32_t kResize4[] = {
#include "resize4.spv.inc"
};

bool resize4_shape(const std::vector<std::uint32_t>& shape) {
  constexpr auto scale4 = std::bit_cast<std::uint32_t>(4.0F);
  return shape[12] == scale4 && shape[13] == scale4 && shape[0] != 0 && shape[0] <= 65535 && shape[1] != 0 &&
         shape[2] != 0 && shape[3] != 0 && shape[0] == shape[6] && shape[1] == shape[7] &&
         std::uint64_t(shape[2]) * 4 == shape[8] && std::uint64_t(shape[3]) * 4 == shape[9] &&
         shape[4] <= 1 && shape[10] <= 1 && shape[8] / 2 <= 65535;
}

}  // namespace

std::span<const std::uint32_t> resize4_dispatch(vk::net::Dispatch& dispatch, const vk::net::Target& target) {
  if (!target.f16_storage || !target.f16_math || target.max_workgroup_invocations < 128 ||
      dispatch.shader != "net_resize_f16" || dispatch.push.size() != 14 || dispatch.constants.size() != 5)
    return {};
  const auto& constants = dispatch.constants;
  const auto& shape = dispatch.push;
  if (constants[0] != 1 || constants[1] != 1 || constants[2] != 1 || constants[3] != 0 ||
      !resize4_shape(shape))
    return {};
  const auto groups = (std::uint64_t(shape[9]) * shape[1] + 127) / 128;
  if (groups > target.max_workgroup_count_x) return {};
  dispatch.shader = "slang_resize4_f16";
  dispatch.constants.clear();
  dispatch.groups = {static_cast<std::uint32_t>(groups), shape[8] / 2, shape[0]};
  return kResize4;
}

}  // namespace kirari::gpu::slang
