// Remplissage des images, poussée pondérée FP32 et normalisation entre la paire et la fusion.
#include <bit>
#include <cmath>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/kernel_dispatch.h"

namespace kirari::gpu::vk {
using detail::tensor_view;
Done VulkanKernelOps::gmfss_frames(const Tensor& a, const Tensor& b, const Tensor& frames) {
  auto av = tensor_view(a, 3, "GMFSS : image A"), bv = tensor_view(b, 3, "GMFSS : image B");
  auto dst = tensor_view(frames, 6, "GMFSS : images remplies");
  if (!av) return std::unexpected(av.error());
  if (!bv) return std::unexpected(bv.error());
  if (!dst) return std::unexpected(dst.error());
  OpKey key;
  key.add(OpCode::GmfssFrames);
  for (const auto* v : {&*av, &*bv, &*dst}) detail::add_view(key, *v);
  auto rec = recording(key, [&](Recorder& r) { return runner_.rife_frames(r, *av, *bv, *dst, 0, 6, true); });
  if (!rec) return std::unexpected(rec.error());
  return submit("images de GMFSS", **rec);
}

Done VulkanKernelOps::soft_splat(const SplatJob& job) {
  auto src = tensor_view(job.source, 0, "GMFSS : source"),
       flow = tensor_view(job.motion, 3, "GMFSS : mouvement");
  if (!src) return std::unexpected(src.error());
  if (!flow) return std::unexpected(flow.error());
  auto dst = tensor_view(job.out, src->planes + (job.coverage ? 1U : 0U), "GMFSS : sortie");
  if (!dst) return std::unexpected(dst.error());
  if (src->width != dst->width || src->height != dst->height || src->precision != dst->precision ||
      src->precision != flow->precision)
    return refuse(RefusalCode::InputMismatch, "GMFSS : tailles ou précisions incompatibles");
  const auto& g = job.grid;
  if (!(job.s >= 0 && job.s <= 1 && g.scale > 0 && std::isfinite(g.scale) && std::isfinite(g.x) &&
        std::isfinite(g.y)))
    return refuse(RefusalCode::FieldOutOfRange, "GMFSS : grille ou instant invalide");
  OpKey key;
  key.add(OpCode::GmfssSplat);
  for (const auto* v : {&*src, &*flow, &*dst}) detail::add_view(key, *v);
  key.add(job.pull);
  for (float v : {g.scale, g.x, g.y}) key.add(std::bit_cast<std::uint32_t>(v));
  const std::array instant{std::bit_cast<std::uint32_t>(job.s)};
  auto rec = recording_with(key, instant, [&](Recorder& r) -> Done {
    const auto channels = job.pull ? 2U : src->planes;
    r.share_scratch(&splat_scratch_);
    auto sums = r.scratch(VkDeviceSize{4} * src->width * src->height * (channels + 1));
    if (!sums) return std::unexpected(sums.error());
    if (auto ok = r.fill(*sums, 0); !ok) return ok;
    auto param = r.parameter(instant[0]);
    if (!param) return std::unexpected(param.error());
    const std::uint32_t nx = detail::groups_for(src->width), ny = detail::groups_for(src->height);
    const std::uint32_t pull = job.pull ? 1U : 0U;
    struct Push {
      std::uint32_t w, h, mw, mh, c, pull;
      float scale, ox, oy;
    };
    const Push push{src->width, src->height, flow->width, flow->height, src->planes, pull, g.scale, g.x, g.y};
    const std::array input{src->buffer, flow->buffer, *sums, *param};
    if (auto ok = detail::run(pipelines_, r, "gmfss_splat", src->precision, {}, input, {},
                              detail::bytes_of(push), {nx, ny, channels + 1});
        !ok)
      return ok;
    const std::array resolve{src->width, src->height, src->planes, pull};
    const std::array output{src->buffer, *sums, dst->buffer, *param};
    return detail::run(pipelines_, r, "gmfss_resolve", src->precision, {}, output, {},
                       detail::bytes_of(resolve), {nx, ny, dst->planes});
  });
  if (!rec) return std::unexpected(rec.error());
  return submit("splat de GMFSS", **rec);
}

Done VulkanKernelOps::soft_splat_fill(std::span<const SplatJob> jobs, const Tensor& time, float t) {
  if (!(t >= 0 && t <= 1)) return refuse(RefusalCode::FieldOutOfRange, "GMFSS : instant hors de [0,1]");
  for (const auto& job : jobs)
    if (auto ok = soft_splat(job); !ok) return ok;
  if (!time.handle) return {};
  auto dst = tensor_view(time, 1, "GMFSS : instant");
  if (!dst) return std::unexpected(dst.error());
  OpKey key;
  key.add(OpCode::GmfssTime);
  detail::add_view(key, *dst);
  const std::array instant{std::bit_cast<std::uint32_t>(t)};
  auto rec = recording_with(key, instant, [&](Recorder& r) -> Done {
    auto param = r.parameter(instant[0]);
    if (!param) return std::unexpected(param.error());
    const std::array buffers{dst->buffer, *param};
    const std::array push{dst->width, dst->height};
    return detail::run(pipelines_, r, "gmfss_time", dst->precision, {}, buffers, {}, detail::bytes_of(push),
                       {detail::groups_for(dst->width), detail::groups_for(dst->height), 1});
  });
  if (!rec) return std::unexpected(rec.error());
  return submit("instant de GMFSS", **rec);
}
}  // namespace kirari::gpu::vk
