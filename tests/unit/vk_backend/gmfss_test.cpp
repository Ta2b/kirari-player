#include "kernels/ref/gmfss.h"

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "net_rig.h"

using namespace kirari;
using namespace kirari::gpu;
using namespace kirari::gpu::vk::test;

TEST_CASE("GMFSS Vulkan : splat et pull avec collisions, trous et grille décalée", "[vk_backend][gmfss]") {
  TempDir dir("gmfss");
  auto backend = make_backend(dir.path());
  const int w = 20, h = 14, mw = 11, mh = 7;
  std::vector<kernels::PlaneF> image(3, kernels::PlaneF(w, h));
  kernels::ref::SplatMotion motion{
      {kernels::PlaneF(mw, mh), kernels::PlaneF(mw, mh), kernels::PlaneF(mw, mh)}};
  for (int c = 0; c < 3; ++c) {
    const auto channel = static_cast<std::size_t>(c);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        image[channel].at(x, y) = static_cast<float>((x * 7 + y * 13 + c * 11) % 32) / 32;
    for (int y = 0; y < mh; ++y)
      for (int x = 0; x < mw; ++x)
        motion.planes[channel].at(x, y) = static_cast<float>((4 * x + 7 * y + 3 * c) % 13 - 6) / 2;
  }
  auto& ops = *backend->kernels();
  const kernels::MotionGrid grid{.scale = 2, .x = 0.25F, .y = -0.5F};
  for (auto type : {TensorType::F32, TensorType::F16}) {
    if (type == TensorType::F16 && !backend->profile().target.f16_storage) continue;
    auto src = rgb_tensor(*backend, w, h, type), flow = rgb_tensor(*backend, mw, mh, type);
    auto dst = rgb_tensor(*backend, w, h, type, 4), time = rgb_tensor(*backend, 1, 1, type, 1);
    write_tensor(*backend, src->tensor(), planes_bytes(image, type));
    write_tensor(*backend, flow->tensor(), planes_bytes(motion.planes, type));
    for (bool pull : {false, true})
      for (float s : {0.25F, 0.75F}) {
        const std::array jobs{KernelOps::SplatJob{.source = src->tensor(),
                                                  .motion = flow->tensor(),
                                                  .s = s,
                                                  .out = dst->tensor(),
                                                  .grid = grid,
                                                  .coverage = true,
                                                  .pull = pull}};
        require_ok(ops.soft_splat_fill(jobs, time->tensor(), s));
        const auto got =
            planes_from(read_tensor(*backend, dst->tensor(), dense_tensor_bytes(type, dst->tensor().shape)),
                        w, h, 4, type);
        auto want = pull ? kernels::ref::soft_pull_at(image, motion, s, grid)
                         : kernels::ref::soft_splat_at(image, motion, s, grid);
        REQUIRE(want);
        want->planes.push_back(std::move(want->coverage));
        for (std::size_t c = 0; c < 4; ++c)
          for (std::size_t i = 0; i < got[c].area(); ++i)
            REQUIRE(std::abs(want->planes[c].data[i] - got[c].data[i]) <
                    (type == TensorType::F16 ? 0.001F : 0.000015F));
        auto instant =
            planes_from(read_tensor(*backend, time->tensor(), dense_tensor_bytes(type, time->tensor().shape)),
                        1, 1, 1, type);
        CHECK(instant[0].at(0, 0) == s);
      }
  }
}
