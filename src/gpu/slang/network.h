#pragma once
#include "plan.h"
#include "vulkan.h"

namespace c64::balanced {
enum class Endpoints { Imported, Host };

class Network {
  Device& gpu;
  VkDevice d;
  Plan& plan;
  Buffer data, staging;
  VkDeviceSize device_size{}, staging_size{};
  VkDescriptorSetLayout sets_layout{};
  VkPipelineLayout layout{};
  VkDescriptorPool pool{};
  VkCommandPool commands{};
  VkCommandBuffer cmd{};
  VkQueryPool queries{};
  U n;
  bool per_stage;
  std::array<U, 2> extent;

  U timestamp_stride() const { return per_stage ? static_cast<U>(plan.launches.size() + 1) : 2; }

  void buffers(Endpoints endpoints, const fs::path& input) {
    const auto align = std::max<VkDeviceSize>(4, gpu.props.limits.minStorageBufferOffsetAlignment);
    for (auto& [name, v] : plan.views) {
      require(v.bytes <= gpu.props.limits.maxStorageBufferRange, "vue dépassant maxStorageBufferRange");
      // En lecture X et Y sont liés directement aux tampons Vulkan importés.
      if (endpoints == Endpoints::Imported && (name == "X" || name == "Y")) continue;
      v.offset = (device_size + align - 1) / align * align;
      device_size = v.offset + v.bytes;
      if (name == "X") v.file = input;
      if (name == "Y") v.file.clear();
      if (!v.file.empty() || name == "X" || name == "Y") {
        v.host = staging_size;
        staging_size += v.bytes;
      }
    }
    data.create(gpu, device_size, false);
    staging.create(gpu, staging_size, true);
    void* mapping{};
    ck(vkMapMemory(d, staging.memory, 0, staging_size, 0, &mapping));
    try {
      for (const auto& [_, v] : plan.views)
        if (!v.file.empty()) {
          require(fs::file_size(v.file) == v.bytes, "fichier de tenseur de taille incompatible");
          std::ifstream in(v.file, std::ios::binary);
          in.read(static_cast<char*>(mapping) + v.host, static_cast<std::streamsize>(v.bytes));
          require(in.good(), "lecture incomplète du tenseur");
        }
    } catch (...) {
      vkUnmapMemory(d, staging.memory);
      throw;
    }
    vkUnmapMemory(d, staging.memory);
  }

  void layouts() {
    require(gpu.props.limits.maxPerStageDescriptorStorageBuffers >= 8 &&
                gpu.props.limits.maxDescriptorSetStorageBuffers >= 8,
            "huit liaisons storage requises");
    std::array<VkDescriptorSetLayoutBinding, 8> bindings;
    for (U i = 0; i < bindings.size(); ++i)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = static_cast<U>(bindings.size());
    li.pBindings = bindings.data();
    ck(vkCreateDescriptorSetLayout(d, &li, nullptr, &sets_layout));
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &sets_layout;
    pi.pushConstantRangeCount = 1;
    pi.pPushConstantRanges = &range;
    ck(vkCreatePipelineLayout(d, &pi, nullptr, &layout));
    const auto sets = static_cast<U>(plan.launches.size());
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * sets};
    VkDescriptorPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_ci.maxSets = sets;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &size;
    ck(vkCreateDescriptorPool(d, &pool_ci, nullptr, &pool));
  }

  void pipeline(Kernel& k) {
    require(k.x <= gpu.props.limits.maxComputeWorkGroupCount[0] &&
                k.y <= gpu.props.limits.maxComputeWorkGroupCount[1],
            "grille Vulkan hors limites");
    const auto bytes = read(k.file);
    require(bytes.size() % 4 == 0, "module SPIR-V tronqué");
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = bytes.size();
    si.pCode = reinterpret_cast<const U*>(bytes.data());
    VkShaderModule shader{};
    ck(vkCreateShaderModule(d, &si, nullptr, &shader));
    VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pc.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.stage.module = shader;
    pc.stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT wave{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
    wave.requiredSubgroupSize = 32;
    if (gpu.force_wave32) pc.stage.pNext = &wave;
    pc.layout = layout;
    const auto rc = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &pc, nullptr, &k.pipeline);
    vkDestroyShaderModule(d, shader, nullptr);
    ck(rc);
  }

  VkDescriptorSet descriptors(const Launch& l, VkDescriptorPool target, VkBuffer x = {}, VkBuffer y = {}) {
    VkDescriptorSet set{};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = target;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &sets_layout;
    ck(vkAllocateDescriptorSets(d, &ai, &set));
    std::array<VkDescriptorBufferInfo, 8> info;
    std::array<VkWriteDescriptorSet, 8> write;
    // Les liaisons non utilisées par ce shader pointent aussi sur une vraie vue.
    for (U i = 0; i < info.size(); ++i) {
      const auto& name = i < l.args.size() ? l.args[i] : l.args.front();
      const auto& v = plan.views.at(name);
      const auto external = name == "X" ? x : name == "Y" ? y : VK_NULL_HANDLE;
      info[i] = {external ? external : data.buffer, external ? 0 : v.offset, v.bytes};
      write[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set,      i,      0, 1,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      nullptr, &info[i], nullptr};
    }
    vkUpdateDescriptorSets(d, static_cast<U>(write.size()), write.data(), 0, nullptr);
    return set;
  }

  void command_resources() {
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pc.queueFamilyIndex = gpu.family;
    ck(vkCreateCommandPool(d, &pc, nullptr, &commands));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commands;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    ck(vkAllocateCommandBuffers(d, &ai, &cmd));
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = timestamp_stride() * n;
    ck(vkCreateQueryPool(d, &qi, nullptr, &queries));
  }
  void begin() {
    ck(vkResetCommandPool(d, commands, 0));
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ck(vkBeginCommandBuffer(cmd, &info));
  }
  void submit() {
    ck(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ck(vkQueueSubmit(gpu.queue, 1, &si, VK_NULL_HANDLE));
    ck(vkQueueWaitIdle(gpu.queue));
  }
  void upload(bool external) {
    begin();
    for (const auto& [_, v] : plan.views)
      if (!v.file.empty()) {
        VkBufferCopy copy{v.host, v.offset, v.bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, data.buffer, 1, &copy);
      }
    const auto& y = plan.views.at("Y");
    if (!external) vkCmdFillBuffer(cmd, data.buffer, y.offset, y.bytes, 0x7e007e00);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    submit();
  }
  void dispatch(VkCommandBuffer list, VkDescriptorSet first = {}, VkDescriptorSet last = {},
                U next_timestamp = 0) const {
    vkCmdPushConstants(list, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extent), extent.data());
    for (const auto& l : plan.launches) {
      const auto& k = plan.kernels.at(l.name);
      vkCmdBindPipeline(list, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
      VkDescriptorSet set = l.set;
      if (first && &l == &plan.launches.front()) set = first;
      if (last && &l == &plan.launches.back()) set = last;
      vkCmdBindDescriptorSets(list, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
      vkCmdDispatch(list, k.x, k.y, 1);
      barrier(list, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      if (next_timestamp)
        vkCmdWriteTimestamp(list, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, next_timestamp++);
    }
  }
  void measure(U pass) {
    begin();
    const U stride = timestamp_stride();
    vkCmdResetQueryPool(cmd, queries, 0, stride * n);
    for (U i = 0; i < std::min<U>(n, 4); ++i) dispatch(cmd);
    for (U i = 0; i < n; ++i) {
      const U first = stride * i;
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, first);
      dispatch(cmd, {}, {}, per_stage ? first + 1 : 0);
      if (!per_stage) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, first + 1);
    }
    submit();
    std::vector<std::uint64_t> t(stride * n);
    ck(vkGetQueryPoolResults(d, queries, 0, stride * n, t.size() * sizeof(t[0]), t.data(), sizeof(t[0]),
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const auto mask =
        gpu.timestamp_bits < 64 ? (std::uint64_t{1} << gpu.timestamp_bits) - 1 : ~std::uint64_t{0};
    const double us_per_tick = double(gpu.props.limits.timestampPeriod) / 1000.;
    const auto elapsed = [&](U first, U last) { return double((t[last] - t[first]) & mask) * us_per_tick; };
    for (U i = 0; i < n; ++i) {
      const U first = stride * i;
      std::cout << "gpu_us_" << pass << '\t' << elapsed(first, first + stride - 1) << '\n';
      if (!per_stage) continue;
      for (U stage = 0; stage + 1 < stride; ++stage)
        std::cout << "stage_us\t" << pass << '\t' << i << '\t' << stage << '\t' << plan.launches[stage].name
                  << '\t' << elapsed(first + stage, first + stage + 1) << '\n';
    }
  }
  void download(const fs::path& output) {
    begin();
    const auto& y = plan.views.at("Y");
    barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy copy{y.offset, y.host, y.bytes};
    vkCmdCopyBuffer(cmd, data.buffer, staging.buffer, 1, &copy);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
    submit();
    void* mapping{};
    ck(vkMapMemory(d, staging.memory, 0, staging_size, 0, &mapping));
    std::ofstream out(output, std::ios::binary);
    out.write(static_cast<const char*>(mapping) + y.host, static_cast<std::streamsize>(y.bytes));
    vkUnmapMemory(d, staging.memory);
    out.close();
    require(out.good(), "sortie SPAN incomplète");
  }

 public:
  Network(Device& device, Plan& p, U iterations, bool profile_stages = false,
          std::array<U, 2> dispatch_extent = {960, 540})
      : gpu(device),
        d(device.device),
        plan(p),
        data(d),
        staging(d),
        n(iterations),
        per_stage(profile_stages),
        extent(dispatch_extent) {
    require(extent[0] != 0 && extent[1] != 0, "dimensions Vulkan nulles");
  }
  ~Network() {
    vkDeviceWaitIdle(d);
    vkDestroyQueryPool(d, queries, nullptr);
    vkDestroyCommandPool(d, commands, nullptr);
    for (const auto& [_, k] : plan.kernels) vkDestroyPipeline(d, k.pipeline, nullptr);
    vkDestroyDescriptorPool(d, pool, nullptr);
    vkDestroyPipelineLayout(d, layout, nullptr);
    vkDestroyDescriptorSetLayout(d, sets_layout, nullptr);
  }
  void initialize(Endpoints endpoints = Endpoints::Imported, const fs::path& input = {}) {
    buffers(endpoints, input);
    layouts();
    for (auto& [_, k] : plan.kernels) pipeline(k);
    for (auto& l : plan.launches) {
      if (endpoints == Endpoints::Imported && (&l == &plan.launches.front() || &l == &plan.launches.back()))
        continue;
      l.set = descriptors(l, pool);
    }
    command_resources();
    std::cout << "device\t" << gpu.props.deviceName << "\nlaunches\t" << plan.launches.size()
              << "\nresident_bytes\t" << device_size << "\nstaging_bytes\t" << staging_size << '\n';
    upload(endpoints == Endpoints::Imported);
  }
  void record_shared(VkCommandBuffer list, VkDescriptorPool endpoint_pool, VkBuffer x, VkBuffer y) {
    auto first = descriptors(plan.launches.front(), endpoint_pool, x, y);
    auto last = descriptors(plan.launches.back(), endpoint_pool, x, y);
    dispatch(list, first, last);
  }
  // Le lecteur Windows fournit les deux vues hôte. Seuls X et Y transitent par
  // cette allocation cohérente et le réseau effectue les étapes du plan sur le GPU.
  void execute_host(std::span<const std::byte> input, std::span<std::byte> output) {
    const auto& x = plan.views.at("X");
    const auto& y = plan.views.at("Y");
    require(input.size() == x.bytes && output.size() == y.bytes, "taille des deux vues FP16 invalide");
    void* mapping{};
    ck(vkMapMemory(d, staging.memory, 0, staging_size, 0, &mapping));
    std::memcpy(static_cast<std::byte*>(mapping) + x.host, input.data(), input.size());
    vkUnmapMemory(d, staging.memory);

    begin();
    VkBufferCopy to_device{x.host, x.offset, x.bytes};
    vkCmdCopyBuffer(cmd, staging.buffer, data.buffer, 1, &to_device);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(cmd);
    barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy to_host{y.offset, y.host, y.bytes};
    vkCmdCopyBuffer(cmd, data.buffer, staging.buffer, 1, &to_host);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
    submit();

    ck(vkMapMemory(d, staging.memory, 0, staging_size, 0, &mapping));
    std::memcpy(output.data(), static_cast<const std::byte*>(mapping) + y.host, output.size());
    vkUnmapMemory(d, staging.memory);
  }
  void execute(const fs::path& input, const fs::path& output) {
    require(!input.empty() && !output.empty(), "chemins d'entrée et de sortie requis");
    initialize(Endpoints::Host, input);
    measure(1);
    measure(2);
    download(output);
  }
};

inline int run(char** a, bool profile_stages) {
  auto plan = load_plan(a[2]);
  Device gpu;
  gpu.open(false, true, false);
  require(gpu.subgroup == 32 && gpu.tile_n == 16,
          "ce paquet de shaders SPAN exige un sous-groupe de 32 et une tuile 16x16x16 FP16");
  Network network(gpu, plan, dimension(a[5], 200), profile_stages);
  network.execute(a[3], a[4]);
  return 0;
}
}  // namespace c64::balanced
