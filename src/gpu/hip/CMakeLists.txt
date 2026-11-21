if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Le backend HIP est disponible sur Linux et Windows")
endif()

set(hip_rocm_default "/opt/rocm")
if(DEFINED ENV{HIP_PATH})
  set(hip_rocm_default "$ENV{HIP_PATH}")
endif()
set(KIRARI_HIP_ROCM_PREFIX "${hip_rocm_default}" CACHE PATH "Distribution ROCm pour compiler les noyaux HIP")
set(KIRARI_HIP_CODE_DIRECTORY "" CACHE PATH "Dossier facultatif de HSACO SPAN précompilés")
set(KIRARI_HIP_ARCHITECTURES "gfx1152" CACHE STRING "Cibles RDNA 3/3.5 pour le noyau SPAN HIP")
set(KIRARI_HIP_CUDA_ARCHITECTURES "" CACHE STRING "Cibles NVIDIA facultatives pour le SPAN HIP, par exemple sm89")
set(KIRARI_HIP_CUDA_CODE_DIRECTORY "" CACHE PATH "Dossier facultatif de CUBIN SPAN HIP précompilés")
set(hip_cuda_default "/opt/cuda")
if(DEFINED ENV{CUDA_PATH})
  set(hip_cuda_default "$ENV{CUDA_PATH}")
endif()
set(KIRARI_HIP_CUDA_ROOT "${hip_cuda_default}" CACHE PATH "Préfixe du toolkit CUDA utilisé par HIPCC")
if(NOT KIRARI_HIP_ARCHITECTURES AND NOT KIRARI_HIP_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Le backend HIP requiert au moins une cible AMD ou NVIDIA")
endif()
if(KIRARI_HIP_ARCHITECTURES OR (KIRARI_HIP_CUDA_ARCHITECTURES AND NOT KIRARI_HIP_CUDA_CODE_DIRECTORY))
  find_path(KIRARI_HIP_INCLUDE hip/hip_runtime_api.h HINTS "${KIRARI_HIP_ROCM_PREFIX}/include" REQUIRED)
endif()
if(KIRARI_HIP_ARCHITECTURES AND NOT KIRARI_HIP_CODE_DIRECTORY)
  find_program(KIRARI_HIP_CLANG NAMES amdclang++ clang++ PATHS
    "${KIRARI_HIP_ROCM_PREFIX}/bin" "${KIRARI_HIP_ROCM_PREFIX}/lib/llvm/bin" NO_DEFAULT_PATH REQUIRED)
  find_program(KIRARI_HIP_BUNDLER NAMES clang-offload-bundler HINTS
    "${KIRARI_HIP_ROCM_PREFIX}/lib/llvm/bin" "${KIRARI_HIP_ROCM_PREFIX}/bin" REQUIRED)
endif()

add_library(kirari_backend_hip MODULE entry.cpp)
if(KIRARI_HIP_ARCHITECTURES)
  target_sources(kirari_backend_hip PRIVATE session.cpp)
  target_include_directories(kirari_backend_hip SYSTEM PRIVATE "${KIRARI_HIP_INCLUDE}")
  target_compile_definitions(kirari_backend_hip PRIVATE KIRARI_HIP_ROCM __HIP_PLATFORM_AMD__)
endif()
target_link_libraries(kirari_backend_hip PRIVATE
  kirari::core_log kirari::core_os kirari::vk_backend kirari::gpu_span_fp8_onnx kirari::dep::spdlog)
set_target_properties(kirari_backend_hip PROPERTIES PREFIX ""
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(WIN32)
  target_compile_definitions(kirari_backend_hip PRIVATE KIRARI_HIP_HOST_STAGING)
else()
  target_link_options(kirari_backend_hip PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
kirari_set_warnings(kirari_backend_hip)

function(kirari_hip_asset binary)
  get_filename_component(filename "${binary}" NAME)
  target_sources(kirari_backend_hip PRIVATE "${binary}")
  set_property(TARGET kirari_backend_hip APPEND PROPERTY LINK_DEPENDS "${binary}")
  add_custom_command(TARGET kirari_backend_hip POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:kirari_backend_hip>/hip"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${binary}"
      "$<TARGET_FILE_DIR:kirari_backend_hip>/hip/${filename}"
    VERBATIM)
  install(FILES "${binary}" DESTINATION hip COMPONENT hip)
endfunction()

foreach(arch IN LISTS KIRARI_HIP_ARCHITECTURES)
  if(NOT arch MATCHES "^gfx11(0[0-3]|5[0-3])$")
    message(FATAL_ERROR "Cible HIP sans fragment WMMA RDNA 3/3.5 pris en charge : ${arch}")
  endif()
  if(KIRARI_HIP_CODE_DIRECTORY)
    set(binary "${KIRARI_HIP_CODE_DIRECTORY}/span-${arch}.hsaco")
    if(NOT EXISTS "${binary}")
      message(FATAL_ERROR "Binaire HIP précompilé absent : ${binary}")
    endif()
  else()
    set(binary "${CMAKE_CURRENT_BINARY_DIR}/span-${arch}.hsaco")
    set(bundle "${CMAKE_CURRENT_BINARY_DIR}/span-${arch}.bundle")
    add_custom_command(OUTPUT "${binary}" BYPRODUCTS "${bundle}"
      COMMAND "${KIRARI_HIP_CLANG}" -x hip -std=c++20 -O3 "--offload-arch=${arch}" --offload-device-only
        "--rocm-path=${KIRARI_HIP_ROCM_PREFIX}" -mno-wavefrontsize64
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip" -o "${bundle}"
      COMMAND "${KIRARI_HIP_BUNDLER}" -type=o -unbundle "-targets=hipv4-amdgcn-amd-amdhsa--${arch}"
        "-input=${bundle}" "-output=${binary}"
      DEPENDS kernels.hip geometry.h matrix.h
      VERBATIM COMMENT "Compilation du réseau SPAN HIP pour ${arch}")
  endif()
  kirari_hip_asset("${binary}")
endforeach()

if(KIRARI_HIP_CUDA_ARCHITECTURES)
  find_path(KIRARI_HIP_CUDA_INCLUDE cuda.h HINTS "${KIRARI_HIP_CUDA_ROOT}/include" "$ENV{CUDA_PATH}/include" REQUIRED)
  target_sources(kirari_backend_hip PRIVATE cuda.cpp)
  target_include_directories(kirari_backend_hip SYSTEM PRIVATE "${KIRARI_HIP_CUDA_INCLUDE}")
  target_compile_definitions(kirari_backend_hip PRIVATE KIRARI_HIP_CUDA)
  target_link_libraries(kirari_backend_hip PRIVATE kirari::cuda_driver)
  if(NOT KIRARI_HIP_CUDA_CODE_DIRECTORY)
    find_program(KIRARI_HIP_HIPCC hipcc PATHS "${KIRARI_HIP_ROCM_PREFIX}/bin" NO_DEFAULT_PATH REQUIRED)
  endif()
  foreach(arch IN LISTS KIRARI_HIP_CUDA_ARCHITECTURES)
    if(NOT arch MATCHES "^sm[0-9][0-9][0-9]?$")
      message(FATAL_ERROR "Cible NVIDIA HIP invalide : ${arch}")
    endif()
    string(SUBSTRING "${arch}" 2 -1 compute_capability)
    if(KIRARI_HIP_CUDA_CODE_DIRECTORY)
      set(binary "${KIRARI_HIP_CUDA_CODE_DIRECTORY}/span-${arch}.cubin")
      if(NOT EXISTS "${binary}")
        message(FATAL_ERROR "CUBIN HIP précompilé absent : ${binary}")
      endif()
    else()
      set(binary "${CMAKE_CURRENT_BINARY_DIR}/span-${arch}.cubin")
      add_custom_command(OUTPUT "${binary}"
        COMMAND "${CMAKE_COMMAND}" -E env "HIP_PLATFORM=nvidia" "CUDA_PATH=${KIRARI_HIP_CUDA_ROOT}"
          "HIP_PATH=${KIRARI_HIP_ROCM_PREFIX}" "ROCM_PATH=${KIRARI_HIP_ROCM_PREFIX}"
          "${KIRARI_HIP_HIPCC}" -D__HIP_PLATFORM_NVIDIA__
          -isystem "${KIRARI_HIP_INCLUDE}" -x cu "-arch=sm_${compute_capability}" --cubin -O3 -std=c++20
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip" -o "${binary}"
        DEPENDS kernels.hip geometry.h matrix.h
        VERBATIM COMMENT "Compilation du réseau SPAN HIP pour ${arch}")
    endif()
    kirari_hip_asset("${binary}")
  endforeach()
endif()

install(TARGETS kirari_backend_hip LIBRARY DESTINATION . COMPONENT hip RUNTIME DESTINATION . COMPONENT hip)
