# Le composant zml_cuda_local est exclu des paquets. Détecter aussi une bibliothèque ou un outil
# de NVIDIA rangé dans le mauvais sous-dossier d'un bundle externe, ou copié par le collecteur de DLL.
if(NOT IS_DIRECTORY "${stage}")
  message(FATAL_ERROR "Dossier de publication introuvable : ${stage}")
endif()
file(GLOB_RECURSE files LIST_DIRECTORIES true RELATIVE "${stage}" "${stage}/*")
file(REAL_PATH "${stage}" root)
foreach(file IN LISTS files)
  string(TOLOWER "${file}" lower)
  cmake_path(GET lower FILENAME name)
  if(lower MATCHES "(^|/)zml/cuda(/|$)" OR
     name MATCHES "^(lib)?(cuda|cudnn|cublas|cufft|cupti|curand|cusolver|cusparse|cufile|cudss|cutensor|cuquantum|cuinj|nvrtc|nvjitlink|nvinfer|nvonnxparser|nccl|nvfatbin|nvptxcompiler|nvvm|nvjpeg|nvtoolsext|npp[a-z]*)([._0-9-]|rt|lt|w|64|$)" OR
     name MATCHES "^(libzml_cuda|zml_cuda|ptxas|nvlink|nvcc|tileiras|cutlass_ir|libdevice)([.]|$)")
    message(FATAL_ERROR "Fichier NVIDIA dans le paquet du lecteur : ${file}")
  endif()
  if(lower MATCHES "(^|/)zml/" AND IS_SYMLINK "${stage}/${file}")
    file(READ_SYMLINK "${stage}/${file}" link)
    file(REAL_PATH "${stage}/${file}" resolved)
    cmake_path(IS_PREFIX root "${resolved}" NORMALIZE inside)
    if(IS_ABSOLUTE "${link}" OR NOT inside OR NOT EXISTS "${resolved}")
      message(FATAL_ERROR "Ressource ZML liée hors du paquet : ${file} -> ${link}")
    endif()
  endif()
endforeach()
