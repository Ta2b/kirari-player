#!/usr/bin/env bash
# Assemble le runtime PJRT distribué par ZML et l'en-tête C associé pour le build de Kirari.
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage : $0 {headers|cuda|metal|oneapi|rocm} DOSSIER" >&2
  exit 2
fi
kirari_root=$(cd "$(dirname "$0")/../.." && pwd)
platform=$1
case "$platform/$(uname -s)/$(uname -m)" in
  headers/*) ;;
  cuda/Linux/x86_64) repo=libzml_cuda_linux_amd64 ;;
  cuda/Linux/aarch64) repo=libzml_cuda_linux_arm64 ;;
  metal/Darwin/arm64) repo=libzml_metal ;;
  oneapi/Linux/x86_64) repo=libzml_oneapi ;;
  rocm/Linux/x86_64) repo=libzml_rocm ;;
  *) echo "ZML ne fournit pas ce runtime pour cet hôte : $platform/$(uname -s)/$(uname -m)" >&2; exit 2 ;;
esac
mkdir -p "$2"
dest=$(cd "$2" && pwd)
plugin=libzml_${platform}.so
if [[ $platform == metal ]]; then plugin=libzml_metal.dylib; fi
if [[ -e "$dest/runtimes/$platform" && ! -f "$dest/runtimes/$platform/lib/$plugin" ]]; then
  echo "Le dossier existant ne contient pas le greffon attendu : $dest/runtimes/$platform/lib/$plugin" >&2
  exit 2
fi
zml_rev=f430c3530fbc6d99f734b641eb8f3fd699868843
xla_rev=6f094faf0fadcd1677e474c06b740876525c99fc
staging=$(mktemp -d "$dest/.preparation.XXXXXX")
installed_frontend="$dest/runtimes/frontend"
cleanup() {
  if [[ -e "$staging/ancien-frontal" && ! -e "$installed_frontend" ]]; then
    mv "$staging/ancien-frontal" "$installed_frontend" || return
  fi
  rm -rf "$staging"
}
trap cleanup EXIT
header() {
  local path=$1 hash=$2
  mkdir -p "$staging/include/$(dirname "$path")"
  curl --fail --location --retry 3 --silent --show-error \
    "https://raw.githubusercontent.com/openxla/xla/$xla_rev/$path" -o "$staging/include/$path"
  (cd "$staging/include" && echo "$hash  $path" | shasum -a 256 -c -)
}
header xla/pjrt/c/pjrt_c_api.h ddedb2e7b4df1d19a8d15c23e41298a3f41d48076707bd563d672f6a95736213
if [[ $platform == cuda || $platform == headers ]]; then
  header xla/pjrt/c/pjrt_c_api_xla_transform_extension.h 51fe2213ba43d80c183edb18ab688d6b82a7a14c0f5a14a72eeffbaa019712c2
  header xla/pjrt/c/pjrt_c_api_stream_extension.h 4c1c8e9ced65324f4c54a67a75c5883d60b1f91d8d8b4b12006f1ed061b8f9d2
fi
if [[ $platform == headers ]]; then
  mkdir -p "$dest/include"
  cp -R "$staging/include/." "$dest/include/"
  echo "KIRARI_ZML_XLA_INCLUDE_DIR=$dest/include"
  exit 0
fi
frontend_id=$(
  { printf '%s\n' "$zml_rev"; uname -sm; (cd "$kirari_root/src/gpu/zml/sdk" && shasum -a 256 BUILD.bazel *.zig); } |
    shasum -a 256 | cut -d' ' -f1
)
git init -q "$staging/zml"
git -C "$staging/zml" fetch --quiet --depth=1 https://github.com/zml/zml.git "$zml_rev"
git -C "$staging/zml" checkout --quiet --detach FETCH_HEAD
cd "$staging/zml"
bazel=${BAZEL:-bazel}
# Les cibles « sandbox » assemblent les dépendances d'exécution et réécrivent leurs chemins de chargement.
assemble() {
  local target=$1 result
  "$bazel" build "$target"
  result=$("$bazel" cquery "$target" --output=files)
  [[ -d "$result" ]] || { echo "Sortie de Bazel inattendue : $result" >&2; return 1; }
  cp -RL "$result" "$2"
}
if [[ ! -e "$dest/runtimes/$platform" ]]; then
  assemble "@$repo//:sandbox" "$staging/$platform"
  if [[ $platform == metal ]]; then
    assemble '//platforms/metal:metal_toolchain' "$staging/metal/toolchain"
    test -x "$staging/metal/toolchain/bin/metal"
  fi
elif [[ $platform == metal && ! -x "$dest/runtimes/metal/toolchain/bin/metal" ]]; then
  echo "Le runtime Metal existant a perdu son compilateur : $dest/runtimes/metal/toolchain/bin/metal" >&2
  exit 2
fi
if [[ ! -x "$installed_frontend/span_frontend" || ! -d "$installed_frontend/span_frontend.runfiles" ||
      ! -f "$installed_frontend/source.sha256" || $(cat "$installed_frontend/source.sha256") != "$frontend_id" ]]; then
  mkdir "$staging/zml/kirari"
  cp "$kirari_root/src/gpu/zml/sdk/BUILD.bazel" "$kirari_root"/src/gpu/zml/sdk/*.zig "$staging/zml/kirari/"
  "$bazel" build --config=release //kirari:span_frontend
  frontend=$("$bazel" cquery --config=release //kirari:span_frontend --output=files)
  test -x "$frontend"
  test -d "$frontend.runfiles"
  mkdir "$staging/frontend"
  cp "$frontend" "$staging/frontend/span_frontend"
  cp -RL "$frontend.runfiles" "$staging/frontend/span_frontend.runfiles"
  echo "$frontend_id" > "$staging/frontend/source.sha256"
fi
mkdir -p "$dest/include" "$dest/runtimes"
if [[ -d "$staging/frontend" ]]; then
  if [[ -e "$installed_frontend" ]]; then mv "$installed_frontend" "$staging/ancien-frontal"; fi
  mv "$staging/frontend" "$installed_frontend"
fi
cp -R "$staging/include/." "$dest/include/"
if [[ -d "$staging/$platform" ]]; then mv "$staging/$platform" "$dest/runtimes/"; fi
echo "KIRARI_ZML_XLA_INCLUDE_DIR=$dest/include"
echo "KIRARI_ZML_BUNDLE=$dest/runtimes"
