#!/usr/bin/env python3
"""Prépare les noyaux portables avant la construction du lecteur."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import tomllib
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile


HERE = Path(__file__).resolve().parent
CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"
SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_PATH = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(required=False):
    url = os.environ.get("KIRARI_PORTABLE_URL", "")
    sha = os.environ.get("KIRARI_PORTABLE_SHA256", "").lower()
    if not url and not sha and not required:
        return None
    parsed = urlsplit(url)
    if (not parsed.hostname or parsed.scheme != "https" or parsed.username is not None
            or any(character.isspace() for character in url) or not SHA256.fullmatch(sha)):
        raise ValueError("donner ensemble une URL HTTPS et son SHA-256 sur 64 caractères")
    return url, sha


def fetch(url, sha, destination):
    try:
        with urlopen(url, timeout=90) as response, destination.open("wb") as output:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("le téléchargement a quitté HTTPS")
            shutil.copyfileobj(response, output)
    except OSError as error:
        reason = getattr(error, "reason", error)
        raise ValueError(f"téléchargement de {destination.name} impossible ({reason})") from error
    if digest(destination) != sha:
        raise ValueError(f"SHA-256 incorrect pour {destination.name}")


def safe_path(name):
    path = PurePosixPath(name)
    if (not SAFE_PATH.fullmatch(name) or path.as_posix() != name or ".." in path.parts
            or any(part.endswith(".") for part in path.parts)):
        raise ValueError(f"chemin interdit dans une archive : {name!r}")
    return path


def copy_member(stream, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with stream, destination.open("xb") as output:
        shutil.copyfileobj(stream, output)


def unpack_assets(archive, destination):
    names = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for item in bundle:
            path = safe_path(item.name)
            if path.parts[0] not in {"hip", "slang", "triton", "sdk"} or len(path.parts) < 2 or not item.isfile():
                raise ValueError(f"entrée inattendue dans le paquet de moteurs : {item.name!r}")
            key = item.name.casefold()
            if key in names:
                raise ValueError(f"entrée en double dans le paquet de moteurs : {item.name!r}")
            names.add(key)
            copy_member(bundle.extractfile(item), destination / path)


def verify_plan(directory):
    listed = set()
    manifest = directory / "SHA256SUMS"
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"ligne invalide dans {manifest}")
        sha, name = match.groups()
        path = safe_path(name)
        normalized = name.casefold()
        if len(path.parts) != 1 or normalized in listed or digest(directory / path) != sha:
            raise ValueError(f"fichier absent, en double ou altéré dans {manifest} : {name}")
        listed.add(normalized)
        yield directory / path
    if "plan.tsv" not in listed or len(listed) < 2:
        raise ValueError(f"plan ou noyaux non vérifiables dans {manifest}")
    yield manifest


def triton_packages(directory, architecture):
    primary = directory / architecture
    if not (primary / "plan.tsv").is_file():
        raise ValueError(f"paquet Triton manquant pour {architecture}")
    return [primary, *sorted(directory.glob(f"{architecture}-*"))]


def hip_packages(directory, extension):
    return sorted(path for path in directory.glob(f"span-*.{extension}") if path.is_file())


def require_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"fichier requis absent ou vide : {path}")


def validate_plan(directory, prefix=None):
    for _ in verify_plan(directory):
        pass
    with (directory / "plan.tsv").open(encoding="ascii") as plan:
        model = plan.readline().strip().split("\t")[-1]
    if not SHA256.fullmatch(model):
        raise ValueError(f"empreinte modèle absente du plan : {directory}")
    name = f"{prefix}-{model}" if prefix else model
    if directory.name not in {name, prefix, "slang"}:
        raise ValueError(f"dossier de plan inattendu : {directory} (attendu : {name})")


def asset_config(assets, platform):
    hip = assets / "hip"
    for file in (hip / "span-sm89.cubin", assets / "slang/plan.tsv"):
        require_file(file)
    slang = assets / "slang"
    extra_slang = sorted(path for path in slang.iterdir() if path.is_dir())
    triton = assets / "triton"
    sm89 = triton_packages(triton, "sm89")
    architectures = hip_packages(hip, "cubin")
    variables = {
        "KIRARI_WITH_HIP": "ON",
        "KIRARI_HIP_ARCHITECTURES": "",
        "KIRARI_HIP_CODE_DIRECTORY": "",
        "KIRARI_HIP_CUDA_CODE_DIRECTORY": str(hip),
        "KIRARI_HIP_CUDA_ARCHITECTURES": ";".join(file.stem.removeprefix("span-") for file in architectures),
        "KIRARI_SLANG_BUNDLE": str(slang),
        "KIRARI_SLANG_ADDITIONAL_BUNDLES": ";".join(map(str, extra_slang)),
        "KIRARI_TRITON_BUNDLE": ";".join(map(str, sm89)),
        "KIRARI_TRITON_CUDA_ARCH": "sm89",
        "KIRARI_TRITON_HIP_BUNDLE": "",
        "KIRARI_TRITON_XPU_BUNDLE": "",
    }
    plans = [(plan, None) for plan in [slang, *extra_slang]]
    plans.extend((plan, "sm89") for plan in sm89)
    if platform == "linux":
        variables["KIRARI_WITH_ZML"] = "ON"
        headers = assets / "sdk/include"
        amd = hip_packages(hip, "hsaco")
        if amd or (triton / "gfx1152").is_dir():
            require_file(headers / "hip/hip_runtime_api.h")
            require_file(assets / "sdk/licenses/hip.txt")
        if amd:
            variables.update(KIRARI_HIP_CODE_DIRECTORY=str(hip), KIRARI_HIP_INCLUDE=str(headers),
                             KIRARI_HIP_ARCHITECTURES=";".join(file.stem.removeprefix("span-") for file in amd))
        if (triton / "gfx1152").is_dir():
            packages = triton_packages(triton, "gfx1152")
            variables.update(KIRARI_TRITON_HIP_BUNDLE=";".join(map(str, packages)),
                             KIRARI_TRITON_HIP_ARCH="gfx1152", KIRARI_TRITON_HIP_INCLUDE_DIR=str(headers))
            plans.extend((plan, "gfx1152") for plan in packages)
    if (triton / "arc-a770").is_dir():
        headers = assets / "sdk/include"
        require_file(headers / "level_zero/ze_api.h")
        require_file(assets / "sdk/licenses/level_zero.txt")
        packages = triton_packages(triton, "arc-a770")
        variables.update(KIRARI_TRITON_XPU_BUNDLE=";".join(map(str, packages)),
                         KIRARI_TRITON_XPU_INCLUDE_DIR=str(headers))
        plans.extend((plan, "arc-a770") for plan in packages)
    for plan, prefix in plans:
        validate_plan(plan, prefix)
    return variables


def cuda_headers(platform, scratch, include):
    versions = dict(re.findall(r"^(CUDA_REDIST_(?:VERSION|MANIFEST_SHA256))=(\S+)$",
                               (HERE / "versions.env").read_text(encoding="utf-8"), re.M))
    manifest = scratch / "cuda-redistrib.json"
    fetch(f"{CUDA_REDIST}/redistrib_{versions['CUDA_REDIST_VERSION']}.json",
          versions["CUDA_REDIST_MANIFEST_SHA256"], manifest)
    nvidia_platform = {"linux": "linux-x86_64", "windows": "windows-x86_64"}[platform]
    package = json.loads(manifest.read_text(encoding="utf-8"))["cuda_cudart"][nvidia_platform]
    location = package["relative_path"]
    archive = scratch / Path(location).name
    fetch(f"{CUDA_REDIST}/{location}", package["sha256"], archive)
    if platform == "windows":
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                parts = entry.filename.split("/")
                if len(parts) >= 3 and parts[1] == "include" and not entry.is_dir():
                    copy_member(bundle.open(entry), include / safe_path("/".join(parts[2:])))
    else:
        with tarfile.open(archive, "r:xz") as bundle:
            for entry in bundle:
                parts = entry.name.split("/")
                if len(parts) >= 3 and parts[1] == "include" and entry.isfile():
                    copy_member(bundle.extractfile(entry), include / safe_path("/".join(parts[2:])))
    require_file(include / "cuda.h")


def configure(platform, destination, env_file):
    url, sha = source(required=True)
    destination.mkdir(parents=True)
    with TemporaryDirectory(prefix="kirari-portable-") as scratch:
        scratch = Path(scratch)
        archive = scratch / "moteurs.tar.gz"
        fetch(url, sha, archive)
        assets = destination / "assets"
        unpack_assets(archive, assets)
        variables = asset_config(assets, platform)
        cuda = destination / "cuda/include"
        cuda_headers(platform, scratch, cuda)
    variables.update(KIRARI_PORTABLE_REQUIRED="1", KIRARI_HIP_CUDA_INCLUDE=str(cuda),
                     KIRARI_TRITON_CUDA_INCLUDE_DIR=str(cuda))
    if platform == "linux":
        headers = destination / "pjrt"
        subprocess.run(("bash", str(HERE.parent / "deps/zml-runtime.sh"), "headers", str(headers)), check=True)
        variables["KIRARI_ZML_XLA_INCLUDE_DIR"] = str(headers / "include")
        variables["KIRARI_ZML_CUDA_INCLUDE_DIR"] = str(cuda)
    with env_file.open("a", encoding="utf-8") as env:
        for name, value in variables.items():
            env.write(f"{name}={value}\n")


def pack(gpu_build, output, rocm, level_zero):
    root = HERE.parent.parent
    files = {f"sdk/licenses/{name}": root / name for name in ("LICENSE", "THIRD_PARTY_NOTICES.md")}
    hip = gpu_build / "hip/hip"
    require_file(hip / "span-sm89.cubin")
    binaries = hip_packages(hip, "cubin")
    if rocm:
        binaries += hip_packages(hip, "hsaco")
    for binary in binaries:
        files[f"hip/{binary.name}"] = binary
    slang = gpu_build / "slang/slang"
    for plan in [slang, *(path for path in sorted(slang.iterdir()) if (path / "plan.tsv").is_file())]:
        for file in verify_plan(plan):
            files[f"slang/{file.relative_to(slang).as_posix()}"] = file
    triton = gpu_build / "triton/triton"
    targets = ["sm89"]
    if rocm:
        targets.append("gfx1152")
    if level_zero:
        targets.append("arc-a770")
    for target in targets:
        for plan in triton_packages(triton, target):
            for file in verify_plan(plan):
                files[f"triton/{file.relative_to(triton).as_posix()}"] = file
    for sdk, folder in ((rocm, "hip"), (level_zero, "level_zero")):
        if sdk:
            require_file(sdk / "LICENSE")
            files[f"sdk/licenses/{folder}.txt"] = sdk / "LICENSE"
            for header in (sdk / "include" / folder).rglob("*.h"):
                files[f"sdk/{header.relative_to(sdk).as_posix()}"] = header
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, path in sorted(files.items()):
                item = tarfile.TarInfo(safe_path(name).as_posix())
                item.size = path.stat().st_size
                with path.open("rb") as file:
                    archive.addfile(item, file)
    print(f"{digest(output)}  {output}")


def check_stage(stage, platform):
    suffix = {"linux": "so", "windows": "dll"}[platform]
    engines = ("hip", "slang", "triton", "zml") if platform == "linux" else ("hip", "slang", "triton")
    for engine in engines:
        require_file(stage / f"kirari_backend_{engine}.{suffix}")
    for name in ("KIRARI_SLANG_BUNDLE", "KIRARI_SLANG_ADDITIONAL_BUNDLES", "KIRARI_TRITON_BUNDLE",
                 "KIRARI_TRITON_HIP_BUNDLE", "KIRARI_TRITON_XPU_BUNDLE"):
        plans = os.environ.get(name, "").split(";")
        for index, plan in enumerate(filter(None, plans)):
            source_plan = Path(plan)
            if name.startswith("KIRARI_SLANG"):
                folder = "slang" if name == "KIRARI_SLANG_BUNDLE" else f"slang/{source_plan.name}"
            else:
                arch = {"KIRARI_TRITON_BUNDLE": "sm89", "KIRARI_TRITON_HIP_BUNDLE": "gfx1152",
                        "KIRARI_TRITON_XPU_BUNDLE": "arc-a770"}[name]
                folder = f"triton/{source_plan.name if index else arch}"
            staged_manifest = stage / folder / "SHA256SUMS"
            if digest(staged_manifest) != digest(source_plan / "SHA256SUMS"):
                raise ValueError(f"le paquet final ne contient pas le plan attendu : {folder}")
            for _ in verify_plan(staged_manifest.parent):
                pass
    for arch_var, dir_var, ext in (("KIRARI_HIP_ARCHITECTURES", "KIRARI_HIP_CODE_DIRECTORY", "hsaco"),
                                  ("KIRARI_HIP_CUDA_ARCHITECTURES", "KIRARI_HIP_CUDA_CODE_DIRECTORY", "cubin")):
        for arch in filter(None, os.environ.get(arch_var, "").split(";")):
            name = f"span-{arch}.{ext}"
            if digest(stage / "hip" / name) != digest(Path(os.environ[dir_var]) / name):
                raise ValueError(f"le noyau HIP attendu a changé dans le paquet final : {name}")
    if expected_url := os.environ.get("KIRARI_PORTABLE_TENSORRT_URL"):
        with (stage / "engines.lock").open("rb") as index:
            engines = tomllib.load(index)
        entries = [entry for entry in engines.get("engine", ())
                   if entry.get("id") == "tensorrt" and entry.get("platform") == platform]
        if (len(entries) != 1 or entries[0].get("url") != expected_url
                or not SHA256.fullmatch(entries[0].get("sha256", ""))):
            raise ValueError("engines.lock ne référence pas le paquet CUDA natif et TensorRT de ce build")
        library = {"linux": "libnvinfer.so.11", "windows": "nvinfer_11.dll"}[platform]
        if not any(item.get("id") == "tensorrt" and item.get("platform") == platform
                   and item.get("name") == library for item in engines.get("vendor_file", ())):
            raise ValueError(f"engines.lock ne fournit pas la bibliothèque TensorRT {library}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source", help="valide les paramètres URL et SHA-256")
    prepare = commands.add_parser("configure", help="télécharge, vérifie et configure les ressources")
    prepare.add_argument("platform", choices=("linux", "windows"))
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("env_file", type=Path)
    stage = commands.add_parser("check-stage", help="vérifie les moteurs installés avant l'archivage du lecteur")
    stage.add_argument("platform", choices=("linux", "windows"))
    stage.add_argument("stage", type=Path)
    build = commands.add_parser("pack", help="assemble les ressources locales dans une archive reproductible")
    build.add_argument("gpu_build", type=Path, help="dossier src/gpu d'une construction avec les moteurs")
    build.add_argument("output", type=Path)
    build.add_argument("--rocm", type=Path, help="dossier des en-têtes HIP, avec include/hip et LICENSE")
    build.add_argument("--level-zero", type=Path, help="dossier des en-têtes Level Zero, avec include/level_zero et LICENSE")
    args = parser.parse_args()
    try:
        if args.command == "source":
            source()
        elif args.command == "configure":
            configure(args.platform, args.destination.resolve(), args.env_file)
        elif args.command == "check-stage":
            check_stage(args.stage, args.platform)
        elif args.command == "pack":
            pack(args.gpu_build, args.output, args.rocm, args.level_zero)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Moteurs portables : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
