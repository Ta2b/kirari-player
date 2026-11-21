#!/usr/bin/env python3
"""Compare les voies SPAN FP16 dans une fenêtre réelle, puis juge les exports sans perte."""

from serie import essai_exclusif

import argparse
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banc_resume import visible_stats


ENGINES = {"vulkan": "vulkan", "zml": "zml", "cuda-natif": "tensorrt", "prototype": "tensorrt", "zml-hote": "zml"}
INTEGRATED = ("cuda", "tensorrt", "hip", "slang", "triton")
ENGINES.update({engine: engine for engine in INTEGRATED})
STAGES = {"vulkan": "rôle agrandissement", "cuda-natif": "SPAN FP16 natif", "prototype": "SPAN Balanced FP16 ZML"}
STAGES.update({engine: "rôle agrandissement" for engine in INTEGRATED})
STAGES["cuda"] = STAGES["cuda-natif"]
MODEL = "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"
MODELS = {
    f"2x_AnimeJaNai_HD_V3.1{sharp}_{name}_SPANF3_{blocks}_unshuffle_fp16.onnx"
    for sharp in ("", "Sharp1")
    for name, blocks in (("Balanced", "b8f64"), ("Performance", "b5f48"))
}
PLAY = """[image]
upscale = true
restore_lines = true
span_precision = "balanced"

[interpolation]
factor = 0
auto = false
smooth_motion = false

[grade]
enabled = false

[engines]
fast_precision = false
"""


def configuration(folder, variant):
    folder.mkdir()
    engine = ENGINES[variant]
    engines = json.dumps(list(dict.fromkeys(["vulkan", engine])))
    (folder / "settings.toml").write_text(
        f'version = 1\nfirst_run_done = true\n[engines]\nchosen = {engines}\ninstalled = {engines}\n'
        f'[engines.preferred]\nupscale = "{engine}"\ninterpolate = "{engine}"\n', encoding="utf-8")
    install_dir = json.dumps(str(folder / "moteurs"), ensure_ascii=False)
    (folder / "play.toml").write_text(f"{PLAY}install_dir = {install_dir}\n", encoding="utf-8")


def generation(path):
    try:
        info = path.stat()
        return info.st_ino, info.st_mtime_ns
    except FileNotFoundError:
        return None


def run(command, log, timeout, env=None):
    """Le garde du lecteur interactif peut attendre dans une boîte SDL après un blocage du pilote."""
    cache = (env or {}).get("XDG_CACHE_HOME")
    watch = Path(cache) / "kirari/present-stall.txt" if cache and "play" in command else None
    original = generation(watch) if watch else None
    deadline = time.monotonic() + timeout
    with log.open("wb") as out, subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                                env=env, start_new_session=os.name == "posix") as proc:
        while True:
            remaining = deadline - time.monotonic()
            reason = None
            if remaining <= 0:
                reason = f"délai de {timeout} s dépassé"
            elif watch and (current := generation(watch)) is not None and current != original:
                reason = "blocage de présentation signalé par le lecteur"
            if reason:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.wait()
                raise RuntimeError(f"{reason}, voir {log}")
            try:
                status = proc.wait(timeout=min(remaining, 1.0) if watch else remaining)
                break
            except subprocess.TimeoutExpired:
                continue
    if status:
        raise RuntimeError(f"sortie {status}, voir {log}")


def player_environment(output):
    env = os.environ.copy()
    profile = output / "profil"
    profile.mkdir(exist_ok=True)
    temp = profile / "tmp"
    temp.mkdir(exist_ok=True)
    if env.get("DISPLAY"):
        env.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    env.update(HOME=str(profile), TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp), XDG_DATA_HOME=str(profile / ".local/share"),
               XDG_CONFIG_HOME=str(profile / ".config"), XDG_CACHE_HOME=str(profile / ".cache"),
               XDG_STATE_HOME=str(profile / ".local/state"))
    return env


def play(args, variant, model, video, tag, extra):
    files = args.out / tag
    data = args.out / f"réglages-{tag}"
    configuration(data, variant)
    env = player_environment(args.out)
    if variant in ("prototype", "zml", "zml-hote") and args.zml_cuda_root:
        root = args.zml_cuda_root
        root_flag = json.dumps(str(root), ensure_ascii=False)
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_gpu_cuda_data_dir={root_flag}".strip()
    binary = {"cuda-natif": args.kirari_cuda, "prototype": args.kirari_prototype, "zml-hote": args.kirari_hote}.get(variant, args.kirari)
    argv = [str(binary), "play", str(video), "--debug", "--mute", "--no-resume", "--rife=0", "--sdr",
            "--no-scene-map", "--no-span-reuse", "--no-span-tiles", "--present=auto",
            f"--window={args.window}", f"--work-height={args.height}", f"--data-dir={data}",
            f"--model={model}", f"--bench-out={files}.toml", f"--stats={files}.tsv",
            "--span-native" if variant in ("cuda", "cuda-natif") else "--no-span-native", *extra]
    print(f"{tag} · {variant}", flush=True)
    run(argv, Path(f"{files}.log"), args.timeout, env)
    with Path(f"{files}.toml").open("rb") as stream:
        reading = tomllib.load(stream)
    check_engine(reading, variant, files, args)
    return files, reading


def check_engine(reading, variant, files, args):
    engine = ENGINES[variant]
    span = "FP16 native" if variant in ("cuda", "cuda-natif") else "FP16"
    actual = reading["engine"].lower()
    matches = actual == engine if variant in INTEGRATED else engine in actual
    if not matches or reading["span_variant"] != span:
        raise RuntimeError(f"{variant} : voie ou précision SPAN inattendue, voir {files}.toml")
    if args.expected_gpu not in reading["gpu"]:
        raise RuntimeError(f"{variant} : carte {reading['gpu']} au lieu de {args.expected_gpu}")
    if reading["work_height"] != args.height or reading["presents"] < 2:
        raise RuntimeError(f"{variant} : hauteur de travail ou présentations inattendues, voir {files}.toml")
    if variant in ("zml", "zml-hote"):
        log = Path(f"{files}.log").read_text(encoding="utf-8", errors="replace")
        samples = re.findall(r"profil · zml [^\n]*SPAN[^\n]* : médiane [0-9.]+ ms, p95 ([0-9.]+) ms \(([1-9][0-9]*) passes\)", log)
        transfers = re.findall(r"interop · zml [^\n]*SPAN[^\n]* pour (\d+)/(\d+) passes", log)
        stages = [stage for stage in reading.get("gpu_stage", []) if stage["name"] == "rôle agrandissement" and stage["count"]]
        if (not samples and not (transfers and stages)) or ("export-" in files.name and not re.search(r"export : [^\n]+ · zml · ",log)):
            raise RuntimeError(f"exécution ZML absente de {files}.log")
        if variant == "zml" and args.kirari_hote:
            if not transfers or any(a != b for a,b in transfers) or "partage GPU indisponible" in log:
                raise RuntimeError(f"partage GPU incomplet dans {files}.log")
        if variant == "zml-hote" and (transfers or "tenseurs par l'hôte" not in log):
            raise RuntimeError(f"le témoin en mémoire hôte manque dans {files}.log")
        reading["span_p95"] = max(stage["p95_ms"] for stage in stages) if stages else max(float(ms) for ms,_ in samples)
        return
    if variant in ("hip", "slang", "triton") and "repli Vulkan" in Path(f"{files}.log").read_text(encoding="utf-8"):
        raise RuntimeError(f"{variant} : le réseau a pris le repli Vulkan, voir {files}.log")
    wanted = STAGES[variant]
    if not any(wanted in stage["name"] and stage["count"] for stage in reading.get("gpu_stage", [])):
        raise RuntimeError(f"{variant} : aucune exécution de {wanted}, voir {files}.toml et {files}.log")


    reading["span_p95"] = max(stage["p95_ms"] for stage in reading["gpu_stage"] if wanted in stage["name"] and stage["count"])


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percent / 100)]


def presentation(files):
    with Path(f"{files}.tsv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    fresh = [row for row in rows if row["fresh"] == "1"]
    if len(fresh) < 2:
        raise RuntimeError(f"moins de deux images neuves dans {files}.tsv")
    ids = {row["present_id"] for row in fresh if row["present_id"] != "0"}
    visible = {row["feedback_id"]: int(row["visible_us"]) for row in rows
               if row["feedback_id"] in ids and int(row["visible_us"]) > 0}
    stamps = sorted(set(visible.values()))
    confirmed = visible_stats(f"{files}.tsv", rows, partial=True)
    visible_fps = (len(stamps) - 1) * 1_000_000 / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
    confirmed_count = len(stamps)
    if confirmed is not None:
        visible_fps = confirmed["images_neuves_s"]
        confirmed_count = confirmed["images_neuves"]
    gpu_us = [int(row["display_gpu_us"]) for row in fresh if int(row["display_gpu_us"]) > 0]
    elapsed_us = int(fresh[-1]["host_us"]) - int(fresh[0]["host_us"])
    if elapsed_us <= 0:
        raise RuntimeError(f"temps des présentations invalide dans {files}.tsv")
    return {
        "feedback": confirmed_count,
        "fresh": len(fresh),
        "first_fresh_host_us": int(fresh[0]["host_us"]),
        "presentation_span_us": elapsed_us,
        "media_span_us": int(fresh[-1]["pts_us"]) - int(fresh[0]["pts_us"]),
        "fresh_fps": (len(fresh) - 1) * 1_000_000 / elapsed_us,
        "visible_fps": visible_fps,
        "levels": ", ".join(sorted({row["level"] for row in fresh})),
        "composite_gpu_p95": percentile(gpu_us, 95),
    }


def timed_reads(args, models):
    results = {variant: [] for variant in args.variants}
    first = args.variants
    for order in (first, first[::-1]):
        for variant in order:
            tag = f"lecture-{variant}-{len(results[variant]) + 1}"
            files, read = play(args, variant, models[variant], args.video, tag,
                               [f"--start={args.start}", f"--seconds={args.seconds}"])
            results[variant].append((read, presentation(files)))
    return results


def frames(args, path):
    # FFV1 met une image intra par paquet. Le PSNR vérifiera ensuite le nombre d'images décodées.
    got = subprocess.run([args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets", "-show_streams",
                          "-show_entries", "stream=codec_name,width,height,pix_fmt,time_base:packet=pts",
                          "-of", "json", str(path)],
                         check=True, capture_output=True, text=True, timeout=args.timeout)
    metadata = json.loads(got.stdout)
    streams, packets = metadata.get("streams", []), metadata.get("packets", [])
    if len(streams) != 1 or streams[0].get("codec_name") != "ffv1" or not packets:
        raise RuntimeError(f"export FFV1 sans images : {path}")
    if any("pts" not in packet for packet in packets):
        raise RuntimeError(f"export sans PTS : {path}")
    return streams[0], [packet["pts"] for packet in packets]


def compare_frames(args, candidate, reference, frame_count, name):
    stats = f"psnr-{name}.txt"
    # Les deux séries de PTS, dimensions et formats ont déjà été comparées par frames().
    result = subprocess.run([args.ffmpeg, "-hide_banner", "-nostats", "-filter_complex_threads", "4",
                             "-threads", "4", "-i", str(candidate), "-threads", "4", "-i", str(reference),
                             "-lavfi", f"[0:v][1:v]psnr=stats_file={stats}", "-f", "null", "-"],
                            cwd=args.out, check=True, capture_output=True, text=True, timeout=args.timeout)
    (args.out / f"psnr-{name}.log").write_text(result.stderr, encoding="utf-8")
    per_frame = re.findall(r"\bpsnr_avg:([\d.]+|inf)\b", (args.out / stats).read_text(encoding="utf-8"))
    if len(per_frame) != frame_count:
        raise RuntimeError(f"{name} : {len(per_frame)} scores pour {frame_count} images")
    return min(map(float, per_frame))


def quality(args, models):
    clip = args.out / "source-qualité.mkv"
    run([args.ffmpeg, "-hide_banner", "-nostdin", "-ss", str(args.start), "-i", str(args.video),
         "-map", "0:v:0", "-an", "-sn", "-t", str(args.quality_seconds), "-c:v", "ffv1", str(clip)],
        args.out / "source-qualité.log", args.timeout)
    exports = {variant: args.out / f"qualité-{variant}.mkv" for variant in args.variants}
    for variant, destination in exports.items():
        if any(char in str(destination) for char in ",;"):
            raise RuntimeError("le chemin de sortie ne peut contenir ni virgule ni point-virgule")
        play(args, variant, models[variant], clip, f"export-{variant}",
             ["--seconds=1", f"--keys=0.5:export-lossless={destination}"])
    reference = frames(args, exports[args.variants[0]])
    fmt, pts = reference
    if fmt["height"] != 2 * args.height or not re.fullmatch(r"(?:yuv\w*|gbrp)10(?:le|be)", fmt["pix_fmt"]):
        raise RuntimeError(f"export SPAN 2× 10 bits attendu, obtenu {fmt}")
    args.export_format = fmt
    args.frame_count = len(pts)
    for variant in args.variants[1:]:
        if frames(args, exports[variant]) != reference:
            raise RuntimeError(f"{variant} : PTS, taille ou pixel format différents de la référence")
    return {f"{a} / {b}": compare_frames(args, exports[a], exports[b], len(reference[1]), f"{a}-{b}")
            for a, b in combinations(args.variants, 2)}


def report(args, results, scores):
    lines = [f"# SPAN FP16 sur {args.expected_gpu}", "",
             f"Modèle {args.model.name}.", "",
             f"Fenêtre {args.window}, travail {args.height}p, source {args.video.name}.", "",
             "| Voie | Images neuves/s soumises | Images neuves/s confirmées | Images source manquées (%) | "
             "Intervalle p95 (ms) | Retard p95 (ms) | "
             "SPAN p95 (ms) | Composition CPU médiane (ms) | Affichage GPU p95 (ms) | Retour daté |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for variant, runs in results.items():
        def worst(key):
            return max(read[key] for read, _ in runs) / 1000
        lost = max(r["missing"] / (r["new_frames"] + r["missing"]) for r, _ in runs) * 100
        composition = statistics.median(read["render_median_us"] for read, _ in runs) / 1000
        gpu = [shown["composite_gpu_p95"] for _, shown in runs if shown["composite_gpu_p95"] is not None]
        gpu_text = f"{max(gpu) / 1000:.2f}" if gpu else "—"
        feedback = " + ".join(f"{shown['levels']} {shown['feedback']}/{shown['fresh']}" for _, shown in runs)
        fps = min(shown["fresh_fps"] for _, shown in runs)
        dated = [shown["visible_fps"] for _, shown in runs if shown["visible_fps"] is not None]
        fps_visible = f"{min(dated):.2f}" if len(dated) == len(runs) else "—"
        span = max(read["span_p95"] for read, _ in runs)
        lines.append(f"| {variant} | {fps:.2f} | {fps_visible} | {lost:.2f} | {worst('interval_p95_us'):.2f} | "
                     f"{worst('lag_p95_us'):.2f} | {span:.2f} | {composition:.2f} | {gpu_text} | {feedback} |")
    lines += ["", "| Voie | Passage | Images neuves | Durée de lecture (s) | Images neuves/s sur la session |",
              "|---|---:|---:|---:|---:|"]
    for variant, runs in results.items():
        for i, (read, _) in enumerate(runs, 1):
            elapsed_us = read.get("playback_elapsed_us", 0)
            total = read["new_frames"]
            duration = f"{elapsed_us / 1e6:.3f}" if elapsed_us > 0 else "—"
            rate = f"{total * 1e6 / elapsed_us:.2f}" if elapsed_us > 0 else "—"
            lines.append(f"| {variant} | {i} | {total} | {duration} | {rate} |")
    lines += ["", "Le débit en lecture est borné par la cadence de la source et de l’écran. "
              "La cadence confirmée peut sous-estimer l’affichage quand des retours datés manquent. Pour ZML, le p95 comprend les transferts et l’attente de PJRT."]
    lines += ["", f"Qualité de {args.frame_count} images alignées en FFV1 {args.export_format['pix_fmt']}, {args.export_format['width']} × {args.export_format['height']}. PSNR de la pire image (seuil {args.quality_db:g} dB)", ""]
    for pair, value in scores.items():
        lines.append(f"- {pair} · {value:.2f} dB · {'seuil tenu' if value >= args.quality_db else 'sous le seuil'}")
    text = "\n".join(lines) + "\n"
    (args.out / "rapport.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("kirari", "model", "video", "out"):
        parser.add_argument(f"--{option}", required=True, type=lambda s: Path(s).expanduser().resolve())
    parser.add_argument("--kirari-cuda", type=Path, help="lecteur équipé de CUDA natif pour la comparaison sur NVIDIA")
    parser.add_argument("--kirari-prototype", type=Path, help="lecteur du prototype .zmlspan")
    parser.add_argument("--kirari-hote", type=Path, help="lecteur ZML sans partage CUDA pour isoler le coût des transferts")
    parser.add_argument("--zml-cuda-root", type=Path, help="sandbox du compilateur CUDA pour le développement")
    parser.add_argument("--expected-gpu", required=True, help="nom ou partie du nom de la carte attendue")
    parser.add_argument("--voies", nargs="+", choices=list(ENGINES), help="voies à comparer dans cet ordre, puis en ordre inverse")
    parser.add_argument("--window", default="1600x900")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--quality-seconds", type=int, default=3)
    parser.add_argument("--quality-db", type=float, default=50.0)
    parser.add_argument("--timeout", type=int, default=1200, help="durée maximale par commande en secondes")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if not args.kirari.is_file() or not args.model.is_file() or not args.video.is_file():
        parser.error("exécutable, modèle ou vidéo introuvable")
    if args.zml_cuda_root and not (args.zml_cuda_root / "lib" / "libzml_cuda.so").is_file():
        parser.error("le sandbox CUDA de ZML doit contenir lib/libzml_cuda.so")
    if args.model.name not in MODELS:
        parser.error("utiliser un modèle AnimeJaNai HD V3.1 Balanced ou Performance FP16")
    if min(args.height, args.seconds, args.quality_seconds, args.timeout) <= 0 or args.height % 2 or args.start < 0:
        parser.error("durées positives et hauteur de travail paire requises")
    try:
        args.out.mkdir(parents=True, exist_ok=False)
        optional = {"cuda-natif": args.kirari_cuda, "prototype": args.kirari_prototype, "zml-hote": args.kirari_hote}
        args.variants = tuple(k for k, v in optional.items() if v) + ("zml", "vulkan")
        if args.voies:
            if (len(set(args.voies)) != len(args.voies) or len(args.voies) < 2 or
                    not set(args.voies).issubset((*args.variants, *INTEGRATED))):
                parser.error("prévoir un exécutable pour chacune des voies et choisir au moins deux voies distinctes")
            args.variants = tuple(args.voies)
        models = dict.fromkeys(args.variants, args.model)
        if "prototype" in args.variants:
            if args.model.name != MODEL:
                parser.error("le prototype .zmlspan demande AnimeJaNai Balanced b8f64 FP16")
            alias = args.out / "Balanced.zmlspan"
            alias.symlink_to(args.model)
            models["prototype"] = alias
        results = timed_reads(args, models)
        scores = quality(args, models)
        report(args, results, scores)
        return 0 if all(value >= args.quality_db for value in scores.values()) else 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Banc interrompu · {exc}\n")


if __name__ == "__main__":
    with essai_exclusif():
        raise SystemExit(main())
