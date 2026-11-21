"""Fige un plan Triton pour l'installation par le système de construction Kirari."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


def safe_name(name):
    return bool(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) and name not in (".", ".."))


def retarget_kernel(row, records, extension):
    label = row[1]
    info = records[label]
    if info["scratch"] != 0:
        raise ValueError(f"Noyau {label} à mémoire de travail implicite")
    grid = (int(row[4]), *(int(value) for value in row[7:])) if len(row) == 9 else (int(row[4]), 1, 1)
    target_grid = info["grille"]
    target_grid = tuple(target_grid) if isinstance(target_grid, list) else (target_grid, 1, 1)
    if info["symbole"] != row[3] or target_grid != grid:
        raise ValueError(f"Symbole ou pavage de noyau incompatible pour {label}")
    filename = f"{label}.{extension}"
    return ["kernel", label, filename, info["symbole"], row[4], str(info["threads"]), str(info["partage"]),
            *row[7:]]


def package(source, dest, hardware, extension):
    if dest.exists():
        raise FileExistsError(dest)

    plan = (source / "plan.tsv").read_text().splitlines()
    if len(plan) < 2 or not re.fullmatch(
            r"(?:span-triton-v1\t1920\t1080|drba-triton(?:-motion)?-v1\t(?:1920\t1088|1280\t768)"
            r"|rife(?:-lite)?-triton-motion-v1\t1920\t1088)\t[0-9a-f]{64}",
            plan[0]):
        raise ValueError("En-tête du plan Triton incorrect")

    symbols = json.loads((hardware / "rapport.json").read_text()) if hardware is not None else {}
    blobs = {}
    output = [plan[0]]
    kernel_labels = set()
    for line in plan[1:]:
        row = line.split("\t")
        if row[0] == "buffer" and len(row) == 4 and row[3] != "-":
            blobs[row[3]] = source
        elif row[0] == "kernel" and len(row) in (7, 9) and safe_name(row[1]):
            kernel_labels.add(row[1])
            if hardware is not None:
                row = retarget_kernel(row, symbols, extension)
            blobs[row[2]] = hardware if hardware is not None else source
        elif row[0] == "view" and len(row) == 5 and all(safe_name(name) for name in row[1:3]):
            pass
        elif not (row[0] == "launch" or (row[0] == "buffer" and len(row) == 4)):
            raise ValueError(f"Enregistrement inconnu : {line}")
        if row[0] == "launch" and hardware is not None and len(row) - 2 != symbols[row[1]]["arguments"]:
            raise ValueError(f"Nombre d'arguments incompatible pour {row[1]}")
        output.append("\t".join(row))
    if hardware is not None and kernel_labels != symbols.keys():
        raise ValueError("Le plan et les noyaux générés doivent être construits avec la même variante du modèle")

    if any(not safe_name(name) or name in ("plan.tsv", "SHA256SUMS") for name in blobs):
        raise ValueError("Noms de fichiers non valides")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
        staging = Path(temp) / "paquet"
        staging.mkdir()
        for name, directory in blobs.items():
            shutil.copyfile(directory / name, staging / name)
        (staging / "plan.tsv").write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
        sums = []
        for name in sorted((*blobs, "plan.tsv")):
            with (staging / name).open("rb") as stream:
                sums.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
        (staging / "SHA256SUMS").write_text("".join(sums), encoding="ascii", newline="\n")
        staging.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Dossier du plan CUDA et de ses poids")
    parser.add_argument("sortie", type=Path, help="Nouveau dossier destiné à Kirari")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--hip", type=Path, help="Dossier compile_amd.py, même variante et tuilage que le plan")
    target.add_argument("--xpu", type=Path, help="Dossier compile_xpu.py, même variante et tuilage que le plan")
    args = parser.parse_args()
    package(args.plan, args.sortie, args.hip or args.xpu, "hsaco" if args.hip else "spv")
