#!/usr/bin/env python3
"""Calcule la borne après Winograd et fusion pour un graphe RIFE CUDA direct en 1080p.

Donner le préfixe commun des morceaux ONNX exportés, jusqu'à l'empreinte incluse.
Le warp RGB et la présentation sont hors de ce graphe. Voir docs/rife-plafonds.md.
"""
import argparse
from dataclasses import dataclass
from math import prod
from pathlib import Path

from onnx import TensorProto, helper, load, shape_inference

WIDTH, HEIGHT = 1920, 1088
TENSOR_FP32 = 86.75e12
CUDA_FP32 = 40.1e12
L2 = 2.748e12
DDR = 504e9

MODELS = {
    "rife_v4.25": (8, 8, 4, 2, 1),
    "rife_v4.22_lite": (8, 4, 2, 1),
}


@dataclass
class Stage:
    name: str
    products: int = 0
    transforms: int = 0
    weights: int = 0
    transfers: float = 0
    scalar_ops: int = 0

    def times(self, ddr_fraction):
        moved = self.weights + self.transfers
        return (2 * self.products / TENSOR_FP32, (2 * self.transforms + self.scalar_ops) / CUDA_FP32,
                moved / L2, moved * ddr_fraction / DDR)


def shape(value):
    return tuple(d.dim_value for d in value.type.tensor_type.shape.dim)


def count_piece(path, stage, reduce, *, final=False):
    graph = shape_inference.infer_shapes(load(path), strict_mode=True).graph
    shapes = {v.name: shape(v) for v in [*graph.input, *graph.value_info, *graph.output]}
    weights = {w.name: tuple(w.dims) for w in graph.initializer}
    (entry,) = graph.input
    if shape(entry)[-2:] != (HEIGHT // reduce, WIDTH // reduce):
        raise ValueError(f"{path}: résolution d'entrée inattendue")
    if entry.type.tensor_type.elem_type != TensorProto.FLOAT16:
        raise ValueError(f"{path}: activations FP16 attendues")
    for node in graph.node:
        if node.op_type == "Mul":
            continue  # Le facteur résiduel est replié dans les poids et le biais.
        if node.op_type in ("Add", "LeakyRelu"):
            stage.scalar_ops += prod(shapes[node.output[0]])
            continue
        if node.op_type not in ("Conv", "ConvTranspose"):
            raise ValueError(f"{path}: opérateur inattendu {node.op_type}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        batch, cin, ih, iw = shapes[node.input[0]]
        out_batch, channels, oh, ow = shapes[node.output[0]]
        transposed = node.op_type == "ConvTranspose"
        cout = 20 if final and transposed else channels
        kernel = weights[node.input[1]][-2:]
        if (batch != 1 or out_batch != batch or cout > channels or min(cin, cout, ih, iw, oh, ow) <= 0 or
                attrs.get("group", 1) != 1 or attrs.get("dilations", [1, 1]) != [1, 1]):
            raise ValueError(f"{path}: forme inattendue pour {node.name}")
        if len(node.input) > 2:
            stage.scalar_ops += cout * oh * ow
        eligible = (not transposed and kernel == (3, 3) and attrs.get("strides", [1, 1]) == [1, 1])
        if eligible:
            tiles = ((oh + 1) // 2) * ((ow + 1) // 2)
            stage.products += tiles * cin * cout * 16
            stage.transforms += tiles * (32 * cin + 24 * cout)
            stage.weights += 2 * cin * cout * 16
        else:
            sites = ih * iw if transposed else oh * ow
            stage.products += sites * cin * cout * prod(kernel)
            stage.weights += 2 * cin * cout * prod(kernel)


def bytes_of(channels, reduce=1):
    return 2 * channels * (HEIGHT // reduce) * (WIDTH // reduce)


def count_model(prefix, scales, new_encode):
    stages = []
    if new_encode:
        encode = Stage("encodeur B", transfers=bytes_of(7))
        count_piece(f"{prefix}-encode.onnx", encode, 1)
        stages.append(encode)

    for index, d in enumerate(scales):
        sites = (HEIGHT // d) * (WIDTH // d)
        if index == 0:
            transfers = 4 * bytes_of(14, d)
            glue = 4 * 14 * sites
        else:
            # Jusqu'à quatre voisins par interpolation bilinéaire et quatre prises pour la réduction.
            transfers = min(1, 16 / d**2) * bytes_of(14)
            transfers += sum(bytes_of(56 if j == index - 1 else 16, 2 * old)
                             for j, old in enumerate(scales[:index]))
            samples = 4 if d > 1 else 1
            glue = (samples * (4 * index * 10 + 4 * (index - 1) + 9 * 9 + 14 * 9)
                    + (28 * 4 if d > 1 else 0)) * sites
        if d == 1:
            # Sortie mouvement et aller-retour HWC4 entre la colle D1 et la tête.
            transfers += bytes_of(5) + 2 * bytes_of(4)
        else:
            transfers += bytes_of(56, 2 * d)
        stage = Stage(f"bloc {index} D{d}", transfers=transfers, scalar_ops=glue)
        count_piece(f"{prefix}-bloc{index}.onnx", stage, d, final=d == 1)
        stages.append(stage)
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="préfixe des exports à 1920×1088")
    parser.add_argument("--encode-repris", action="store_true")
    parser.add_argument("--ddr", type=float, default=.25, help="part du trafic logique supposée en DDR, entre 0 et 1")
    parser.add_argument("--graphe-ms", type=float, nargs=2, metavar=("AVANT", "APRES"),
                        help="médianes appariées du même graphe GPU, en ms")
    args = parser.parse_args()
    name = args.prefix.name.split("-fusion", 1)[0]
    if name not in MODELS or "-1088x1920-" not in args.prefix.name or not 0 <= args.ddr <= 1:
        parser.error("préfixe RIFE CUDA 1920×1088 et fraction DDR entre 0 et 1 attendus")
    if args.graphe_ms and not all(0 < value < float("inf") for value in args.graphe_ms):
        parser.error("les deux temps du graphe doivent être finis et strictement positifs")
    stages = count_model(args.prefix, MODELS[name], not args.encode_repris)
    print("Étage\tGF Tensor\tTensor (µs)\tFP32 (µs)\tL2 (µs)\tDDR (µs)\tBorne (µs)")
    total = 0.
    for stage in stages:
        times = stage.times(args.ddr)
        total += max(times)
        columns = "\t".join(f"{seconds * 1e6:.2f}" for seconds in (*times, max(times)))
        print(f"{stage.name}\t{2 * stage.products / 1e9:.3f}\t{columns}")
    bound = total * 1e3
    print(f"Borne {bound:.4f} ms · palier 90 % à {bound / .9:.4f} ms")
    if args.graphe_ms:
        before, after = (100 * bound / elapsed_ms for elapsed_ms in args.graphe_ms)
        print(f"{before:.2f} % → {after:.2f} % du même palier")


if __name__ == "__main__":
    main()
