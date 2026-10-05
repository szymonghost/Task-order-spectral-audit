"""Command-line interface for a complete-landscape spectral audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .audit import AuditConfig, analyze_landscape, publish_audit
from .schema import load_npz_landscape


def _partition_argument(value: str) -> tuple[int, ...]:
    text = value.strip()
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
        text = text[1:-1].strip()
    try:
        parts = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "partition must be comma-separated positive integers, e.g. 3,2,1"
        ) from error
    if not parts or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(
            "partition must contain positive integers, e.g. 3,2,1"
        )
    if any(parts[index] < parts[index + 1]
           for index in range(len(parts) - 1)):
        raise argparse.ArgumentTypeError(
            "partition parts must be nonincreasing"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskorder-spectral",
        description=(
            "Audit one or more complete scalar landscapes on S_K. NPZ field "
            "names, matrix layout, and output location are always explicit."
        ),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True, dest="K")
    parser.add_argument(
        "--permutations-key",
        required=True,
        help="NPZ key for the complete (K!, K) permutation grid",
    )
    parser.add_argument(
        "--values-key",
        required=True,
        help="NPZ key for one or more complete scalar landscapes",
    )
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds-key",
        help="NPZ key for seed IDs",
    )
    seed_group.add_argument(
        "--implicit-seeds",
        action="store_true",
        help="use integer column indices as seed IDs",
    )
    parser.add_argument(
        "--values-layout",
        required=True,
        choices=("group-by-seed", "seed-by-group"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--high-order-min", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-exact-seeds", type=int, default=16)
    parser.add_argument("--monte-carlo-signs", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20_260_524)
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help=(
            "publish exact plug-in decompositions and projection summaries "
            "without seed-level signal inference; required for fewer than "
            "three independent landscapes"
        ),
    )
    parser.add_argument(
        "--allocation-target",
        action="append",
        default=[],
        type=_partition_argument,
        metavar="PARTS",
        help=(
            "test one partition against its dimension-squared share within "
            "the same interaction order (e.g. 3,2,1); repeat to define a "
            "Holm-adjusted requested-target family"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    landscape = load_npz_landscape(
        args.artifact,
        K=args.K,
        permutations_key=args.permutations_key,
        values_key=args.values_key,
        seeds_key=None if args.implicit_seeds else args.seeds_key,
        values_layout=args.values_layout,
    )
    config = AuditConfig(
        high_order_min=args.high_order_min,
        confidence=args.confidence,
        max_exact_seeds=args.max_exact_seeds,
        monte_carlo_signs=args.monte_carlo_signs,
        random_seed=args.random_seed,
        descriptive_only=args.descriptive_only,
        allocation_targets=tuple(args.allocation_target),
    )
    result = analyze_landscape(landscape, config)
    manifest = publish_audit(result, args.output_dir)
    high = result.high_order_rows[0]
    if args.descriptive_only:
        print(
            f"K={landscape.K}; landscapes={landscape.n_seeds}; "
            f"partitions={len(result.partition_rows)}; "
            f"high-order plugin={float(high['plugin_energy']):.10g}; "
            "seed inference=not run (descriptive only)"
        )
    else:
        print(
            f"K={landscape.K}; seeds={landscape.n_seeds}; "
            f"partitions={len(result.partition_rows)}; "
            f"high-order signal={float(high['signal_energy']):.10g}; "
            f"p={float(high['seed_signflip_p_raw']):.10g}"
        )
    if result.allocation_rows:
        print(
            "within-order allocation targets: "
            + ", ".join(
                str(row["target_partition"]) for row in result.allocation_rows
            )
        )
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
