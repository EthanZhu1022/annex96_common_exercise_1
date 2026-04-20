"""
Baseline Comparison Utility — extends independent_sac/compare.py to include
RLlib Independent SAC results alongside MAPPO and SB3/custom Independent SAC.

Usage:
    python -m rllib_sac.compare \\
        --rllib  results/rllib_sac/test_metrics.csv \\
        --sac    results/independent_sac/test_metrics.csv \\
        --mappo  results/mappo/test_metrics.csv \\
        --output results/comparison_table.csv

Delegates to independent_sac.compare.compare() which already handles the
common CSV schema, then just passes additional label/path pairs in.
"""

import argparse
import sys
from typing import Dict, Optional

from independent_sac.compare import compare, METRIC_COLS  # reuse shared logic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RLlib Independent SAC + other baselines for Annex96-CE1"
    )
    parser.add_argument("--rllib",  default=None, metavar="CSV",
                        help="Path to RLlib Independent SAC test_metrics.csv")
    parser.add_argument("--sac",    default=None, metavar="CSV",
                        help="Path to Independent SAC test_metrics.csv")
    parser.add_argument("--mappo",  default=None, metavar="CSV",
                        help="Path to MAPPO test_metrics.csv")
    parser.add_argument("--rbc",    default=None, metavar="CSV",
                        help="Path to RBC test_metrics.csv (if available)")
    parser.add_argument("--extra",  nargs="+", metavar="LABEL=PATH",
                        help="Additional baselines as LABEL=PATH pairs")
    parser.add_argument("--output", default=None, metavar="CSV",
                        help="Save merged table to this CSV path")
    args = parser.parse_args()

    files: Dict[str, str] = {}
    if args.rllib:
        files["RLlib-Independent-SAC"] = args.rllib
    if args.sac:
        files["Independent-SAC"] = args.sac
    if args.mappo:
        files["MAPPO"] = args.mappo
    if args.rbc:
        files["RBC"] = args.rbc
    if args.extra:
        for item in args.extra:
            if "=" not in item:
                print(f"[error] --extra items must be LABEL=PATH, got: {item}",
                      file=sys.stderr)
                sys.exit(1)
            label, path = item.split("=", 1)
            files[label] = path

    if not files:
        parser.print_help()
        sys.exit(1)

    compare(files, output=args.output)


if __name__ == "__main__":
    main()
