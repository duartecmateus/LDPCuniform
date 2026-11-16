#!/usr/bin/env python3

import math
import os
import subprocess
import sys

TRIALS = 32
EXECUTABLE = os.path.join("build", "genU")

# Parameter sets: (field_name, field_code, beta_pct, alpha, c, n_values)
# alpha and c from the paper's recommendations in sec05-syndrome-analysis.tex
PARAM_SETS = [
    # (filename_prefix, field_code, c, alpha, description)
    # field_code: 0=GF(2) binary, 1=GF(256)
    ("H_bin_BER1",   0, 6, 0.20, "F2 @ 1%"),
    ("H_GF256_BER1", 1, 6, 0.12, "F256 @ 1%"),
    ("H_bin_BER4",   0, 6, 0.43, "F2 @ 4%"),
    ("H_GF256_BER4", 1, 6, 0.30, "F256 @ 4%"),
]

N_VALUES = [2000, 3000]


def run_command(cmd):
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main():
    if not os.path.exists(EXECUTABLE):
        raise RuntimeError(
            f"Executable '{EXECUTABLE}' not found. Run 'make' first in the genmatrix directory."
        )

    for prefix, field, c, alpha, desc in PARAM_SETS:
        for n in N_VALUES:
            m = int(round(alpha * n))
            threshold = int(round(c / alpha))  # d_v = c/alpha = expected row weight
            output_name = f"{prefix}_{n}.txt"

            cmd = [
                EXECUTABLE,
                str(m),
                str(n),
                str(c),
                str(threshold),
                output_name,
                str(TRIALS),
                str(field),
            ]
            print(f"\n--- Generating {desc}, n={n}, m={m}, c={c}, d_v={threshold} ---")
            run_command(cmd)

    print("\n[OK] All matrices generated successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
