#!/usr/bin/env python3.10
import os
import pandas as pd
import matplotlib.pyplot as plt

PARALLEL_CSV = "results_parallel.csv"
SEQUENTIAL_CSV = "results_sequential.csv"
CUDA_CSV = "results_cuda.csv"

OUT_RUNTIME_SIZE = "alg2_runtime_vs_size.pdf"
OUT_SPEEDUP_SIZE = "alg2_speedup_vs_size.pdf"
OUT_RUNTIME_ERROR = "alg2_runtime_vs_error_parallel.pdf"
OUT_SPEEDUP_CUDA = "alg2_speedup_cuda_vs_size.pdf"
OUT_RUNTIME_SIZE_ALL = "alg2_runtime_vs_size_all.pdf"
OUT_RUNTIME_ERROR_ALL = "alg2_runtime_vs_error_all.pdf"

REPRESENTATIVE_SIZES = [1500, 4000, 6000]
REFERENCE_ERROR_RATE = 2.0


def load_data():
    par = pd.read_csv(PARALLEL_CSV)
    seq = pd.read_csv(SEQUENTIAL_CSV)
    cuda = pd.read_csv(CUDA_CSV)
    return par, seq, cuda


def add_time_seconds(df):
    df = df.copy()
    df["time_s"] = df["time_ms"] / 1000.0
    return df


def plot_runtime_vs_size(par, seq, cuda):
    par2 = add_time_seconds(par[par["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()).sort_values("n")
    seq2 = add_time_seconds(seq[seq["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()).sort_values("n")
    cuda2 = add_time_seconds(cuda[cuda["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()).sort_values("n")

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(seq2["n"], seq2["time_s"], marker="o", linewidth=2, label="Sequential CPU")
    plt.plot(par2["n"], par2["time_s"], marker="s", linewidth=2, label="Metal GPU")
    plt.plot(cuda2["n"], cuda2["time_s"], marker="^", linewidth=2, label="CUDA GPU")
    plt.yscale("log")
    plt.xlabel("Number of columns $n$")
    plt.ylabel("Decoding time (s)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_RUNTIME_SIZE_ALL, bbox_inches="tight")
    plt.close()


def plot_speedup_vs_size(par, seq, cuda):
    par2 = par[par["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()
    seq2 = seq[seq["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()
    cuda2 = cuda[cuda["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()

    merged = pd.merge(
        seq2[["n", "time_ms"]],
        par2[["n", "time_ms"]],
        on="n",
        suffixes=("_seq", "_metal"),
    )
    merged = pd.merge(
        merged,
        cuda2[["n", "time_ms"]],
        on="n",
    ).rename(columns={"time_ms": "time_ms_cuda"})
    merged["speedup_metal"] = merged["time_ms_seq"] / merged["time_ms_metal"]
    merged["speedup_cuda"] = merged["time_ms_seq"] / merged["time_ms_cuda"]
    merged = merged.sort_values("n")

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(merged["n"], merged["speedup_metal"], marker="s", linewidth=2, label="Sequential / Metal")
    plt.plot(merged["n"], merged["speedup_cuda"], marker="^", linewidth=2, label="Sequential / CUDA")
    plt.xlabel("Number of columns $n$")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_SPEEDUP_SIZE, bbox_inches="tight")
    plt.close()


def plot_cuda_speedup_vs_size(par, cuda):
    par2 = par[par["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()
    cuda2 = cuda[cuda["error_rate_percent"] == REFERENCE_ERROR_RATE].copy()
    merged = pd.merge(
        par2[["n", "time_ms"]],
        cuda2[["n", "time_ms"]],
        on="n",
        suffixes=("_metal", "_cuda"),
    )
    merged["speedup"] = merged["time_ms_metal"] / merged["time_ms_cuda"]
    merged = merged.sort_values("n")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(merged["n"], merged["speedup"], marker="o", linewidth=2, color="tab:green")
    plt.xlabel("Number of columns $n$")
    plt.ylabel("Speedup $T_{\mathrm{Metal}}/T_{\mathrm{CUDA}}$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_SPEEDUP_CUDA, bbox_inches="tight")
    plt.close()


def plot_runtime_vs_error_all(par, cuda):
    par2 = add_time_seconds(par)
    cuda2 = add_time_seconds(cuda)
    plt.figure(figsize=(6.6, 4.6))

    for n in REPRESENTATIVE_SIZES:
        sub_par = par2[par2["n"] == n].sort_values("error_rate_percent")
        sub_cuda = cuda2[cuda2["n"] == n].sort_values("error_rate_percent")
        if len(sub_par):
            plt.plot(sub_par["error_rate_percent"], sub_par["time_s"], marker="o", linewidth=2, label=f"Metal, n={n}")
        if len(sub_cuda):
            plt.plot(sub_cuda["error_rate_percent"], sub_cuda["time_s"], marker="^", linewidth=2, linestyle="--", label=f"CUDA, n={n}")

    plt.xlabel("Error rate (\%)")
    plt.ylabel("Decoding time (s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_RUNTIME_ERROR_ALL, bbox_inches="tight")
    plt.close()


def plot_runtime_vs_error_parallel(par):
    par2 = add_time_seconds(par)
    plt.figure(figsize=(6.2, 4.2))

    for n in REPRESENTATIVE_SIZES:
        sub = par2[par2["n"] == n].sort_values("error_rate_percent")
        if len(sub) == 0:
            continue
        plt.plot(sub["error_rate_percent"], sub["time_s"], marker="o", linewidth=2, label=f"n={n}")

    plt.xlabel("Error rate (\%)")
    plt.ylabel("Decoding time (s)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_RUNTIME_ERROR, bbox_inches="tight")
    plt.close()


def main():
    for path in (PARALLEL_CSV, SEQUENTIAL_CSV, CUDA_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")

    par, seq, cuda = load_data()
    plot_runtime_vs_size(par, seq, cuda)
    plot_speedup_vs_size(par, seq, cuda)
    plot_cuda_speedup_vs_size(par, cuda)
    plot_runtime_vs_error_parallel(par)
    plot_runtime_vs_error_all(par, cuda)

    print("Generated:")
    for out in (
        OUT_RUNTIME_SIZE_ALL,
        OUT_SPEEDUP_SIZE,
        OUT_SPEEDUP_CUDA,
        OUT_RUNTIME_ERROR,
        OUT_RUNTIME_ERROR_ALL,
    ):
        print(f"  {out}")


if __name__ == "__main__":
    main()
