

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wsr_miso as m

# ---------------------------------------------------------------- parameters
N_TX = 8                 # antennas per BS
N_USERS = 4              # total users
N_CELLS = 2              # cells
CELL_OF_USER = np.array([0, 0, 1, 1])

P_TX_DBM = 30.0          # per-cell transmit power, fixed (1 W)
PATHLOSS_DB = -100.0     # large scale gain from a BS to a user
N0_DBM_HZ = -174.0       # thermal noise PSD
NOISE_FIGURE_DB = 9.0    # receiver noise figure

# 1 MHz ... 100 GHz.  The upper decades are not realizable systems; they are
# included to expose the saturation of the absolute rate (power limited regime).
BANDWIDTHS_HZ = np.array(
    [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8,
     1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11]
)

N_REALIZATIONS = 20      # Monte Carlo channels (identical across bandwidths)
SEED = 2012

WEIGHT_SETS = {
    "equal weights a = [1, 1, 1, 1]": np.ones(N_USERS),
    "unequal weights a = [0.14, 0.21, 0.28, 0.36]":
        np.array([0.14, 0.21, 0.28, 0.36]),
}


def normalized_snr(bandwidth_hz):
    """Per-cell transmit power normalized by the receiver noise power."""
    noise_dbm = N0_DBM_HZ + NOISE_FIGURE_DB + 10 * np.log10(bandwidth_hz)
    return 10 ** ((P_TX_DBM + PATHLOSS_DB - noise_dbm) / 10.0)


def wideband_limit(H, weights):
    
    k = 10 ** ((P_TX_DBM + PATHLOSS_DB - N0_DBM_HZ - NOISE_FIGURE_DB) / 10.0)
    total = 0.0
    for c in range(N_CELLS):
        idx = np.where(CELL_OF_USER == c)[0]
        gains = np.array([weights[u] * np.linalg.norm(H[u, c, :]) ** 2 for u in idx])
        total += k * gains.max() / np.log(2)
    return total


def run():
    rng_ch = np.random.default_rng(SEED)
    channels = [m.generate_channel(N_USERS, N_CELLS, N_TX, rng_ch)
                for _ in range(N_REALIZATIONS)]

    rows = []
    for label, weights in WEIGHT_SETS.items():
        for b in BANDWIDTHS_HZ:
            rho = normalized_snr(b)
            se, thr, iters = [], [], []
            for r, H in enumerate(channels):
                out = m.solve(
                    H, CELL_OF_USER, rho * np.ones(N_CELLS), weights,
                    noise_power=1.0, rng=np.random.default_rng(1000 + r),
                    n_iterations=40, tol=1e-4,
                )
                se.append(out["wsr"][-1])
                thr.append(b * out["wsr"][-1])
                iters.append(out["n_iter"])
            rows.append({
                "weights": label,
                "B_Hz": b,
                "B_MHz": b / 1e6,
                "rho_dB": 10 * np.log10(rho),
                "WSR_bps_per_Hz": float(np.mean(se)),
                "WSR_std": float(np.std(se)),
                "WSR_Gbps": float(np.mean(thr)) / 1e9,
                "mean_iterations": float(np.mean(iters)),
            })
            print(f"{label[:14]:14s} B={b/1e6:8.1f} MHz  rho={10*np.log10(rho):6.2f} dB "
                  f" WSR={np.mean(se):7.3f} b/s/Hz  {np.mean(thr)/1e9:7.3f} Gb/s "
                  f" ({np.mean(iters):.1f} iters)")

    df = pd.DataFrame(rows)
    df.to_csv("wsr_vs_bandwidth.csv", index=False)

    limits = {lab: np.mean([wideband_limit(H, w) for H in channels])
              for lab, w in WEIGHT_SETS.items()}

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {list(WEIGHT_SETS)[0]: "tab:red", list(WEIGHT_SETS)[1]: "tab:blue"}
    marks = {list(WEIGHT_SETS)[0]: "s", list(WEIGHT_SETS)[1]: "d"}

    for lab in WEIGHT_SETS:
        d = df[df.weights == lab]
        axes[0].semilogx(d.B_Hz, d.WSR_bps_per_Hz, "-", marker=marks[lab],
                         color=colors[lab], mfc=colors[lab], label=lab)
        axes[1].semilogx(d.B_Hz, d.WSR_Gbps, "-", marker=marks[lab],
                         color=colors[lab], mfc=colors[lab], label=lab)
        axes[1].axhline(limits[lab] / 1e9, ls="--", lw=1, color=colors[lab])

    axes[0].set_xlabel("Bandwidth B (Hz)")
    axes[0].set_ylabel("Weighted sum rate (b/s/Hz)")
    axes[0].set_title("Spectral efficiency: noise grows with B")
    axes[1].set_xlabel("Bandwidth B (Hz)")
    axes[1].set_ylabel("Weighted sum rate (Gb/s)")
    axes[1].set_title("Throughput: saturates at the wideband limit (dashed)")
    for ax in axes:
        ax.grid(True, which="both", ls=":")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("WSR_vs_Bandwidth.png", dpi=160)
    for lab, v in limits.items():
        print(f"wideband (B -> inf) limit, {lab}: {v/1e9:.2f} Gb/s")
    print("\nsaved WSR_vs_Bandwidth.png and wsr_vs_bandwidth.csv")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    run()
