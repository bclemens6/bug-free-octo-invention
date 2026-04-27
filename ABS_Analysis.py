# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:25:36 2026

@author: benja
"""

"""
ABS Zone Comparison: 2025 vs 2026
=================================

Compares the called strike zone in 2025 (pre-ABS-challenge) and 2026
(with ABS challenges enforcing the height-based rule-book zone, 27%-53%
of batter height).

Key design choices:
- 2026 sz_top/sz_bot is height-based and constant per batter (verified).
  We use it as the canonical vertical reference in BOTH years for the
  matched-batter set. This puts both years on the same ruler.
- The "effective zone" measure: outcomes are post-challenge as provided.
- Outcome: called_strike = 1, ball/blocked_ball = 0.
- Surface: 2D Nadaraya-Watson kernel smoother for P(strike | x, z_norm).
- Edges from the 50% contour: top/bottom (at x=0), inside/outside (at
  z_norm=0.5), with width/height/area derived. Reported separately so a
  shift in just the top of the zone doesn't get hidden in aggregate area.
- Bootstrap: clustered by game_pk for honest CIs.
- Placebo: split 2025 by date into two halves and rerun the pipeline.

Run as cells in Spyder; section markers are `# %%`.
"""

# %% imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# %% config

DATA_DIR = Path(__file__).parent
OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

# Toggles ---------------------------------------------------------------
USE_HEIGHT_BASED_SZ = True   # matched batters; 2026 sz substituted into 2025
RESTRICT_CALENDAR = True     # restrict 2025 to the date-of-year span of 2026
# ----------------------------------------------------------------------

# Grid for evaluating the strike probability surface
X_GRID  = np.linspace(-1.5, 1.5, 121)   # feet, horizontal
ZN_GRID = np.linspace(-0.5, 1.5, 121)   # normalized vertical, [0,1] = rulebook

# Kernel bandwidths
H_X  = 0.15   # feet
H_ZN = 0.07   # normalized z units

# Bootstrap
N_BOOT = 100
RNG = np.random.default_rng(42)

# Plate edges in feet (for reference rectangle in plots)
PLATE_HALF_W = 17 / 2 / 12         # 17" plate, half-width in feet ≈ 0.708
BALL_RADIUS  = 1.45 / 12           # ≈ 0.121 ft
RULEBOOK_X_HALF = PLATE_HALF_W + BALL_RADIUS   # ≈ 0.83 ft

# Display aspect for contour/heatmap plots: 1 z_norm unit renders 1.1× the
# width of the 17" plate, matching the real-life zone proportions of a 6'0"
# batter (whose zone is ~19.08" tall × 17" wide).
DISPLAY_ASPECT = 1.2 * 17 / 12     # ≈ 1.558, y-unit per x-unit visual ratio


# %% load + clean

def clean(df):
    df = df.copy()
    df = df[df["description"].isin(["called_strike", "ball", "blocked_ball"])]
    df["is_strike"] = (df["description"] == "called_strike").astype(int)
    # drop bogus sz values (a handful of zeros etc.)
    df = df[(df["sz_top"] > 1.5) & (df["sz_bot"] > 0.5) & (df["sz_top"] > df["sz_bot"])]
    # drop bogus locations
    df = df[df["plate_x"].abs() < 4]
    df = df[(df["plate_z"] > -1) & (df["plate_z"] < 6)]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.reset_index(drop=True)


df25 = clean(pd.read_csv(DATA_DIR / "2025_zone_data.csv"))
df26 = clean(pd.read_csv(DATA_DIR / "2026_zone_data.csv"))
print(f"After cleaning: 2025 n={len(df25):,}, 2026 n={len(df26):,}")


# %% matched-batter set + sz substitution

# 2026 sz_top/sz_bot is constant within batter (height-based). Build a
# canonical per-batter zone reference.
batter_height_zone = (
    df26.groupby("batter")[["sz_top", "sz_bot"]]
    .median()
    .rename(columns={"sz_top": "sz_top_h", "sz_bot": "sz_bot_h"})
)
matched_batters = batter_height_zone.index
print(f"Batters in 2026 (matched set): {len(matched_batters):,}")


def apply_height_sz(df, year):
    """Restrict to matched batters; substitute 2026 height-based sz."""
    out = df.merge(batter_height_zone, on="batter", how="inner")
    out["sz_top"] = out["sz_top_h"]
    out["sz_bot"] = out["sz_bot_h"]
    return out.drop(columns=["sz_top_h", "sz_bot_h"])


if USE_HEIGHT_BASED_SZ:
    df25_a = apply_height_sz(df25, 2025)
    df26_a = apply_height_sz(df26, 2026)
else:
    # year-specific sz_top/sz_bot (robustness check)
    df25_a = df25.copy()
    df26_a = df26.copy()

print(
    f"After matching: 2025 n={len(df25_a):,} "
    f"({len(df25_a)/len(df25):.1%}), "
    f"2026 n={len(df26_a):,} "
    f"({len(df26_a)/len(df26):.1%})"
)


# %% calendar window

if RESTRICT_CALENDAR:
    day_min = df26_a["game_date"].dt.dayofyear.min()
    day_max = df26_a["game_date"].dt.dayofyear.max()
    df25_a = df25_a[
        df25_a["game_date"].dt.dayofyear.between(day_min, day_max)
    ].copy()
    print(f"Calendar window day-of-year [{day_min}, {day_max}]")
    print(f"After window: 2025 n={len(df25_a):,}")


# %% normalize vertical

def add_znorm(df):
    df = df.copy()
    df["z_norm"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    return df


df25_a = add_znorm(df25_a)
df26_a = add_znorm(df26_a)


# %% kernel-smoothed strike probability surface

def strike_prob_grid(df, x_grid=X_GRID, zn_grid=ZN_GRID, h_x=H_X, h_zn=H_ZN, x_col="plate_x"):
    """Bin-then-smooth estimator of P(strike | x, z_norm) on a regular grid.

    Bin pitches into the grid, then Gaussian-smooth (n_strikes) and
    (n_pitches) separately and divide. Equivalent to Nadaraya-Watson
    with a Gaussian kernel when bin width is small relative to bandwidth.
    Returns (P, N) where P is smoothed strike probability and N is
    smoothed pitch density (for density-at-contour diagnostics).

    `x_col` selects the horizontal coordinate; switch to "plate_x_inside"
    for handedness-mirrored analyses where + = inside, − = outside."""
    nx, nz = len(x_grid), len(zn_grid)
    dx = x_grid[1] - x_grid[0]
    dz = zn_grid[1] - zn_grid[0]

    x = df[x_col].to_numpy()
    z = df["z_norm"].to_numpy()
    y = df["is_strike"].to_numpy().astype(float)

    ix = ((x - x_grid[0]) / dx).round().astype(int)
    iz = ((z - zn_grid[0]) / dz).round().astype(int)
    keep = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    ix, iz, y = ix[keep], iz[keep], y[keep]

    n_grid = np.zeros((nx, nz))
    s_grid = np.zeros((nx, nz))
    np.add.at(n_grid, (ix, iz), 1.0)
    np.add.at(s_grid, (ix, iz), y)

    sigma_x = h_x / dx
    sigma_z = h_zn / dz
    n_smooth = gaussian_filter(n_grid, sigma=(sigma_x, sigma_z), mode="constant")
    s_smooth = gaussian_filter(s_grid, sigma=(sigma_x, sigma_z), mode="constant")

    P = s_smooth / np.maximum(n_smooth, 1e-9)
    # Mask cells with negligible support so the contour doesn't wander into noise
    P = np.where(n_smooth < 0.5, np.nan, P)
    return P, n_smooth


print("Fitting 2025 surface...")
P25, W25 = strike_prob_grid(df25_a)
print("Fitting 2026 surface...")
P26, W26 = strike_prob_grid(df26_a)


# %% edge metrics

def zone_metrics(P, x_grid=X_GRID, zn_grid=ZN_GRID, level=0.5):
    """Edges at the level contour:
       top/bot at x=0; inside/outside at z_norm=0.5.
       Area is the gridded count above level (in ft × z_norm units)."""
    P = np.where(np.isnan(P), 0.0, P)
    ix0 = np.argmin(np.abs(x_grid))
    z_col = P[ix0, :]
    above = z_col >= level
    z_top = zn_grid[above].max() if above.any() else np.nan
    z_bot = zn_grid[above].min() if above.any() else np.nan

    iz_mid = np.argmin(np.abs(zn_grid - 0.5))
    x_row = P[:, iz_mid]
    above = x_row >= level
    x_right = x_grid[above].max() if above.any() else np.nan
    x_left  = x_grid[above].min() if above.any() else np.nan

    dx = x_grid[1] - x_grid[0]
    dz = zn_grid[1] - zn_grid[0]
    area = float((P >= level).sum() * dx * dz)

    return dict(
        z_top=z_top, z_bot=z_bot,
        x_left=x_left, x_right=x_right,
        width=x_right - x_left,
        height=z_top - z_bot,
        area=area,
    )


m25 = zone_metrics(P25)
m26 = zone_metrics(P26)
print("\n=== Point estimates ===")
print(pd.DataFrame({"2025": m25, "2026": m26, "Δ": {k: m26[k] - m25[k] for k in m25}}))


# %% bootstrap by game

def bootstrap_by_game(df, rng):
    games = df["game_pk"].unique()
    sampled = rng.choice(games, size=len(games), replace=True)
    groups = df.groupby("game_pk").indices
    parts = [groups[g] for g in sampled]
    return df.iloc[np.concatenate(parts)]


def bootstrap_metrics(df, n_boot=N_BOOT, label="", x_col="plate_x"):
    rows = []
    for b in range(n_boot):
        boot = bootstrap_by_game(df, RNG)
        P, _ = strike_prob_grid(boot, x_col=x_col)
        rows.append(zone_metrics(P))
        if (b + 1) % 10 == 0:
            print(f"  {label} boot {b+1}/{n_boot}")
    return pd.DataFrame(rows)


print("\nBootstrapping 2025...")
boot25 = bootstrap_metrics(df25_a, label="2025")
print("Bootstrapping 2026...")
boot26 = bootstrap_metrics(df26_a, label="2026")


def summarize(boot, label):
    return pd.DataFrame({
        f"{label}_mean": boot.mean(),
        f"{label}_lo":   boot.quantile(0.025),
        f"{label}_hi":   boot.quantile(0.975),
    })


summary = pd.concat([summarize(boot25, "2025"), summarize(boot26, "2026")], axis=1)

# delta CI
delta = boot26.reset_index(drop=True) - boot25.reset_index(drop=True)
summary["delta_mean"] = delta.mean()
summary["delta_lo"]   = delta.quantile(0.025)
summary["delta_hi"]   = delta.quantile(0.975)

print("\n=== Bootstrap summary ===")
print(summary.round(4))
summary.to_csv(OUT_DIR / "summary.csv")


# %% plots

def plot_contours(P25, P26, fname=OUT_DIR / "contours.png"):
    fig, ax = plt.subplots(figsize=(7, 7))
    cs25 = ax.contour(X_GRID, ZN_GRID, P25.T, levels=[0.5], colors="C0")
    cs26 = ax.contour(X_GRID, ZN_GRID, P26.T, levels=[0.5], colors="C3")
    h25 = cs25.legend_elements()[0][0]
    h26 = cs26.legend_elements()[0][0]
    # rulebook reference rectangle
    ax.plot(
        [-RULEBOOK_X_HALF, RULEBOOK_X_HALF, RULEBOOK_X_HALF, -RULEBOOK_X_HALF, -RULEBOOK_X_HALF],
        [0, 0, 1, 1, 0],
        "k--", alpha=0.5, label="rulebook (height-based)",
    )
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("z_norm  (0=bot, 1=top of rulebook zone)")
    ax.set_title("50% called-strike contour, 2025 vs 2026")
    ax.legend([h25, h26, ax.lines[0]], ["2025", "2026", "rulebook"])
    ax.set_aspect(DISPLAY_ASPECT)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def plot_heatmaps(P25, P26, fname=OUT_DIR / "heatmaps.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, P, title in zip(axes, [P25, P26], ["2025", "2026"]):
        im = ax.imshow(
            P.T, origin="lower",
            extent=[X_GRID[0], X_GRID[-1], ZN_GRID[0], ZN_GRID[-1]],
            aspect=DISPLAY_ASPECT, vmin=0, vmax=1, cmap="RdBu_r",
        )
        ax.contour(X_GRID, ZN_GRID, P.T, levels=[0.5], colors="k")
        ax.plot(
            [-RULEBOOK_X_HALF, RULEBOOK_X_HALF, RULEBOOK_X_HALF, -RULEBOOK_X_HALF, -RULEBOOK_X_HALF],
            [0, 0, 1, 1, 0], "k--", alpha=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("plate_x (ft)")
    axes[0].set_ylabel("z_norm")
    fig.colorbar(im, ax=axes, label="P(called strike)", shrink=0.8)
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fname


plot_contours(P25, P26)
plot_heatmaps(P25, P26)
print(f"\nSaved plots to {OUT_DIR}/")


# %% pitch density at contour (sanity)

def density_at_contour(P, W, level=0.5, band=0.05):
    """Total kernel weight at grid cells where |P - level| < band.
    Compares 'how much data we have on the borderline' between years."""
    mask = np.abs(P - level) < band
    return float(W[mask].sum())


d25 = density_at_contour(P25, W25)
d26 = density_at_contour(P26, W26)
# normalize per pitch in the dataset to compare apples-to-apples
print(f"\nDensity at 50% contour: 2025 {d25/len(df25_a):.3f} per pitch, 2026 {d26/len(df26_a):.3f} per pitch")


# %% placebo: split 2025 in half, run same pipeline

print("\n=== Placebo (2025 split in half by date) ===")
median_date = df25_a["game_date"].median()
df25_h1 = df25_a[df25_a["game_date"] <= median_date]
df25_h2 = df25_a[df25_a["game_date"]  > median_date]
print(f"H1 n={len(df25_h1):,}, H2 n={len(df25_h2):,}")

P_h1, _ = strike_prob_grid(df25_h1)
P_h2, _ = strike_prob_grid(df25_h2)
m_h1 = zone_metrics(P_h1)
m_h2 = zone_metrics(P_h2)
print(pd.DataFrame({"2025_H1": m_h1, "2025_H2": m_h2,
                    "Δ_placebo": {k: m_h2[k] - m_h1[k] for k in m_h1}}))


# %% handedness-stratified analysis
#
# Statcast plate_x is from the catcher's view: positive = catcher's right
# (third-base side). For a RHB standing on the third-base side, that's the
# INSIDE corner; for a LHB standing on the first-base side, the OUTSIDE
# corner. To compare "inside vs outside" rather than "third-base vs
# first-base", mirror plate_x for LHB so positive = inside for both.

def add_inside_x(df):
    df = df.copy()
    df["plate_x_inside"] = np.where(df["stand"] == "L", -df["plate_x"], df["plate_x"])
    return df


df25_a = add_inside_x(df25_a)
df26_a = add_inside_x(df26_a)


def analyze_pair(df_2025, df_2026, label, x_col="plate_x_inside", n_boot=N_BOOT):
    """Fit surfaces and bootstrap deltas for a (year-2025, year-2026) pair."""
    print(f"\n--- {label} (n: 2025={len(df_2025):,}, 2026={len(df_2026):,}) ---")
    P25, _ = strike_prob_grid(df_2025, x_col=x_col)
    P26, _ = strike_prob_grid(df_2026, x_col=x_col)
    m25, m26 = zone_metrics(P25), zone_metrics(P26)
    print("Point estimates:")
    print(pd.DataFrame({"2025": m25, "2026": m26,
                        "Δ": {k: m26[k] - m25[k] for k in m25}}).round(4))

    print(f"  Bootstrapping {label} 2025...")
    b25 = bootstrap_metrics(df_2025, n_boot=n_boot, label=f"{label}-2025", x_col=x_col)
    print(f"  Bootstrapping {label} 2026...")
    b26 = bootstrap_metrics(df_2026, n_boot=n_boot, label=f"{label}-2026", x_col=x_col)
    delta = b26.reset_index(drop=True) - b25.reset_index(drop=True)
    print("Δ 95% CI (bootstrap):")
    print(pd.DataFrame({
        "mean": delta.mean(),
        "lo":   delta.quantile(0.025),
        "hi":   delta.quantile(0.975),
    }).round(4))
    return P25, P26, b25, b26, delta


# Split and run
df25_R = df25_a[df25_a["stand"] == "R"]
df25_L = df25_a[df25_a["stand"] == "L"]
df26_R = df26_a[df26_a["stand"] == "R"]
df26_L = df26_a[df26_a["stand"] == "L"]

P25R, P26R, b25R, b26R, dR = analyze_pair(df25_R, df26_R, "RHB")
P25L, P26L, b25L, b26L, dL = analyze_pair(df25_L, df26_L, "LHB")


# Plot both handedness contours side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
for ax, (P25_h, P26_h, hand) in zip(
    axes,
    [(P25R, P26R, "RHB"), (P25L, P26L, "LHB")],
):
    cs25 = ax.contour(X_GRID, ZN_GRID, P25_h.T, levels=[0.5], colors="C0")
    cs26 = ax.contour(X_GRID, ZN_GRID, P26_h.T, levels=[0.5], colors="C3")
    h25 = cs25.legend_elements()[0][0]
    h26 = cs26.legend_elements()[0][0]
    ax.plot(
        [-RULEBOOK_X_HALF, RULEBOOK_X_HALF, RULEBOOK_X_HALF, -RULEBOOK_X_HALF, -RULEBOOK_X_HALF],
        [0, 0, 1, 1, 0], "k--", alpha=0.5,
    )
    ax.set_xlabel("plate_x_inside (ft);  + = inside, − = outside")
    ax.set_title(hand)
    ax.set_aspect(DISPLAY_ASPECT)
    ax.grid(alpha=0.3)
    ax.legend([h25, h26], ["2025", "2026"])
axes[0].set_ylabel("z_norm")
fig.suptitle("50% called-strike contour by batter handedness, 2025 vs 2026")
fig.tight_layout()
fig.savefig(OUT_DIR / "contours_by_hand.png", dpi=140)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/contours_by_hand.png")


# Save handedness summary
hand_summary = pd.concat([
    dR.agg(["mean"]).rename(index={"mean": "RHB_Δ_mean"}),
    dR.quantile([0.025, 0.975]).rename(index={0.025: "RHB_Δ_lo", 0.975: "RHB_Δ_hi"}),
    dL.agg(["mean"]).rename(index={"mean": "LHB_Δ_mean"}),
    dL.quantile([0.025, 0.975]).rename(index={0.025: "LHB_Δ_lo", 0.975: "LHB_Δ_hi"}),
]).T
hand_summary.to_csv(OUT_DIR / "summary_by_hand.csv")
print("Saved summary_by_hand.csv")