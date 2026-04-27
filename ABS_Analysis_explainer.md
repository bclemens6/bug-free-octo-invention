# ABS_Analysis.py — A Plain-English Walkthrough

## What This Script Is Trying to Do

In 2026, MLB introduced ABS challenges — a system where batters and pitchers can contest called balls and strikes against the official rulebook zone, which is defined by batter height (27%–53% of height). The question this script answers is: **did the zone actually change?** And if so, how much and where?

It does this by taking pitch-by-pitch Statcast data from 2025 (no ABS challenges) and 2026 (with ABS challenges), building a mathematical picture of where called strikes tend to happen in each year, and then comparing those pictures precisely.

---

## Step 1: Setup and Configuration

**File:** lines 33–72

Before any data is touched, the script declares all its constants and settings in one place. The important ones:

- **`USE_HEIGHT_BASED_SZ = True`** — This is a key methodological choice. In 2026, the rulebook zone is defined by batter height and is constant for each player. The script uses that same height-based zone for 2025 too, so both years are measured on the same ruler. (Without this, you'd be comparing zones measured differently, which could create false differences.)
- **`RESTRICT_CALENDAR = True`** — 2025 data covers a full season, but 2026 data only covers however far we are into the season. This setting trims the 2025 data to only include games that fall within the same part of the calendar as the 2026 data. That way you're not comparing April vs. a full season.
- **`X_GRID` / `ZN_GRID`** — The analysis works by dividing the strike zone area into a fine grid (121×121 points). Think of it as a very fine mesh draped over the zone.
- **`H_X = 0.15` / `H_ZN = 0.07`** — These are "bandwidth" settings that control how much neighboring pitches influence each grid point during smoothing. More on this in Step 3.
- **`N_BOOT = 100`** — The number of bootstrap resamples used to calculate confidence intervals. More on this in Step 5.
- **`PLATE_HALF_W` / `RULEBOOK_X_HALF`** — The physical width of home plate (17 inches), plus the radius of a baseball, converted to feet. This defines the horizontal boundaries of the rulebook zone used as a reference in the plots.

---

## Step 2: Load and Clean the Data

**File:** lines 74–91

The script reads two CSV files — one for each year — containing one row per called pitch (called strikes, balls, and blocked balls). It immediately drops anything it doesn't need or trust:

- Pitches that weren't a called ball or called strike are removed (swings, HBP, etc.)
- Pitches with obviously broken zone measurements are dropped (e.g., a `sz_top` of zero, which would mean the top of the zone is at ground level)
- Pitches with locations so extreme they must be errors are dropped (horizontally more than 4 feet off center, vertically below the ground or 6 feet high)

Each remaining pitch gets a binary outcome: **1 if it was called a strike, 0 if it was called a ball.**

---

## Step 3: Match Batters and Standardize the Zone

**File:** lines 94–128

This is the most important methodological piece of the script.

The problem: umpires call strikes relative to each batter's actual height. A 5'9" player and a 6'4" player have different zones in absolute terms (feet off the ground), but both have the same *relative* zone (the rulebook says 27%–53% of their height). To compare 2025 and 2026 meaningfully, you need to put all those batters on a common scale.

**What it does:**
1. For every batter who appeared in 2026, it takes their median `sz_top` and `sz_bot` from 2026 data. Because 2026 uses the height-based rule, these are consistent per player.
2. It restricts *both* datasets to only batters who appeared in 2026 — the "matched set."
3. For 2025, it substitutes each batter's 2026 height-based zone measurements in place of whatever zone was recorded at the time.

This means 2025 and 2026 are now both measured against the *same* zone reference for the same players. Any differences you find are about umpire behavior, not zone-measurement differences.

---

## Step 4: Calendar Window

**File:** lines 133–140

Straightforward: it finds the earliest and latest day-of-year in the 2026 dataset, then trims 2025 to only include games within that same window. This prevents seasonal effects (e.g., umpire tendencies might differ in September) from confounding the comparison.

---

## Step 5: Normalize Vertical Location — z_norm

**File:** lines 145–152

This is the core coordinate transformation that makes cross-batter comparison possible.

Raw pitch height is in feet off the ground, which varies by batter height. After the zone substitution in Step 3, the script converts every pitch's vertical location into a **normalized score** called `z_norm`:

```
z_norm = (pitch_height - bottom_of_zone) / (top_of_zone - bottom_of_zone)
```

In this system:
- **z_norm = 0** is the bottom of the rulebook zone
- **z_norm = 1** is the top of the rulebook zone
- **z_norm = 0.5** is the middle of the zone
- Values above 1 or below 0 are outside the zone

Now a pitch that clips the top of the zone looks the same for every batter, regardless of their height. The horizontal coordinate (`plate_x`) stays in feet — it's already on a fixed scale.

---

## Step 6: Build the Strike Probability Surface (Nadaraya-Watson Kernel Smoothing)

**File:** lines 157–200

This is the statistical heart of the script. The goal is to estimate, for any (x, z_norm) location in the zone, what's the probability that a pitch there gets called a strike?

**How it works — the "bin-then-smooth" estimator:**

1. **Bin:** The script divides the zone into that 121×121 grid. Each pitch gets placed in the grid cell closest to its actual location. The script counts, for each cell: how many pitches landed there? How many were called strikes?

2. **Smooth (Gaussian kernel smoothing):** Raw bin counts are noisy — a cell might have 0 or 2 pitches just by chance. The script blurs both the pitch count and strike count across neighboring cells using a **Gaussian filter** (think of it like applying a blur to an image). It blurs horizontally by a "bandwidth" of 0.15 feet and vertically by 0.07 normalized units.

3. **Divide:** After smoothing, it divides smoothed-strikes by smoothed-pitches at each cell to get the estimated probability.

This technique is mathematically equivalent to **Nadaraya-Watson kernel regression** — a nonparametric method that estimates a smooth function from noisy data without assuming any particular shape for the zone (e.g., it doesn't force the zone to be a rectangle). The result is a probability surface: a heat map of "how likely is a called strike here?"

Cells with very few pitches are masked out (set to NaN) so the contour doesn't chase noise at the edges.

This runs for both 2025 and 2026, producing two surfaces: `P25` and `P26`.

---

## Step 7: Measure Zone Edges

**File:** lines 205–238

With the probability surfaces in hand, the script measures where each year's *effective* strike zone is. It does this by finding the **50% contour** — the line where a pitch has a 50/50 chance of being called a strike.

Specifically, it pulls four edge measurements:
- **Top edge:** highest z_norm where P = 0.5, measured directly above the center of the plate (x = 0)
- **Bottom edge:** lowest z_norm where P = 0.5, at x = 0
- **Inside/outside edges:** leftmost and rightmost x where P = 0.5, measured at mid-zone height (z_norm = 0.5)
- **Width, height, area:** derived from the above

These are reported as point estimates first, then made rigorous through bootstrapping in the next step.

---

## Step 8: Bootstrap Confidence Intervals

**File:** lines 243–286

Point estimates (Step 7) tell you where the zone edges are, but not whether the differences are meaningful or just noise. The script uses **clustered bootstrap resampling** to build 95% confidence intervals.

**How it works:**
1. Get the list of all unique games in the dataset.
2. Draw a new sample of games *with replacement* — the same game can appear multiple times, and some games will be absent. This is called **bootstrapping**.
3. Re-run the entire surface estimation and edge measurement on this resampled dataset.
4. Repeat 100 times.

**Why cluster by game?** Pitches within a game are correlated — the same umpire, the same weather, the same teams. Treating each pitch as independent would understate the true uncertainty. By resampling whole games, you respect that structure.

The output is:
- The mean and 95% confidence interval (2.5th–97.5th percentile) for each zone metric in each year
- The mean and 95% CI for the *difference* (2026 minus 2025) — the key result

If the confidence interval for a difference doesn't include zero, you have strong evidence that the zone genuinely changed.

Results are saved to `out/summary.csv`.

---

## Step 9: Visualize the Zones

**File:** lines 291–338

Two types of plots are saved to the `out/` folder:

**`contours.png`** — Overlays the 50% called-strike contour from 2025 (blue) and 2026 (red) on the same plot, plus a dashed box showing where the rulebook zone is. If the zone changed, the contours will be in different places.

**`heatmaps.png`** — Side-by-side color maps for each year. Blue = rarely called a strike, red = almost always called a strike. The 50% contour is drawn in black on top of each heatmap.

---

## Step 10: Sanity Check — Density at the Contour

**File:** lines 344–354

This is a diagnostic step. The 50% contour is where the action is — it defines the zone edge. But if there are very few pitches near that boundary (because pitchers avoid the edge, or there just aren't many borderline pitches), the estimated edge location will be unreliable.

This step measures how much "weight" of pitches falls near the 50% contour in each year, normalized per pitch. If one year has much less data near the boundary, you'd want to be more cautious about its edge estimates.

---

## Step 11: Placebo Test

**File:** lines 359–370

A **placebo test** (sometimes called a falsification test) checks whether your method produces false positives. Here's the logic:

If you split 2025 data into two halves by date and run the exact same pipeline comparing the two halves, you should find *no real differences* — it's all the same year, same umpires, same rules. If you *do* find big differences, that suggests your method has some artifact or the data is noisier than you thought.

The script splits 2025 at the median game date, produces surfaces for each half (`P_h1`, `P_h2`), and compares the zone metrics. Small, unsystematic differences are reassuring; large or structured ones would be a warning flag.

---

## Step 12: Handedness Stratification

**File:** lines 374–460

The final section repeats the entire analysis separately for right-handed and left-handed batters.

**Why this matters:** The inside and outside corners of the zone are structurally different. A pitch on the inner third of the plate to a right-handed batter is on the outer third to a left-handed batter (from the pitcher's perspective), and umpires have historically called them differently. ABS challenges might also affect inside vs. outside differently.

**The coordinate flip:** Statcast `plate_x` is from the catcher's perspective — positive = toward third base. That means positive is the *inside* corner for right-handed batters and the *outside* corner for left-handed batters. The script creates a new coordinate `plate_x_inside` that flips the sign for lefties, so that positive always means "inside" regardless of handedness.

With this fix, the same analysis runs on RHB and LHB subsets, and the resulting plots show "inside edge" and "outside edge" in comparable terms.

Results are saved to `out/summary_by_hand.csv` and plotted in `out/contours_by_hand.png`.

---

## Output Summary

| File | Contents |
|------|----------|
| `out/summary.csv` | Bootstrap means and 95% CIs for all zone metrics, both years, plus delta |
| `out/summary_by_hand.csv` | Same, split by batter handedness |
| `out/contours.png` | 50% contour overlay, 2025 vs 2026 |
| `out/heatmaps.png` | Strike probability heatmaps, side by side |
| `out/contours_by_hand.png` | Contour overlay, RHB and LHB panels |
