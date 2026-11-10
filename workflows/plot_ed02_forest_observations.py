#!/usr/bin/env python3
"""Render Extended Data Fig. 2 from the current 2020 comparison's thin sources."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Require Arial before drawing; replacement fonts are not permitted.
for _style, _weight in (('normal', 'normal'), ('normal', 'bold'),
                        ('italic', 'normal'), ('italic', 'bold')):
    fm.findfont(fm.FontProperties(family='Arial', style=_style, weight=_weight),
                fallback_to_default=False)
mpl.rcParams.update({
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'mathtext.sf': 'Arial', 'mathtext.tt': 'Arial', 'mathtext.cal': 'Arial',
    'mathtext.fallback': None})
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr

STEM = "ed02_forest_observations_2020"
REGIONS = ["AFRICA", "CHINA+", "EUROPE", "INDIA+", "LATIN_AM", "MIDDLE_EAST",
           "NORTH_AM", "PAC_OECD", "REF_ECON", "REST_ASIA", "ROWO"]
NAMES = {"ROWO": "Other†", "AFRICA": "Africa", "CHINA+": "China+", "EUROPE": "Europe",
         "INDIA+": "India+", "LATIN_AM": "Latin America", "MIDDLE_EAST": "Middle East",
         "NORTH_AM": "North America", "PAC_OECD": "Pacific OECD",
         "REF_ECON": "Reforming economies", "REST_ASIA": "Rest of Asia"}
MODEL = "#33658a"
GAMI = "#47766c"
FRA = "#ad7046"
AGES = ["0_to_20", "21_to_40", "41_to_60", "61_to_80", "81_to_100", "over_100_resolved",
        "primary_unaged", "gami_mature_code300", "gami_unresolved_code299"]
AGE_NAMES = ["0–20", "21–40", "41–60", "61–80", "81–100", ">100, resolved",
             "Primary, no age", "GAMI mature class", "GAMI unresolved"]
AGE_COLORS = ["#e8bd80", "#dca367", "#c78a63", "#a37665", "#7b695f", "#57544f",
              "#d8dce1", "#b9b3cc", "#ffffff"]


def panel(ax, letter, heading, title_x=.055):
    ax.text(0, 1.08, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax.text(title_x, 1.08, heading, transform=ax.transAxes, fontsize=7, fontweight="normal", va="bottom")


def draw_map(ax, values, grid, vmax):
    dominant = grid.dominant_region_index.values
    field = np.full(dominant.shape, np.nan)
    for i, r in enumerate(grid.region.values.astype(str)):
        field[dominant == i] = values[r]
    norm = Normalize(0, vmax)
    cm = mpl.colormaps["YlGn"]
    lon, lat = grid.lon.values, grid.lat.values
    # Plot an existing geographic mask with constant values within each of the
    # eleven model regions, not a reconstructed grid-scale forest prediction.
    mesh = ax.pcolormesh(lon, lat, field, cmap=cm, norm=norm, shading="nearest", rasterized=True)
    for i in range(len(grid.region)):
        z = (dominant == i).astype(float)
        if z.any():
            ax.contour(lon, lat, z, levels=[.5], colors="#ffffff", linewidths=.25, alpha=.85)
    ax.set(xlim=(-180, 180), ylim=(-58, 85))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return mesh


def render(analysis: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source_data"
    reproduction = output / "reproduction"
    preview = output / "preview"
    for p in [source, reproduction, preview]:
        p.mkdir(exist_ok=True)
    for name in ["model_regional_members_all500.csv", "model_age_composition_all500.csv.gz",
                 "gami_regional_members_all20.csv", "gami_age_composition_all20.csv",
                 "fra2025_reported_2020_with_coverage.csv", "regional_display_statistics.csv",
                 "regional_map_geometry.nc", "gami_native_2020_reduced.nc", "summary.json", "manifest.json"]:
        src, dst = analysis / name, source / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    shutil.copy2(__file__, reproduction / Path(__file__).name)
    summary = json.loads((source / "summary.json").read_text())
    data = pd.read_csv(source / "regional_display_statistics.csv").set_index("region")
    ma = pd.read_csv(source / "model_age_composition_all500.csv.gz")
    ga = pd.read_csv(source / "gami_age_composition_all20.csv")
    with xr.open_dataset(source / "regional_map_geometry.nc") as d:
        grid = d.load()
    mpl.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 7, "axes.labelsize": 6.6, "axes.titlesize": 7,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "axes.linewidth": .55, "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "hatch.linewidth": .4})
    width_mm, height_mm = 183, 210
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    a = fig.add_axes([.065, .75, .415, .18])
    b = fig.add_axes([.545, .75, .415, .18])
    mvals = {r: data.loc[r, "model_forest_area_Mha_mean"] / 1000 for r in grid.region.values.astype(str)}
    gvals = {r: data.loc[r, "gami_forest_area_Mha_mean"] / 1000 for r in grid.region.values.astype(str)}
    top = max(max(mvals.values()), max(gvals.values()))
    vmax = np.ceil(top * 10) / 10
    draw_map(a, mvals, grid, vmax)
    mesh = draw_map(b, gvals, grid, vmax)
    panel(a, "a", "Model forest area")
    panel(b, "b", "GAMI forest area")
    cax = fig.add_axes([.30, .729, .40, .011])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=np.linspace(0, vmax, 4))
    cb.outline.set_linewidth(.4)
    cb.set_label("Regional forest area (Gha)", fontsize=6.6, labelpad=2)

    c = fig.add_axes([.14, .535, .82, .115])
    # Title offset matches the ~0.023 figure-width offset of panels a and b (axes c is wider).
    panel(c, "c", "Global age composition", title_x=.055 * .415 / .82)
    for frame, yy, label in [(ma, 1, "Model"), (ga, 0, "GAMI")]:
        means = frame[frame.region.eq("GLOBAL")].groupby("age_class").share_total_forest_pct.mean()
        left = 0.
        for k, group in enumerate(AGES):
            val = float(means.get(group, 0))
            if val == 0:
                continue
            hatch = "///" if group == "primary_unaged" else ".." if group == "gami_mature_code300" else None
            c.barh(yy, val, height=.45, left=left, color=AGE_COLORS[k],
                   edgecolor="white", linewidth=.5, hatch=hatch)
            if val > 14:
                t = "Primary,\nno age" if group == "primary_unaged" else "Mature\nclass" if group == "gami_mature_code300" else AGE_NAMES[k]
                c.text(left + val / 2, yy, t, fontsize=6.3, ha="center", va="center",
                       color="white" if k == 5 else "#20242a")
            left += val
    c.set(xlim=(0, 100), ylim=(-.5, 1.65), xlabel="Share of each source’s forest area (%)")
    c.set_yticks([0, 1], ["GAMI", "Model"])
    c.spines["left"].set_visible(False)
    c.tick_params(axis="y", length=0, pad=5)
    handles = [Patch(facecolor=AGE_COLORS[k], label=AGE_NAMES[k],
                     hatch="///" if k == 6 else ".." if k == 7 else None) for k in range(8)]
    c.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.01, -.49),
             ncol=4, fontsize=5.9, handlelength=1.4, columnspacing=1.1, handletextpad=.4)

    d = fig.add_axes([.17, .115, .31, .285])
    e = fig.add_axes([.615, .115, .345, .285])
    panel(d, "d", "Age-resolved young forest")
    panel(e, "e", "Living forest carbon")
    y = np.arange(len(REGIONS))[::-1]
    for yy, region in zip(y, REGIONS):
        r = data.loc[region]
        mv = r.model_known_young_share_total_forest_pct_median
        gv = r.gami_young_share_total_forest_pct_median
        if yy % 2 == 0:
            for ax in [d, e]:
                ax.axhspan(yy - .48, yy + .48, color="#f5f6f7", zorder=0)
        d.plot([mv, gv], [yy + .13, yy - .13], color="#c6cbd1", linewidth=.6, zorder=1)
        d.errorbar(mv, yy + .13, xerr=[[mv - r.model_known_young_share_total_forest_pct_q05],
                       [r.model_known_young_share_total_forest_pct_q95 - mv]],
                       fmt="o", color=MODEL, markersize=3.3, linewidth=.8, capsize=0)
        d.errorbar(gv, yy - .13, xerr=[[gv - r.gami_young_share_total_forest_pct_q05],
                       [r.gami_young_share_total_forest_pct_q95 - gv]],
                       fmt="s", color=GAMI, markersize=3.1, linewidth=.8, capsize=0)
        cv = r.model_live_carbon_PgC_median
        fv = r.fra_live_carbon_available_PgC
        e.plot([cv, fv], [yy + .13, yy - .13], color="#c6cbd1", linewidth=.6, zorder=1)
        e.errorbar(cv, yy + .13,
                   xerr=[[cv - r.model_live_carbon_PgC_q05], [r.model_live_carbon_PgC_q95 - cv]],
                   fmt="o", color=MODEL, markersize=3.3, linewidth=.8, capsize=0)
        complete = bool(np.isfinite(r.fra_live_carbon_complete_PgC))
        e.plot(fv, yy - .13, marker="D", markersize=3.1, markeredgewidth=.8,
               markerfacecolor=FRA if complete else "white", markeredgecolor=FRA, linestyle="none")
        if not complete:
            # The arrow encodes missing nonnegative country stocks, not an error bar.
            e.annotate("", xy=(fv + 3.0, yy - .13), xytext=(fv + .3, yy - .13),
                       arrowprops={"arrowstyle": "->", "color": FRA, "linewidth": .65, "mutation_scale": 5})
    for ax in [d, e]:
        ax.set_ylim(-.7, 10.7)
        ax.set_yticks(y, [NAMES[r] for r in REGIONS])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#dcdfe4", linewidth=.35, alpha=.7)
        ax.set_axisbelow(True)
    d.set(xlim=(0, 88), xlabel="Forest aged ≤20 yr (% of all forest)")
    d.set_xticks([0, 20, 40, 60, 80])
    carbon_upper = max(data.loc[REGIONS, "model_live_carbon_PgC_q95"].max(),
                       data.loc[REGIONS, "fra_live_carbon_available_PgC"].max())
    carbon_limit = 10 * np.ceil(carbon_upper * 1.06 / 10)
    e.set(xlim=(-2, carbon_limit), xlabel="Living carbon (PgC)")
    e.set_xticks([0, 40, 80, 120, 160])
    d.legend(handles=[Line2D([], [], color=MODEL, marker="o", linestyle="none", markersize=3.4, label="Model"),
                      Line2D([], [], color=GAMI, marker="s", linestyle="none", markersize=3.1, label="GAMI")],
             loc="upper left", bbox_to_anchor=(-.02, -.15), ncol=2, fontsize=6.2,
             columnspacing=1., handlelength=1., handletextpad=.4)
    e.legend(handles=[Line2D([], [], color=MODEL, marker="o", linestyle="none", markersize=3.4, label="Model W+F"),
                      Line2D([], [], color=FRA, marker="D", linestyle="none", markersize=3.1, label="FRA above- and below-ground carbon"),
                      Line2D([], [], color=FRA, marker="D", markerfacecolor="white", linestyle="none", markersize=3.1, label="FRA subtotal →")],
             loc="upper left", bbox_to_anchor=(-.02, -.15), ncol=1, fontsize=6.2,
             handlelength=1., handletextpad=.5, labelspacing=.45)
    # Translate the panel group upward by 5 mm to balance outer margins.
    fig.canvas.draw()
    for axis in fig.axes:
        axis.set_position(axis.get_position().translated(0, 5 / height_mm))
    fig.savefig(output / f"{STEM}.png", dpi=300)
    fig.savefig(output / f"{STEM}.pdf")
    fig.savefig(output / f"{STEM}.svg")
    fig.savefig(output / f"{STEM}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(output / f"{STEM}.png") as im:
        size = (round(width_mm / 25.4 * 96), round(height_mm / 25.4 * 96))
        im.resize(size, Image.Resampling.LANCZOS).save(preview / f"{STEM}_final_size_183mm.png", dpi=(96, 96))
    qa = {"status": "rendered", "width_mm": width_mm,
        "height_mm": height_mm, "formal_png_dpi": 300, "preview_dpi": 96, "tiff_dpi": 600,
        "text_font": mpl.font_manager.findfont(mpl.font_manager.FontProperties(family=["sans-serif"])),
        "style": "Regional observation comparisons with typography shared by the main figures.",
        "regions_in_quantitative_sources": len(REGIONS), "model_members_retained": 500,
        "gami_realizations_retained": 20, "fitted_parameters": 0,
        "map_scope": "Two regional-total choropleths on existing one-degree geography; map raster marks are intentional, labels remain editable.",
        "fra_marker_policy": "Solid diamond: complete regional country total; open diamond plus right arrow: available subtotal with missing countries. Not an observation uncertainty interval.",
        "global_forest_area_Gha": {"model": float(data.loc["GLOBAL", "model_forest_area_Mha_mean"] / 1000),
            "GAMI": float(data.loc["GLOBAL", "gami_forest_area_Mha_mean"] / 1000),
            "FRA": float(data.loc["GLOBAL", "fra_forest_area_Mha"] / 1000)},
        "source_model_identity": summary["model_run_identity"]}
    qa["carbon_axis_max_PgC"] = float(carbon_limit)
    qa["largest_displayed_carbon_q95_PgC"] = float(carbon_upper)
    qa["outer_margin_adjustment_mm"] = 5
    (reproduction / "export_QA.json").write_text(json.dumps(qa, indent=2) + "\n")
    plt.close(fig)
    print(output / f"{STEM}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis_dir.resolve(), args.output.resolve())
