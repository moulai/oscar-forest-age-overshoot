#!/usr/bin/env python3
"""Figure 3: local establishment responses and the actual overshoot-age landscape.

Uses completed formal sources only. No carbon or climate model is run here.
Regions are separate conditions, and original parameter members are never pooled,
resampled or selected by their response. Drawn with matplotlib only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch, FancyBboxPatch, Ellipse, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 7, "axes.labelsize": 6.6, "axes.titlesize": 7,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": .55,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": .55, "ytick.major.width": .55,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white"})

GROUPS = ["0_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_60", "60_to_100", "100_plus", "primary_unaged"]
MERGED = [("<5", ["0_to_5"]), ("5–20", ["5_to_10", "10_to_20"]),
          ("20–60", ["20_to_30", "30_to_40", "40_to_60"]),
          ("60–100", ["60_to_100"]), ("≥100", ["100_plus"]),
          ("Primary\n(no age)", ["primary_unaged"])]
AGE_COLORS = ["#C77F4D", "#E2BE83", "#A9C9B1", "#609E9D", "#355F70", "#D6DBDF"]
SCHEDULE_COLORS = {"pulse": "#626975", "distributed": "#338787"}
SCENARIO_COLORS = {"VL": "#146E7C", "LN": "#805C8A"}
WIDTH_MM = 183
HEIGHT_MM = 215
SCREEN_PREVIEW_DPI = 96  # Screen preview; formal PNG=300 dpi, TIFF=600 dpi.
REGION = "REST_ASIA"
EXPERIMENTS = (("L1", "pulse", True), ("L2", "pulse", False),
               ("L3", "distributed", True), ("L4", "distributed", False))


def quantiles(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if not a.size:
        return dict(n=0, mean=None, q05=None, median=None, q95=None, positive_n=0, negative_n=0)
    q = np.quantile(a, [.05, .5, .95])
    return dict(n=int(a.size), mean=float(np.mean(a)), q05=float(q[0]), median=float(q[1]), q95=float(q[2]),
                positive_n=int((a > 0).sum()), negative_n=int((a < 0).sum()))


def panel_title(fig, x, y, letter, label):
    fig.text(x, y, letter, fontsize=8, fontweight="bold", va="baseline")
    fig.text(x+.023, y, label, fontsize=7, va="baseline")


def read_sources(idealized, actual, bridge):
    lm = json.loads((idealized / "manifest.json").read_text())
    am = json.loads((actual / "manifest.json").read_text())
    bm = json.loads((bridge / "summary.json").read_text())
    if any(m["status"] != "complete" for m in (lm, am, bm)):
        raise ValueError("Only completed formal sources may be plotted")
    if am["scenarios"] != ["H", "HL", "M", "VL", "LN"]:
        raise ValueError("Complete five-path detailed-age source is required")
    flux = xr.load_dataset(idealized / "fluxes_all500.nc")
    macro = xr.load_dataset(bridge / "annual_contrasts_all500.nc")
    geometry = pd.read_csv(bridge / "landscape_production_geometry.csv")
    endpoints = pd.read_csv(bridge / "member_cumulative_net_endpoints.csv")
    return lm, am, bm, flux, macro, geometry, endpoints


def read_bundle(path):
    """The delivered small source bundle reproduces every plotted panel."""
    lm=json.loads((path/"source_local_manifest.json").read_text())
    am=json.loads((path/"source_actual_manifest.json").read_text())
    bm=json.loads((path/"source_bridge_manifest.json").read_text())
    if any(m["status"]!="complete" for m in (lm,am,bm)):raise ValueError("Incomplete bundled source")
    return (lm,am,bm,xr.load_dataset(path/"source_local_curves_all500.nc"),
        xr.load_dataset(path/"source_annual_contrasts_all500.nc"),
        pd.read_csv(path/"source_landscape_production_geometry.csv"),
        pd.read_csv(path/"source_member_cumulative_net_endpoints.csv"))


def copy_source(src,dst):
    if src.resolve()!=dst.resolve():shutil.copy2(src,dst)


def draw_schematic(fig):
    ax = fig.add_axes([.045, .838, .925, .095]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    # Shared initial carbon stocks are retained; this is a design diagram, not data.
    initial = FancyBboxPatch((.02,.35),.155,.40, boxstyle="round,pad=.012,rounding_size=.025",
                facecolor="#F0F0EA", edgecolor="#B3B5AA", lw=.7)
    ax.add_patch(initial)
    ax.text(.0925,.55,"1 ha pasture",ha="center",va="center",fontsize=7,fontweight="bold")
    for y in (.79,.21):
        ax.add_patch(FancyArrowPatch((.183,.55),(.265,y),arrowstyle="-|>",mutation_scale=8,lw=.8,color="#6B7077"))
    for schedule,y,label in (("pulse",.79,"Single (year 0)"),("distributed",.21,"0.05 ha yr$^{-1}$, years 0–19")):
        color=SCHEDULE_COLORS[schedule]
        ax.text(.277,y+.11,label,fontsize=6.5,color=color,fontweight="bold")
        # Area-time ribbons are drawn on a common 0–100-year physical scale.
        xx=np.r_[np.arange(20),100]/100*.37+.28
        hh=np.ones(21) if schedule=="pulse" else np.r_[np.arange(1,21)/20,1.]
        ax.fill_between(xx,y-.07,y-.07+.105*hh,step="post",color=color,alpha=.32,lw=0)
        ax.step(xx,y-.07+.105*hh,where="post",color=color,lw=.8)
        ax.text(.66,y-.035,"1 ha",fontsize=6,va="center",color=color)
        ax.add_patch(FancyArrowPatch((.72,y),(.77,y),arrowstyle="-|>",mutation_scale=8,lw=.8,color=color))
        ax.plot([.79,.84],[y+.035,y+.035],color=color,lw=1.4)
        ax.text(.852,y+.035,"Age on",fontsize=6.2,va="center")
        ax.plot([.79,.84],[y-.09,y-.09],color=color,lw=1.2,ls=(0,(3,1.5)))
        ax.text(.852,y-.09,"Age off",fontsize=6.2,va="center")
    for xx,label in ((.28,"0"),(.354,"20"),(.65,"100 years")):
        ax.text(xx,-.035,label,fontsize=5.8,ha="center",color=".35")


def curve_data(flux, experiment, metric, region):
    return flux[metric].sel(experiment=experiment,reg_land=region).transpose("relative_year","config").values


def draw_curves(fig, flux, source_rows):
    years=flux.relative_year.values
    for metric,pos,letter,title,ylabel in (
        ("forest_production",[.105,.620,.362,.150],"b","Net primary production","NPP (MgC initial ha$^{-1}$ yr$^{-1}$)"),
        ("cumulative_net_uptake",[.605,.620,.362,.150],"c","Cumulative net uptake","Cumulative net uptake\n(MgC ha$^{-1}$)")):
        ax=fig.add_axes(pos)
        panel_title(fig,pos[0]-.060,.797,letter,title)
        for experiment,schedule,on in EXPERIMENTS:
            a=curve_data(flux,experiment,metric,REGION)
            q=np.quantile(a,[.05,.95],axis=1); mean=a.mean(axis=1)
            yy=np.r_[0,years] if metric=="cumulative_net_uptake" else years
            mm=np.r_[0,mean] if metric=="cumulative_net_uptake" else mean
            ax.plot(yy,mm,color=SCHEDULE_COLORS[schedule],lw=1.4 if on else 1.0,
                    ls="-" if on else (0,(3.2,1.8)),label=f"{schedule} {'on' if on else 'off'}")
            for j,y in enumerate(years):
                source_rows.append(dict(region=REGION,experiment=experiment,metric=metric,relative_year=int(y),
                    mean=float(mean[j]),q05=float(q[0,j]),q95=float(q[1,j])))
        ax.axvline(20,color=".75",lw=.6,ls=(0,(2,2)),zorder=0)
        ax.set_xlim(0,100); ax.set_xticks([0,20,50,100]); ax.set_xlabel("Years since first establishment")
        ax.set_ylabel(ylabel,labelpad=5 if metric=="cumulative_net_uptake" else 3)
        ax.set_ylim(bottom=min(0,ax.get_ylim()[0]))
        fig.text(pos[0],.781,"Rest of Asia",fontsize=6.1,color=".3")
        # Paired contrast inset: complete member interval, not an interval over regions.
        inset=ax.inset_axes([.495,.16,.475,.32])
        inset.patch.set_facecolor("white"); inset.patch.set_alpha(.97)
        for schedule,on,off in (("pulse","L1","L2"),("distributed","L3","L4")):
            v=curve_data(flux,on,metric,REGION)-curve_data(flux,off,metric,REGION)
            q=np.quantile(v,[.05,.95],axis=1); mean=v.mean(axis=1)
            inset.fill_between(years,q[0],q[1],color=SCHEDULE_COLORS[schedule],alpha=.13,lw=0)
            inset.plot(years,mean,color=SCHEDULE_COLORS[schedule],lw=.85)
        inset.axhline(0,color=".5",lw=.55); inset.set_xlim(0,100); inset.set_xticks([0,100]);
        inset.tick_params(labelsize=5.5,length=2,pad=1.2)
        inset.text(.02,1.04,"Age on − off",transform=inset.transAxes,fontsize=5.7)
        inset.spines[["top","right"]].set_visible(False)
        inset.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        inset.set_xlabel("")
    # Four line identities, shared by the two absolute trajectory panels.
    handles=[Line2D([0],[0],color=SCHEDULE_COLORS[s],ls="-" if on else (0,(3.2,1.8)),lw=1.2,
                 label=f"{'Single' if s=='pulse' else 'Staggered'} · {'age on' if on else 'age off'}")
             for _,s,on in EXPERIMENTS]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.53,.563),ncol=4,frameon=False,
               handlelength=1.7,columnspacing=1.3,fontsize=6.1)


def regional_extended_data(flux, output):
    """All eleven regional conditions, separately; identical parameter summaries."""
    for metric,stem,ylabel in (("forest_production","ED_local_forest_production","NPP (MgC initial ha$^{-1}$ yr$^{-1}$)"),
                               ("cumulative_net_uptake","ED_local_cumulative_net_uptake","Cumulative net uptake\n(MgC ha$^{-1}$)")):
        fig,axes=plt.subplots(3,4,figsize=(WIDTH_MM/25.4,176/25.4))
        for ax,region in zip(axes.flat,flux.reg_land.values):
            for exp,schedule,on in EXPERIMENTS:
                a=curve_data(flux,exp,metric,region);q=np.quantile(a,[.05,.5,.95],axis=1);mean=a.mean(axis=1)
                yrs=flux.relative_year.values
                if metric=="cumulative_net_uptake":
                    yrs=np.r_[0,yrs];q=np.column_stack([np.zeros(3),q]);mean=np.r_[0,mean]
                if on: ax.fill_between(yrs,q[0],q[2],color=SCHEDULE_COLORS[schedule],alpha=.10,lw=0)
                ax.plot(yrs,mean,color=SCHEDULE_COLORS[schedule],ls="-" if on else (0,(3,1.8)),lw=.85)
            ax.axvline(20,color=".8",ls=(0,(2,2)),lw=.55);ax.set_title(str(region),loc="left",fontsize=6.5)
            ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.tick_params(labelsize=5.5)
        axes.flat[-1].axis("off")
        for si,(exp,schedule,on) in enumerate(EXPERIMENTS):
            axes.flat[-1].plot([],[],color=SCHEDULE_COLORS[schedule],ls="-" if on else "--",label=f"{'Single' if schedule=='pulse' else 'Staggered'} / age {'on' if on else 'off'}")
        axes.flat[-1].legend(frameon=False,loc="center",fontsize=6)
        fig.supxlabel("Years since first establishment",fontsize=7)
        fig.supylabel(ylabel,fontsize=7)
        fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.34,hspace=.50)
        for suffix in ("pdf","svg","png"):
            fig.savefig(output/f"{stem}.{suffix}",dpi=300)
        plt.close(fig)


def draw_area_production_bridge(fig, geometry):
    """A signed area-by-intensity mosaic; means of totals preserve products."""
    panel_title(fig,.045,.513,"d","Direct NPP change by age class")
    age_groups=("under_20","20_to_40","40_to_100","100_plus")
    labels=("<20 years","20–40 years","40–100 years","≥100 years")
    colors=("#BD8054","#DBBA88","#A0BFB4","#527F88")
    fig.legend(handles=[Patch(facecolor=c,edgecolor="none",label=l) for c,l in zip(colors,labels)],
        loc="center",bbox_to_anchor=(.55,.473),ncol=4,frameon=False,fontsize=5.9,
        handlelength=1.3,columnspacing=1.5)
    for scenario,xpos in (("VL",.105),("LN",.605)):
        ax=fig.add_axes([xpos,.350,.362,.102])
        g=geometry[geometry.scenario==scenario].set_index("age_class")
        left=0.
        for group,color in zip(age_groups,colors):
            row=g.loc[group];width=float(row.mean_area_Gha);height=float(row.height_MgC_ha_yr)
            ax.add_patch(Rectangle((left,0),width,height,facecolor=color,edgecolor="white",lw=.55))
            ax.plot([left,left],[0,.08],color=".4",lw=.5)
            left+=width
        ax.plot([left,left],[0,.08],color=".4",lw=.5)
        ax.axhline(0,color=".35",lw=.6)
        ax.set_xlim(0,2.18);ax.set_ylim(-2.9,.35)
        ax.set_xticks([0,.5,1,1.5,2]);ax.set_yticks([-2,-1,0])
        ax.set_xlabel("Age-resolved forest area (Gha)",labelpad=2.5)
        if scenario=="VL":ax.set_ylabel("Direct NPP change\n(MgC ha$^{-1}$ yr$^{-1}$)",labelpad=2.5)
        else:ax.tick_params(labelleft=False)
        ax.text(.01,1.01,scenario,ha="left",va="center",transform=ax.transAxes,
            color=SCENARIO_COLORS[scenario],fontweight="bold",fontsize=7)
        young=g.loc["under_20"]
        area_pct=100*float(young.fraction_of_mean_age_resolved_forest_area)
        npp_pct=100*float(young.fraction_of_mean_direct_adjustment)
        ax.text(.68,-1.65,f"<20 years\n{area_pct:.0f}% of area · {npp_pct:.0f}% of change",
                fontsize=6.0,color=".25",linespacing=1.4,va="center")
        ax.annotate("",xy=(float(young.mean_area_Gha)*.7,-1.65),xytext=(.63,-1.65),
                    arrowprops=dict(arrowstyle="->",color=".4",lw=.65))


def draw_macro_counterfactual(fig, macro, endpoints, bridge_manifest):
    panel_title(fig,.045,.272,"e","Role of cohorts established after 2022")
    handles=[Line2D([0],[0],color=".25",lw=1.35,label="Age dependence: all cohorts"),
             Line2D([0],[0],color=".25",lw=1.15,ls=(0,(3,1.8)),label="Age dependence: existing cohorts only")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.55,.250),ncol=2,frameon=False,
               fontsize=6,handlelength=1.8,columnspacing=2.2)
    for scenario,xpos in (("VL",.105),("LN",.605)):
        ax=fig.add_axes([xpos,.097,.362,.129])
        s=macro.sel(scenario=scenario);years=s.future_year.values
        total=s.cumulative_net_PgC.sel(contrast="total_on_minus_off").mean("config").values
        residual=s.cumulative_net_PgC.sel(contrast="remaining_existing_response").mean("config").values
        color=SCENARIO_COLORS[scenario]
        # The signed fill is the realized 11-10 conditional intervention, not an
        # allocation of mixed carbon pools by age or a no-afforestation comparison.
        ax.fill_between(years,total,residual,color=color,alpha=.13,lw=0)
        ax.plot(years,total,color=color,lw=1.4)
        ax.plot(years,residual,color=color,lw=1.0,ls=(0,(3,1.8)))
        ax.axhline(0,color=".5",lw=.6)
        ax.set_xlim(2023,2100);ax.set_xticks([2023,2060,2099]);ax.set_ylim(-31.5,1)
        ax.set_yticks([-30,-20,-10,0]);ax.set_xlabel("Calendar year",labelpad=2.3)
        if scenario=="VL":ax.set_ylabel("Cumulative net-uptake\ndifference (PgC)",labelpad=2)
        else:ax.tick_params(labelleft=False)
        ax.text(.02,.06,scenario,transform=ax.transAxes,fontweight="bold",color=color,fontsize=7)
        frac=bridge_manifest["scenarios"][scenario]["2099"]["fraction_of_mean_total"]
        ax.text(2079,(total[-1]+residual[-1])/2,f"{frac:.0%}",ha="center",va="center",fontsize=7,
                fontweight="bold",color=color,bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1.4))
        sub=endpoints[(endpoints.scenario==scenario)&(endpoints.contrast=="future_response_existing_on")]
        improvement=-sub.cumulative_net_PgC_to_2099.values
        q=quantiles(improvement)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--idealized-source",type=Path)
    p.add_argument("--actual-source",type=Path)
    p.add_argument("--bridge-source",type=Path)
    p.add_argument("--source-bundle",type=Path,help="Reproduce directly from a delivered Fig3 source bundle")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--caption",type=Path,help="English caption Markdown, the sole caption source")
    p.add_argument("--only-graphics",action="store_true",help="Re-export existing graphics without rewriting scientific Source Data")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data_output=args.output/"source_data"
    preview_output=args.output/"preview"
    extended_output=args.output/"extended_data"
    reproduction_output=args.output/"reproduction"
    for folder in (data_output,preview_output,extended_output,reproduction_output):
        folder.mkdir(parents=True,exist_ok=True)
    bundle_root=args.source_bundle
    if args.source_bundle and (args.source_bundle/"source_data").is_dir():
        args.source_bundle=args.source_bundle/"source_data"
    if args.source_bundle:
        lm,am,bm,flux,macro,geometry,endpoints=read_bundle(args.source_bundle)
    else:
        if any(x is None for x in (args.idealized_source,args.actual_source,args.bridge_source)):
            p.error("Provide --source-bundle or all three canonical source directories")
        lm,am,bm,flux,macro,geometry,endpoints=read_sources(args.idealized_source,args.actual_source,args.bridge_source)
    local_rows=[]
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    panel_title(fig,.045,.946,"a","Establishment schedules")
    draw_schematic(fig);draw_curves(fig,flux,local_rows)
    draw_area_production_bridge(fig,geometry)
    draw_macro_counterfactual(fig,macro,endpoints,bm)
    stem=args.output/"fig03_age_establishment_landscape"
    fig.savefig(str(stem)+".svg",dpi=600)
    fig.savefig(str(stem)+".pdf",dpi=600)
    fig.savefig(str(stem)+".png",dpi=300)
    fig.savefig(str(stem)+".tiff",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
    # Match the other main figures: downsample the formal PNG, do not render
    # a separately hinted low-DPI text image. Scientific/vector exports are unchanged.
    with Image.open(str(stem)+".png") as preview_source:
        preview_width=round(WIDTH_MM/25.4*SCREEN_PREVIEW_DPI)
        preview_height=round(preview_source.height*preview_width/preview_source.width)
        preview_source.resize((preview_width,preview_height),Image.Resampling.LANCZOS).save(
            preview_output/"fig03_age_establishment_landscape_final_size_183mm.png",dpi=(SCREEN_PREVIEW_DPI,SCREEN_PREVIEW_DPI))
    plt.close(fig)
    regional_extended_data(flux,extended_output)
    if not args.only_graphics:
        pd.DataFrame(local_rows).to_csv(data_output/"source_local_display_curves.csv",index=False)
        # Keep all original IDs, all 11 local regions and five actual futures; only
        # the declared VL/LN scientific subset is displayed in the landscape panels.
        local_thin=flux[["forest_production","net_uptake","cumulative_net_uptake"]]
        if not args.source_bundle or args.source_bundle.resolve()!=data_output.resolve():
            local_thin.to_netcdf(data_output/"source_local_curves_all500.nc",engine="h5netcdf",
                encoding={v:{"zlib":True,"complevel":4} for v in local_thin.data_vars})
        for name in ("landscape_production_source_all500.nc","member_landscape_production_geometry.csv.gz",
                     "landscape_production_geometry.csv","annual_contrasts_all500.nc",
                     "annual_contrast_summary.csv","member_cumulative_net_endpoints.csv",
                     "member_stage_net_contrasts.csv.gz","cumulative_endpoint_summary.csv","stage_contrast_summary.csv"):
            src=(args.source_bundle/("source_"+name)) if args.source_bundle else args.bridge_source/name
            copy_source(src,data_output/("source_"+name))
        curve_src=(args.source_bundle/"source_all_region_curve_summary.csv") if args.source_bundle else args.idealized_source/"analysis/curves_summary.csv"
        copy_source(curve_src,data_output/"source_all_region_curve_summary.csv")
        for name,obj in (("source_local_manifest.json",lm),("source_actual_manifest.json",am),("source_bridge_manifest.json",bm)):
            (data_output/name).write_text(json.dumps(obj,indent=2)+"\n")
    caption_path=args.caption or (args.output/"caption.md")
    if not caption_path.exists() and bundle_root:
        caption_path=(bundle_root if (bundle_root/"caption.md").exists() else bundle_root.parent)/"caption.md"
    caption=caption_path.read_text()
    if caption_path.resolve()!=(args.output/"caption.md").resolve():
        shutil.copy2(caption_path,args.output/"caption.md")
    shutil.copy2(__file__,reproduction_output/"plot_age_establishment_landscape.py")
    root=ET.parse(str(stem)+".svg").getroot();im=Image.open(str(stem)+".tiff")
    qa=dict(status="rendered_awaiting_actual_visual_review",width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
        tiff_pixels=list(im.size),tiff_dpi=[float(x) for x in im.info["dpi"]],
        svg_text_elements=len(root.findall(".//{http://www.w3.org/2000/svg}text")),
        svg_raster_elements=len(root.findall(".//{http://www.w3.org/2000/svg}image")),
        caption_words=len(caption.split()),main_actual_population={s:500 for s in ("VL","LN")},
        local_region=REGION,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (reproduction_output/"export_qa.json").write_text(json.dumps(qa,indent=2)+"\n")
    print(json.dumps({"status":"rendered","output":str(args.output),"qa":qa}))


if __name__=="__main__":
    main()
