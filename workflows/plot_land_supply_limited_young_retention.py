#!/usr/bin/env python3
"""Render the supply-limited young-forest retention experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use('Agg')
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr

WIDTH_MM = 183
HEIGHT_MM = 251
SCREEN_PPI = 96
SCENARIOS = ('VL', 'LN')
CUTOFFS = (10, 20, 30, 40)
COLORS = {'VL': '#146E7C', 'LN': '#805C8A'}
INK, MUTED = '#253039', '#707C83'
GROUPS = ['0_to_5', '5_to_10', '10_to_20', '20_to_30', '30_to_40',
          '40_to_60', '60_to_100', '100_plus']
AGE_LABELS = ['<5', '5–10', '10–20', '20–30', '30–40', '40–60', '60–100', '≥100']
MARKERS = {10: 'o', 20: '^', 30: 'D', 40: 's'}
STEM = 'fig04_young_forest_retention'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': 6.6, 'axes.titlesize': 7,
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .55, 'xtick.major.width': .55, 'ytick.major.width': .55,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'legend.frameon': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def ax_mm(fig, x, top, w, h):
    return fig.add_axes([x / WIDTH_MM, 1 - (top+h) / HEIGHT_MM,
                         w / WIDTH_MM, h / HEIGHT_MM])


def text_mm(fig, x, top, value, **kwargs):
    return fig.text(x / WIDTH_MM, 1-top/HEIGHT_MM, value, va='top', **kwargs)


def panel(fig, x, top, letter, title):
    text_mm(fig, x, top, letter, fontsize=8, fontweight='bold')
    text_mm(fig, x+4.5, top+.1, title, fontsize=7)


def read_caption(path):
    text = path.read_text()
    return (text.split('## English caption', 1)[1] if '## English caption' in text else text).strip()+'\n'


def blend(color, amount):
    rgb=np.asarray(mpl.colors.to_rgb(color))
    return tuple(1-amount*(1-rgb))


def read_sources(root):
    members=pd.read_csv(root/'member_retention_outcomes_all500.csv')
    summaries=pd.read_csv(root/'condition_retention_summary.csv')
    transitions=pd.read_csv(root/'return_state_transitions.csv')
    report=json.loads((root/'summary.json').read_text())
    if report.get('status') != 'complete' or len(members) != 4000:
        raise ValueError('Use all eight completed original-500 conditions')
    if set(members.scenario) != set(SCENARIOS) or set(members.cutoff_years) != set(CUTOFFS):
        raise ValueError('Expected the matched VL/LN four-cutoff matrix')
    for _, g in members.groupby(['scenario','cutoff_years']):
        if len(g) != 500 or g.source_config.duplicated().any():
            raise ValueError('Every condition must retain its original 500 unique IDs')
    display=root/'annual_age_area_contrast.csv.gz'
    if display.exists():
        age=pd.read_csv(display)
    else:
        with xr.open_dataset(root/'source_annual_retention_all500.nc') as ds:
            rows=[]
            for bg in SCENARIOS:
                for cutoff in CUTOFFS:
                    a=ds.sel(cell=f'{bg}_protect_under_{cutoff}')
                    mean=a.age_area_Mha.mean('config')
                    base=a.baseline_age_area_Mha.mean('config')
                    for year in a.future_year.values:
                        for group in a.detailed_age_group.values:
                            p=float(mean.sel(future_year=year,detailed_age_group=group))
                            b=float(base.sel(future_year=year,detailed_age_group=group))
                            rows.append(dict(scenario=bg,cutoff_years=cutoff,year=int(year),
                                             age_group=str(group),policy_mean_area_Mha=p,
                                             baseline_mean_area_Mha=b,area_difference_Mha=p-b))
            age=pd.DataFrame(rows)
    return members,summaries,transitions,report,age


def select_summary(summaries, bg, cutoff, metric):
    rows=summaries.loc[summaries.scenario.eq(bg)&summaries.cutoff_years.eq(cutoff)&
                        summaries.population.eq('all_original_members')&summaries.metric.eq(metric)]
    if len(rows) != 1:
        raise ValueError(f'One canonical summary is required: {bg}, {cutoff}, {metric}')
    return rows.iloc[0]


def draw_rule(fig):
    panel(fig,4,4,'a','Young-forest protection without harvest displacement')
    a=ax_mm(fig,9,13,165,21);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    boxes=[(.005,.32,.215,.54,'Secondary forest\nage < a','#E2ECE6'),
           (.285,.32,.29,.54,'Cancel conversion and\nstanding-harvest proposals','#EAF0ED'),
           (.65,.32,.335,.54,'Retain forest\nNo replacement from older ages','#EDF1F3')]
    for x,y,w,h,label,color in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01,rounding_size=.02',fc=color,ec='none'))
        a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=6.5)
    for x0,x1 in [(.23,.275),(.585,.64)]:
        a.annotate('',xy=(x1,.59),xytext=(x0,.59),arrowprops=dict(arrowstyle='-|>',color=INK,lw=.9,mutation_scale=7))
    a.text(.5,.03,'a = 10, 20, 30 or 40 years',ha='center',va='bottom',fontsize=6.1,color=MUTED)


def draw_age_landscape(fig, members, age):
    panel(fig,4,41,'b','Forest area by age class under the 40-year rule')
    matrices=[]
    for bg in SCENARIOS:
        g=age.loc[age.scenario.eq(bg)&age.cutoff_years.eq(40)&age.age_group.isin(GROUPS)]
        matrices.append(g.pivot(index='age_group',columns='year',values='area_difference_Mha').loc[GROUPS])
    limit=np.ceil(max(np.abs(g.to_numpy()).max() for g in matrices)/50)*50
    cmap=LinearSegmentedColormap.from_list('signed_area',['#BA8762','#F8F8F5','#3D7882'])
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    for i,(bg,matrix) in enumerate(zip(SCENARIOS,matrices)):
        x=19+i*86
        a=ax_mm(fig,x,53,65,35)
        heat=a.pcolormesh(np.arange(2022.5,2100.5),np.arange(9)-.5,
                          matrix.to_numpy(),norm=norm,cmap=cmap,shading='flat',rasterized=False)
        a.set(xlim=(2022.5,2099.5),ylim=(-.5,7.5),xticks=[2023,2050,2075,2099],yticks=np.arange(8))
        a.set_yticklabels(AGE_LABELS if i==0 else [])
        a.tick_params(labelsize=5.7,pad=1.6,length=2)
        a.set_xlabel('Calendar year',fontsize=5.9,labelpad=2)
        if i==0:a.set_ylabel('Age class (yr)',fontsize=6)
        a.set_title(bg,fontsize=6.8,color=COLORS[bg],pad=4)
        a.axhline(4.5,color=INK,lw=.45,ls=(0,(2,2)))
        g=members.loc[members.scenario.eq(bg)&members.cutoff_years.eq(40)]
        forest=g.effect_forest_saved_area_2099_Mha.median()
        older=g.effect_age_resolved_atleast40_area_2099_Mha.median()
        text_mm(fig,x+32.5,100,f'2099: +{forest:.0f} Mha forest · +{older:.0f} Mha ≥40 yr',
                fontsize=5.8,color=COLORS[bg],ha='center')
    cax=ax_mm(fig,175,55,1.6,31)
    cb=fig.colorbar(heat,cax=cax,ticks=[-limit,0,limit])
    cb.ax.tick_params(labelsize=5.1,pad=1,length=1);cb.outline.set_linewidth(.35)
    cb.ax.set_title('Δ area\n(Mha)',fontsize=5.1,pad=4)
    cb.solids.set_rasterized(False)


def draw_carbon_climate(fig, members, summaries):
    panel(fig,4,112,'c','Uptake gain and remaining peak-warming difference')
    x=members.effect_cumulative_net_uptake_PgC.to_numpy()
    y=1000*members.remaining_window_max_K_vs_ageoff.to_numpy()
    xlim=(-3,np.ceil(x.max()/10)*10+3)
    ylim=(np.floor(y.min()/25)*25-5,np.ceil(y.max()/10)*10+5)
    for i,bg in enumerate(SCENARIOS):
        a=ax_mm(fig,19+i*86,124,65,48)
        a.axhline(0,color=MUTED,lw=.55,ls=(0,(2,2)))
        a.axvline(0,color=MUTED,lw=.55)
        centers=[]
        baseline=members.loc[members.scenario.eq(bg)&members.cutoff_years.eq(10)]
        reference=1000*baseline.original_age_response_window_max_K.to_numpy()
        r05,r50,r95=np.quantile(reference,[.05,.5,.95])
        a.plot([0,0],[r05,r95],color=INK,lw=1,zorder=5)
        a.plot(0,r50,'o',ms=4,color=INK,zorder=7)
        a.annotate('Unprotected',xy=(0,r50),xytext=(5,5),textcoords='offset points',fontsize=5.7,color=INK)
        for j,cutoff in enumerate(CUTOFFS):
            g=members.loc[members.scenario.eq(bg)&members.cutoff_years.eq(cutoff)]
            color=blend(COLORS[bg],[.38,.57,.78,1][j])
            a.scatter(g.effect_cumulative_net_uptake_PgC,1000*g.remaining_window_max_K_vs_ageoff,
                      s=3.5,alpha=.20,color=color,edgecolors='none',rasterized=False)
            cx=select_summary(summaries,bg,cutoff,'effect_cumulative_net_uptake_PgC')['median']
            cy=1000*g.remaining_window_max_K_vs_ageoff.median()
            centers.append((cx,cy))
        a.plot([0]+[v[0] for v in centers],[r50]+[v[1] for v in centers],color=COLORS[bg],lw=.8,zorder=5)
        for cutoff,(cx,cy) in zip(CUTOFFS,centers):
            a.plot(cx,cy,MARKERS[cutoff],ms=4.5,mfc='white',mec=COLORS[bg],mew=.8,zorder=6)
            a.annotate(str(cutoff),xy=(cx,cy),xytext=(6,1),textcoords='offset points',
                       fontsize=5.8,color=COLORS[bg],zorder=7)
        a.set(xlim=xlim,ylim=ylim,xlabel='Additional net land uptake (PgC)')
        a.set_yscale('asinh',linear_width=35)
        ticks=[v for v in [-200,-100,-50,0,50,100,200] if ylim[0]<=v<=ylim[1]]
        a.set_yticks(ticks);a.set_yticklabels([f'{v/1000:g}'.replace('-','−') for v in ticks])
        a.tick_params(labelsize=5.7,pad=1.5);a.xaxis.label.set_size(5.9)
        if i==0:a.set_ylabel('Peak warming relative to age-invariant reference (°C)',fontsize=5.5)
        a.set_title(bg,fontsize=6.8,color=COLORS[bg],pad=4)


def draw_delivery(fig, summaries):
    panel(fig,4,188,'d','Wood harvest and land-use trade-offs')
    specs=[('effect_Q_achieved_PgC',-1,'Standing-harvest delivery\nreduction (PgC)'),
           ('policy_nonforest_gross_unmet_Mha',1,'Non-forest activity\nnot executed (Mha gross)'),
           ('policy_nonforest_to_forest_unmet_Mha',1,'Of which forest\nestablishment (Mha gross)')]
    for i,(metric,sign,label) in enumerate(specs):
        a=ax_mm(fig,21+57*i,207,41,31)
        maximum=0
        for bi,bg in enumerate(SCENARIOS):
            for j,cutoff in enumerate(CUTOFFS):
                r=select_summary(summaries,bg,cutoff,metric)
                lo,med,hi=sorted([sign*r.q05,sign*r['median'],sign*r.q95])
                yy=j+(-.10 if bi==0 else .10)
                if metric=='effect_Q_achieved_PgC':a.plot([lo,hi],[yy,yy],color=COLORS[bg],lw=.7)
                a.plot(med,yy,MARKERS[cutoff],ms=3.3,mfc='white',mec=COLORS[bg],mew=.75)
                maximum=max(maximum,hi)
        a.set(xlim=(-.03*maximum,1.08*maximum),ylim=(3.5,-.5),yticks=range(4))
        a.set_yticklabels([str(c) for c in CUTOFFS] if i==0 else [])
        a.tick_params(labelsize=5.7,pad=1.8,length=2)
        if i==0:a.set_ylabel('Age limit (yr)',fontsize=6)
        a.set_title(label,fontsize=5.9,pad=5)
    handles=[mpl.lines.Line2D([],[],ls='none',marker='o',ms=3.3,mfc='white',mec=COLORS[b],mew=.75,label=b) for b in SCENARIOS]
    leg=ax_mm(fig,60,193,63,7);leg.axis('off')
    leg.legend(handles=handles,loc='center',ncol=2,fontsize=5.8,handlelength=1.0,
               handletextpad=.4,columnspacing=1.5)


def export(fig, args, members, summaries, transitions, report, age):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=out/'source_data';repro=out/'reproduction';preview=out/'preview'
    for p in (source,repro,preview):p.mkdir(exist_ok=True)
    fig.savefig(out/f'{STEM}.svg',facecolor='white')
    fig.savefig(out/f'{STEM}.pdf',facecolor='white')
    fig.savefig(out/f'{STEM}.png',dpi=300)
    fig.savefig(out/f'{STEM}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    with Image.open(out/f'{STEM}.png') as image:
        image.resize((round(WIDTH_MM/25.4*SCREEN_PPI),round(HEIGHT_MM/25.4*SCREEN_PPI)),
                     Image.Resampling.LANCZOS).save(preview/f'{STEM}_final_size_183mm.png',dpi=(SCREEN_PPI,SCREEN_PPI))
    caption=read_caption(args.caption_file);(out/'caption.md').write_text(caption)
    input_root=args.analysis_dir or args.source_bundle
    for name in ['member_retention_outcomes_all500.csv','condition_retention_summary.csv',
                 'return_state_transitions.csv','summary.json']:
        original=input_root/name
        if original.resolve() != (source/name).resolve():
            shutil.copy2(original,source/name)
    age.to_csv(source/'annual_age_area_contrast.csv.gz',index=False,compression='gzip')
    shutil.copy2(__file__,repro/'plot_land_supply_limited_young_retention.py')
    tree=ET.parse(out/f'{STEM}.svg');ns='{http://www.w3.org/2000/svg}'
    with Image.open(out/f'{STEM}.tiff') as im:
        pixels=list(im.size);dpi=[float(v) for v in im.info['dpi']]
    qa=dict(created_utc=datetime.now(timezone.utc).isoformat(),width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
            svg_text_elements=sum(1 for _ in tree.iter(ns+'text')),svg_raster_images=sum(1 for _ in tree.iter(ns+'image')),
            svg_text_editable=True,pdf_fonttype=42,png_dpi=300,tiff_dpi=dpi,tiff_pixel_dimensions=pixels,
            preview_dpi=SCREEN_PPI,source_analysis=str(args.analysis_dir or args.source_bundle),
            scientific_population='Eight distinct original-500 matched conditions; no pooling or selection',
            heatmap_scope='Both pathways, cutoff 40 only; all eight annual age contrasts retained in Source Data',
            centers='Condition-wise medians, connected for navigation; not individual member paths',
            area_variation='Area paths are determined by inputs and policy; no artificial ensemble uncertainty or jitter',
            climate_axis='All observed signed values retained; asinh y coordinate with 35 mK linear width',
            render_backend='Python matplotlib')
    (repro/'export_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    (repro/'reproduce_commands.txt').write_text(
        '.venv/bin/python -B reproduction/plot_land_supply_limited_young_retention.py '
        '--source-bundle source_data --caption-file caption.md --output regenerated\n')
    print(json.dumps(qa))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--analysis-dir',type=Path)
    source.add_argument('--source-bundle',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--caption-file',type=Path,required=True)
    args=p.parse_args();members,summaries,transitions,report,age=read_sources(args.analysis_dir or args.source_bundle)
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    draw_rule(fig);draw_age_landscape(fig,members,age)
    draw_carbon_climate(fig,members,summaries);draw_delivery(fig,summaries)
    export(fig,args,members,summaries,transitions,report,age)
    plt.close(fig)


if __name__=='__main__':
    main()
