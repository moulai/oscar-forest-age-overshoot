#!/usr/bin/env python3
"""Draw ED3 from completed historical simulations; no model integration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr

WIDTH_MM = 183
HEIGHT_MM = 174
SCREEN_PPI = 96
INK, MUTED, BLUE = '#26353C', '#6B777D', '#286779'
AGE_COLORS = ['#CDD3D5', '#6D9D9B', '#A5C6CD', '#C78B55']
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': 6.6, 'axes.titlesize': 7,
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
})


def axes_mm(fig, x, bottom, width, height):
    return fig.add_axes([x/WIDTH_MM, bottom/HEIGHT_MM, width/WIDTH_MM, height/HEIGHT_MM])


def panel(fig, x, top, letter, title):
    fig.text(x/WIDTH_MM, top/HEIGHT_MM, letter, fontsize=8, fontweight='bold', va='bottom')
    fig.text((x+4.3)/WIDTH_MM, top/HEIGHT_MM, title, fontsize=7, va='bottom')


def q(a):
    return np.quantile(a, [.05,.25,.5,.75,.95])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-dir', type=Path, default=Path('outputs/age_module/main_figure_redesign/figures/ed03_history_age_response/source_data'))
    ap.add_argument('--output', type=Path, default=Path('outputs/age_module/main_figure_redesign/figures/ed03_history_age_response'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.analysis_dir/'source_history_paired_all500.nc'
    ds = xr.load_dataset(source)
    if ds.sizes['config'] != 500:
        raise ValueError('All original 500 historical members are required.')
    data = ds.sel(driver_mode='emissions')
    delta = data.sel(production='H1',drop=True)-data.sel(production='H0',drop=True)
    end = delta.sel(year=2022)
    src = args.output/'source_data'
    src.mkdir(exist_ok=True)
    preview = args.output/'preview'; preview.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), facecolor='white')
    a = axes_mm(fig, 16, 108, 72, 54)
    b = axes_mm(fig, 113, 108, 62, 54)
    c = axes_mm(fig, 16, 20, 72, 56)
    d = axes_mm(fig, 113, 20, 62, 56)
    panel(fig, 3, 166, 'a', 'Historical forest age composition')
    panel(fig, 100, 166, 'b', 'Carbon stock differences in 2022')
    panel(fig, 3, 80, 'c', 'Net uptake through history')
    panel(fig, 100, 80, 'd', 'CO$_2$ and temperature differences in 2022')

    # Means preserve additivity in the composition panel; no member is discarded.
    order = ['primary_unaged','over_100','5_to_100','0_to_5']
    labels = ['Primary (no age)','>100 yr','5–100 yr','<5 yr']
    area = data.forest_age_area_Mha.sel(production='H1', age_support=order).mean('config')/1000
    a.stackplot(data.year, area.transpose('age_support','year').values,
                colors=AGE_COLORS, linewidth=0)
    a.set(xlim=(1750,2022), ylim=(0,5), xlabel='Calendar year', ylabel='Forest area (Gha)')
    a.set_xticks([1750,1850,1950,2022])
    a.set_yticks([0,1,2,3,4,5])
    a.legend(handles=[Patch(fc=v,ec='none',label=l) for v,l in zip(AGE_COLORS,labels)],
             loc='upper left', bbox_to_anchor=(-.02,-.25), ncol=2, fontsize=6.1,
             frameon=False, handlelength=1.1, columnspacing=1.1, handletextpad=.4)
    a.text(1774, 1.4, 'Primary\n(no age)', fontsize=6.2, color='#59656C', ha='left')
    pd.DataFrame({'year':data.year.values, **{s:area.sel(age_support=s).values for s in order}}).to_csv(src/'panel_a_age_area_Gha.csv',index=False)

    pools = [('live_carbon_PgC','Live vegetation'),('cwd_carbon_PgC','Coarse woody debris'),
             ('soil_carbon_PgC','Soil'),('product_carbon_PgC','Wood products'),
             ('land_carbon_PgC','All four pools')]
    vv = [end[name].values for name,_ in pools]
    positions = np.arange(len(pools))[::-1]
    bodies = b.violinplot(vv,positions=positions,vert=False,widths=.62,
                          showmeans=False,showmedians=False,showextrema=False,points=120)
    for body in bodies['bodies']:
        body.set_facecolor(BLUE);body.set_edgecolor(BLUE);body.set_linewidth(.4);body.set_alpha(.24)
    for y,values in zip(positions,vv):
        v=q(values)
        b.plot(v[[0,4]],[y,y],color=BLUE,lw=.8)
        b.plot(v[[1,3]],[y,y],color=BLUE,lw=2.7,solid_capstyle='butt')
        b.plot(v[2],y,'o',ms=3.5,mec='white',mew=.5,color=INK,zorder=4)
    b.axvline(0,color=MUTED,lw=.6,ls=(0,(2,2)),zorder=0)
    b.set_yticks(positions,[s for _,s in pools]);b.tick_params(axis='y',length=0,pad=4)
    b.set(xlabel='H1 − H0 (PgC)',xlim=(-80,15),ylim=(-.55,4.55))
    b.set_xticks([-80,-60,-40,-20,0])
    b.spines['left'].set_visible(False)
    pd.DataFrame([dict(member_position=i,source_config=int(data.source_config.values[i]),metric=name,value=float(v))
                  for (name,_),values in zip(pools,vv) for i,v in enumerate(values)]).to_csv(src/'panel_b_carbon_pool_differences.csv',index=False)

    years = np.arange(1751,2023)
    x = delta.cumulative_net_uptake_since_1750_PgC.sel(year=years).median('config').values
    y = delta.net_land_uptake_PgC_yr.sel(year=years).median('config').values
    ex = end.cumulative_net_uptake_since_1750_PgC.values
    ey = end.net_land_uptake_PgC_yr.values
    c.scatter(ex,ey,s=4,color=BLUE,alpha=.16,edgecolors='none',rasterized=False)
    c.axhline(0,color='#ADB5B9',lw=.55,zorder=0);c.axvline(0,color='#ADB5B9',lw=.55,zorder=0)
    c.plot(x,y,color=INK,lw=1.25,zorder=4)
    for index in (55,140,225):
        c.annotate('',xy=(x[index+10],y[index+10]),xytext=(x[index],y[index]),
                   arrowprops=dict(arrowstyle='->',mutation_scale=6,color=INK,lw=.8),zorder=5)
    for year,offset in ((1751,(5,4)),(1950,(-24,15)),(2022,(5,-12))):
        i=int(year-1751)
        c.plot(x[i],y[i],'o',ms=3.2,color=INK,mec='white',mew=.4,zorder=6)
        c.annotate(str(year),(x[i],y[i]),xytext=offset,textcoords='offset points',fontsize=6.2,zorder=7,
                   arrowprops=dict(arrowstyle='-',lw=.4,color=MUTED) if year==1950 else None)
    c.set(xlabel='Cumulative uptake change since 1750 (PgC)',
          ylabel='Annual net uptake change (PgC yr$^{-1}$)',xlim=(-69,8),ylim=(-.86,.15))
    c.set_xticks([-60,-40,-20,0])
    c.set_yticks([-.8,-.6,-.4,-.2,0])
    c.text(.03,.05,'Ensemble members in 2022',color=BLUE,fontsize=6.1,transform=c.transAxes)
    pd.DataFrame(dict(year=years,cumulative_delta_median_PgC=x,annual_delta_median_PgC_yr=y)).to_csv(src/'panel_c_historical_net_uptake_path.csv',index=False)

    co2 = end.co2_anomaly_ppm.values
    # Keep the original mK coordinates; only labels and annotations display °C.
    temp = end.temperature_raw_K.values*1000
    d.scatter(co2,temp,s=7,color=BLUE,alpha=.30,edgecolors='none',rasterized=False)
    d.axhline(0,color='#ADB5B9',lw=.55,zorder=0);d.axvline(0,color='#ADB5B9',lw=.55,zorder=0)
    xq,yq=q(co2),q(temp)
    d.errorbar(xq[2],yq[2],xerr=[[xq[2]-xq[0]],[xq[4]-xq[2]]],
               yerr=[[yq[2]-yq[0]],[yq[4]-yq[2]]],fmt='o',color=INK,ms=3.5,
               elinewidth=.85,capsize=2,mec='white',mew=.45,zorder=6)
    joint=int(((co2>0)&(temp>0)).sum())
    d.text(.05,.95,f'{joint/5:.1f}% jointly higher',transform=d.transAxes,
           ha='left',va='top',fontsize=6.1,color=BLUE)
    d.text(.05,.85,f'Medians: +{xq[2]:.1f} ppm\n+{yq[2]/1000:.3f} °C',transform=d.transAxes,
           ha='left',va='top',fontsize=6.1)
    d.set(xlabel='CO$_2$ difference in 2022 (ppm)',ylabel='Temperature difference in 2022 (°C)',
          xlim=(-3,20),ylim=(-20,215))
    d.set_xticks([0,5,10,15,20]);d.set_yticks([0,50,100,150,200])
    d.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v/1000:g}'.replace('-', '−')))
    pd.DataFrame(dict(member_position=np.arange(500),source_config=data.source_config.values,
                      cumulative_net_delta_PgC=ex,annual_net_delta_PgC_yr=ey,
                      CO2_delta_ppm=co2,temperature_raw_delta_mK=temp,
                      temperature_own_PI_delta_mK=end.temperature_PI_K.values*1000)).to_csv(src/'panels_c_d_2022_members_all500.csv',index=False)

    name='ed03_history_age_response'
    fig.savefig(args.output/(name+'.svg'),facecolor='white')
    fig.savefig(args.output/(name+'.pdf'),facecolor='white')
    fig.savefig(args.output/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(args.output/(name+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(preview/(name+'_final_size_183mm.png'),dpi=SCREEN_PPI,facecolor='white')
    plt.close(fig)
    manifest=dict(status='complete',figure='Extended Data Fig. 3',schema='ed03-history-figure-v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
                  source=str(source),source_schema=ds.attrs['schema'],source_provenance=json.loads(ds.attrs['provenance_json']),
                  shown_driver='emissions',archived_source_drivers=['emissions','concentration'],
                  paired_members=500,exclusions=0,figure_delta='H1 minus H0 within original ID',
                  panel_a_center='Member mean for additive stacked forest areas; primary unaged.',
                  panel_b_center='Median, thick interquartile range, thin empirical 5..95 interval; density includes all members.',
                  panel_c_center='Year-specific marginal medians of member-wise annual and cumulative differences; dots retain every 2022 pair.',
                  panel_d_center='All 500 paired differences; marginal median and empirical 5..95 crosshairs.',
                  temperature_reference='Unrecentered native D_Tg difference, equivalently any identical member-wise baseline subtracted from both arms. Own-history PI differences only retained in Source Data.',
                  joint_higher_co2_temperature_count=joint,
                  formats=['png_300dpi','svg_editable_text','pdf_fonttype42','tiff_600dpi','preview_png_96dpi'])
    (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='complete',output=str(args.output),joint_higher=joint,preview=str(preview/(name+'_final_size_183mm.png')))))


if __name__=='__main__':
    main()
