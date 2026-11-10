#!/usr/bin/env python3
"""Draw the climate-outcome and phase-carbon main figures."""
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

ALL_SCENARIOS = ('H', 'HL', 'M', 'VL', 'LN')
SCENARIOS = ('VL', 'LN')
COLORS = {'H': '#8C6D5A', 'HL': '#92934E', 'M': '#667C91',
          'VL': '#146E7C', 'LN': '#805C8A'}
INK, MUTED, LIGHT = '#253039', '#707C83', '#DCE1E4'
STATE_COLORS = {'never_above': '#C9D0D4', 'exceeded_returned': '#3F8886',
                'exceeded_still_above': '#D1AA73'}
STATES = tuple(STATE_COLORS)
WIDTH_MM = 183
SCREEN_PPI = 96

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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ax_mm(fig, x, top, w, h):
    height = fig.get_figheight() * 25.4
    return fig.add_axes([x / WIDTH_MM, 1 - (top + h) / height, w / WIDTH_MM, h / height])


def text_mm(fig, x, top, value, **kwargs):
    height = fig.get_figheight() * 25.4
    return fig.text(x / WIDTH_MM, 1 - top / height, value, va='top', **kwargs)


def panel(fig, x, top, letter, title):
    text_mm(fig, x, top, letter, fontsize=8, fontweight='bold')
    text_mm(fig, x + 4.5, top + .1, title, fontsize=7)


def balance_vertical_outer_margins(fig):
    """Translate the complete content without resizing any axes or altering data."""
    def visible_bounds():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [t.get_window_extent(renderer) for t in fig.texts
                  if t.get_visible() and t.get_text()]
        for axis in fig.axes:
            if not axis.get_visible():
                continue
            if axis.axison:
                bounds.append(axis.get_tightbbox(renderer))
            else:
                artists = list(axis.patches) + list(axis.lines) + list(axis.texts)
                if axis.get_legend() is not None:
                    artists.append(axis.get_legend())
                bounds.extend(a.get_window_extent(renderer) for a in artists if a.get_visible())
        bounds = [b for b in bounds if b is not None and np.isfinite(b.extents).all()
                  and b.width > 0 and b.height > 0]
        mm_per_pixel = 25.4 / fig.dpi
        return dict(top=(fig.bbox.height - max(b.y1 for b in bounds)) * mm_per_pixel,
                    bottom=min(b.y0 for b in bounds) * mm_per_pixel)

    before = visible_bounds()
    # Keep the fixed vertical placement of the complete climate-outcome panel group.
    down_mm = 4.105142361111102
    delta = down_mm / (fig.get_figheight() * 25.4)
    for axis in fig.axes:
        pos = axis.get_position()
        axis.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, y - delta))
    fig._vertical_outer_margins_mm = dict(before=before, content_shift_down=down_mm,
                                          after=visible_bounds())


def pathway_legend(ax, **kwargs):
    size = kwargs.pop('fontsize', 6.1)
    handles = [mpl.lines.Line2D([], [], color=COLORS[b], lw=1.3, label=b) for b in SCENARIOS]
    return ax.legend(handles=handles, fontsize=size, handlelength=1.2, handletextpad=.3,
                     columnspacing=.8, **kwargs)


def flow_patch(ax, x0, x1, y0, y1, z0, z1, color, alpha):
    vertices = [(x0, y0), (x0 + .28, y0), (x1 - .28, z0), (x1, z0),
                (x1, z1), (x1 - .28, z1), (x0 + .28, y1), (x0, y1), (x0, y0)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor='none', alpha=alpha))


def experiment_schematic(fig):
    panel(fig, 4, 4, 'a', 'Paired future simulations')
    ax = ax_mm(fig, 5, 8, 173, 21)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.add_patch(FancyBboxPatch((.005, .22), .255, .57, boxstyle='round,pad=.01,rounding_size=.025',
                              fc='#EDF1F2', ec='none'))
    ax.text(.132, .51, 'Shared 2022 forest\nand carbon state', ha='center', va='center', fontsize=6.8)
    ax.plot([.277, .355], [.51, .51], color=INK, lw=1.1)
    ax.plot(.355, .51, 'o', color=INK, ms=3)
    for y, label, color in [( .77, 'Age-dependent production (age on)', '#547F87'),
                             (.24, 'Age-invariant reference (age off)', '#9B8B7E')]:
        ax.annotate('', xy=(.495, y), xytext=(.365, .51),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, mutation_scale=8))
        ax.add_patch(FancyBboxPatch((.50, y-.17), .295, .34,
                                  boxstyle='round,pad=.01,rounding_size=.02', fc=color, alpha=.13, ec='none'))
        ax.text(.647, y, label, ha='center', va='center', fontsize=6.6, color=INK)
    ax.text(.815, .51, 'Same prescribed drivers\nand harvest requests\nVL and LN pathways',
            fontsize=6.1, va='center', color=MUTED)
    # Keep the fixed left edge of the complete schematic at 10.624125 mm.
    pos = ax.get_position()
    ax.set_position([10.624125 / WIDTH_MM, pos.y0, pos.width, pos.height])


def target_points(ax, members, limits, inset=False):
    lo, hi = limits
    ax.add_patch(Rectangle((lo, 1.5), 1.5-lo, hi-1.5, fc='#F6E5D8', ec='none', zorder=-3))
    ax.plot([lo, hi], [lo, hi], color='#AAB3B8', lw=.6, zorder=-2)
    ax.axvline(1.5, color='#B17F5B', lw=.55, ls=(0, (2, 2)), zorder=-1)
    ax.axhline(1.5, color='#B17F5B', lw=.55, ls=(0, (2, 2)), zorder=-1)
    for bg in SCENARIOS:
        g = members.loc[members.scenario.eq(bg)]
        ax.scatter(g.off_T2099_K, g.on_T2099_K, s=6 if not inset else 12,
                   facecolors='none', edgecolors=COLORS[bg], linewidths=.33,
                   alpha=.46 if not inset else .62, rasterized=False)
    lost = members.loc[members.lost_return_by_2099]
    ax.scatter(lost.off_T2099_K, lost.on_T2099_K, s=10 if not inset else 18,
               color='#B76C55', edgecolors='white', lw=.3, zorder=5, rasterized=False)
    ax.set(xlim=limits, ylim=limits)
    if inset:
        ax.set(xticks=[1.3, 1.5, 1.7], yticks=[1.3, 1.5, 1.7])
        ax.tick_params(labelsize=5.7, pad=1.5, length=2)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(.45)


def big_state_flow(ax, members, bg):
    g = members.loc[members.scenario.eq(bg)]
    counts = {a: g[a+'_state'].value_counts() for a in ('off', 'on')}
    bottoms = {a: {} for a in counts}
    for a in counts:
        total = 0.
        for state_index, state in enumerate(STATES):
            bottoms[a][state] = total + 10 * state_index
            total += float(counts[a].get(state, 0)) / 5
        require(abs(total-100) < 1e-10, 'State proportions must sum to 100%')
    offsets = {a: dict(bottoms[a]) for a in counts}
    for first in STATES:
        for last in STATES:
            n = int((g.off_state.eq(first) & g.on_state.eq(last)).sum())
            if not n:
                continue
            height = n / 5
            y0, y1 = offsets['off'][first], offsets['on'][last]
            loss = first == 'exceeded_returned' and last == 'exceeded_still_above'
            flow_patch(ax, .12, .88, y0, y0+height, y1, y1+height,
                       '#B76C55' if loss else STATE_COLORS[first], .82 if loss else .42)
            offsets['off'][first] += height
            offsets['on'][last] += height
    for a, x, ha, dx in [('off', .085, 'left', .046), ('on', .915, 'right', -.046)]:
        for state in STATES:
            height = float(counts[a].get(state, 0)) / 5
            ax.add_patch(Rectangle((x-.025, bottoms[a][state]), .05, height,
                                   fc=STATE_COLORS[state], ec='white', lw=.45, zorder=5))
            if height >= 4:
                ax.text(x+dx, bottoms[a][state]+height/2, f'{height:.1f}%',
                        ha=ha, va='center', fontsize=6.1, color=INK,
                        bbox=dict(fc='white', ec='none', alpha=.84, pad=.35), zorder=6)
    ax.set(xlim=(0, 1), ylim=(0, 120), xticks=[.085, .915], xticklabels=['Age off', 'Age on'])
    ax.set_yticks([]); ax.tick_params(axis='x', length=0, labelsize=6.3)
    for spine in ax.spines.values(): spine.set_visible(False)


def figure01(members, annual, transitions, summary):
    fig = plt.figure(figsize=(WIDTH_MM/25.4, 220/25.4))
    experiment_schematic(fig)
    panel(fig,4,36,'b','Median warming trajectories')
    a=ax_mm(fig,16,46,73,53)
    for bg in SCENARIOS:
        g=annual.loc[annual.scenario.eq(bg)].sort_values('year')
        a.plot(g.year,g.on_temperature_K_median,color=COLORS[bg],lw=1.4,label=bg)
        a.plot(g.year,g.off_temperature_K_median,color=COLORS[bg],ls=(0,(3,2)),lw=1)
    a.axhline(1.5,color=MUTED,lw=.6,ls=(0,(2,2)))
    median_columns=['on_temperature_K_median','off_temperature_K_median']
    lower=np.floor(annual[median_columns].min().min()*20)/20-.025
    upper=np.ceil(annual[median_columns].max().max()*20)/20+.025
    a.set(xlim=(2022,2099),ylim=(lower,upper),
          xticks=[2022,2060,2099],xlabel='Calendar year',ylabel='Warming from 1850–1900 (°C)')
    pathway_legend(a,loc='upper left',ncol=2)
    a.legend(handles=[mpl.lines.Line2D([], [], color=INK, lw=1.1, label='Age on'),
                      mpl.lines.Line2D([], [], color=INK, lw=1, ls=(0,(3,2)), label='Age off')],
             loc='upper left', bbox_to_anchor=(0,-.20), ncol=2, fontsize=5.9,
             handlelength=1.6, columnspacing=1.1)
    a.add_artist(mpl.legend.Legend(a,
                  [mpl.lines.Line2D([], [], color=COLORS[b], lw=1.3) for b in SCENARIOS],
                  list(SCENARIOS),loc='upper left',ncol=2,fontsize=6.1,frameon=False,
                  handlelength=1.2,handletextpad=.3,columnspacing=.8))

    panel(fig,98,36,'c','Warming in 2099')
    b=ax_mm(fig,112,46,67,62)
    lo=min(.2,float(members[['off_T2099_K','on_T2099_K']].min().min())-.1)
    hi=np.ceil(float(members[['off_T2099_K','on_T2099_K']].max().max())*2)/2+.1
    target_points(b,members,(lo,hi));b.set(xlabel='2099 warming, age off (°C)',ylabel='2099 warming, age on (°C)')
    b.add_patch(Rectangle((1.2,1.2),.6,.6,fc='none',ec=INK,lw=.65))
    ins=b.inset_axes([.09,.59,.39,.37]);target_points(ins,members,(1.2,1.8),inset=True)
    ins.set_title('Near 1.5 °C',fontsize=6,pad=2)

    panel(fig,4,126,'d','Return to 1.5 °C by 2099')
    for i,bg in enumerate(SCENARIOS):
        x=16+i*39
        axis=ax_mm(fig,x,138,32,52);big_state_flow(axis,members,bg)
        axis.set_title(bg,color=COLORS[bg],fontsize=7,fontweight='bold',pad=5)
        g=members.loc[members.scenario.eq(bg)]
        ref=int(g.off_state.eq('exceeded_returned').sum());loss=int(g.lost_return_by_2099.sum())
        text_mm(fig,x+16,197,f'Lost return: {100*loss/ref:.1f}%',fontsize=6.1,
                ha='center',color='#A05B47')

    panel(fig,98,126,'e','Peak warming and years above 1.5 °C')
    rawx=members.delta_window_max_K.to_numpy()*1000;rawy=members.delta_exceedance_years_77.to_numpy()
    xlo=np.floor(rawx.min()/10)*10-10;xhi=np.ceil(rawx.max()/10)*10+10
    xedges=np.arange(xlo,xhi+1,10);yedges=np.arange(rawy.min()-.5,rawy.max()+1.5,1)
    transform=lambda x:np.arcsinh(np.asarray(x)/30.)
    counts={};rows=[]
    for bg in SCENARIOS:
        g=members.loc[members.scenario.eq(bg)]
        h,_,_=np.histogram2d(g.delta_window_max_K*1000,g.delta_exceedance_years_77,bins=(xedges,yedges))
        require(h.sum()==500,'Joint bins must retain all original 500')
        counts[bg]=h
        for i,j in zip(*np.nonzero(h)):
            rows.append(dict(scenario=bg,delta_max_mK_left=xedges[i],delta_max_mK_right=xedges[i+1],
                             delta_threshold_years_center=(yedges[j]+yedges[j+1])/2,
                             original_member_count=int(h[i,j]),ensemble_share_pct=100*h[i,j]/len(g)))
    vmax=max(h.max() for h in counts.values())/5
    norm=LogNorm(vmin=.2,vmax=vmax)
    share_cmap=LinearSegmentedColormap.from_list('ensemble_share',['#E1E7E8','#658A94','#153B4C'])
    share_cmap.set_bad((1,1,1,0))
    for bi,bg in enumerate(SCENARIOS):
        x=110+bi*34
        g=members.loc[members.scenario.eq(bg)]
        margin=ax_mm(fig,x,139,23,6)
        margin.stairs(counts[bg].sum(axis=1)/5,transform(xedges),fill=True,color=COLORS[bg],alpha=.42,lw=.3)
        margin.set(xlim=(transform(xlo),transform(xhi)));margin.axis('off')
        axis=ax_mm(fig,x,147,23,41)
        shares=counts[bg]/5
        axis.pcolormesh(transform(xedges),yedges,np.ma.masked_where(shares.T==0,shares.T),
                        cmap=share_cmap,norm=norm,shading='flat',edgecolors='none',rasterized=False)
        right=ax_mm(fig,x+24,147,4,41)
        right.stairs(shares.sum(axis=0),yedges,orientation='horizontal',fill=True,
                     color=COLORS[bg],alpha=.42,lw=.3)
        right.set(ylim=(yedges[0]-1,yedges[-1]+1));right.axis('off')
        axis.axhline(0,color=MUTED,lw=.45,ls=(0,(2,2)));axis.axvline(0,color=MUTED,lw=.45,ls=(0,(2,2)))
        ticks=[v for v in (-100,0,50,150,400) if xlo<=v<=xhi]
        axis.set(xlim=(transform(xlo),transform(xhi)),ylim=(yedges[0]-1,yedges[-1]+1),
                 xticks=transform(ticks),xticklabels=[f'{v/1000:g}'.replace('-', '−') for v in ticks])
        axis.tick_params(labelsize=5.6,pad=2)
        axis.set_yticks([v for v in (0,10,20,30,40) if yedges[0]<=v<=yedges[-1]])
        if bi:axis.set_yticklabels([])
        else:axis.set_ylabel('Additional years above 1.5 °C',fontsize=6.2)
        margin.set_title(f'{bg} · median +{g.delta_window_max_K.median():.3f} °C',fontsize=5.6,color=COLORS[bg],pad=2,x=14/23)
        text_mm(fig,x+14,196,f'{100*g.delta_exceedance_years_77.gt(0).mean():.1f}% add years',ha='center',fontsize=5.7,color=COLORS[bg])
    fig.text(141/WIDTH_MM,1-204/220,'Change in peak warming (°C)',
             fontsize=5.9,ha='center',va='center')
    legend_axis=ax_mm(fig,13,199.5,77,9)
    legend_axis.axis('off')
    legend_axis.legend(handles=[
        Rectangle((0,0),1,1,fc=STATE_COLORS['never_above'],ec='none',label='Never above'),
        Rectangle((0,0),1,1,fc=STATE_COLORS['exceeded_still_above'],ec='none',label='Still above'),
        Rectangle((0,0),1,1,fc=STATE_COLORS['exceeded_returned'],ec='none',label='Returned'),
        Rectangle((0,0),1,1,fc='#B76C55',ec='none',label='Lost return')],
        loc='center',ncol=2,fontsize=5.5,handlelength=1.1,handletextpad=.5,
        columnspacing=2.1,labelspacing=.75,borderaxespad=0)
    cax=ax_mm(fig,177,156,1.1,25)
    color_ticks=[v for v in (.2,1,5,10,20) if v<=vmax]
    cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=share_cmap),cax=cax,ticks=color_ticks)
    cbar.ax.tick_params(labelsize=5.1,length=1,pad=1);cbar.outline.set_linewidth(.35)
    cbar.ax.set_yticklabels([f'{v:g}' for v in color_ticks])
    cbar.solids.set_rasterized(False);cbar.ax.set_title('Share\n(%)',fontsize=5.1,pad=4)
    balance_vertical_outer_margins(fig)
    return fig,pd.DataFrame(rows)


def arrow_path(ax,x,y,index,color):
    ax.annotate('',xy=(x[index+2],y[index+2]),xytext=(x[index],y[index]),
                arrowprops=dict(arrowstyle='-|>',mutation_scale=7,color=color,lw=.9))


AGE_MERGED = [('<5', ['0_to_5']), ('5–20', ['5_to_10', '10_to_20']),
              ('20–40', ['20_to_30', '30_to_40']),
              ('40–100', ['40_to_60', '60_to_100']), ('≥100', ['100_plus']),
              ('Primary (no age)', ['primary_unaged'])]
AGE_COLORS = ['#C77F4D', '#E2BE83', '#A9C9B1', '#609E9D', '#355F70', '#D6DBDF']
AGE_INTERVALS = ['[0,5)', '[5,20)', '[20,40)', '[40,100)', '[100,infinity)', 'unaged']
AGE_BOUNDS = [(0, 5), (5, 20), (20, 40), (40, 100), (100, None), (None, None)]
STAGE_LABELS = ['Before exceedance', 'Exceedance to peak', 'Peak to return', 'After return']


def export_age_groups(age_means, destination, age_source):
    """Export exactly the additive age-area values drawn in panel b."""
    destination.mkdir(parents=True, exist_ok=True)
    age_means.to_csv(destination/'common_four_stage_age_means.csv', index=False)
    wide_rows = []
    for bg in SCENARIOS:
        for si, stage_name in enumerate(STAGE_LABELS):
            group = age_means.loc[age_means.scenario.eq(bg) & age_means.stage_index.eq(si)]
            require(len(group) == len(AGE_MERGED), 'Each phase requires all six forest-area groups')
            row = dict(scenario=bg, stage_index=si, stage=stage_name,
                       n_common_members=int(group.n_common_members.iloc[0]),
                       total_forest_area_Mha=float(group.ensemble_mean_forest_area_Mha.iloc[0]))
            for (label, _), interval in zip(AGE_MERGED, AGE_INTERVALS):
                displayed = label if interval == 'unaged' else f'{label} yr {interval}'
                row[displayed+' area_Mha'] = float(group.loc[group.age_support.eq(label), 'ensemble_mean_area_Mha'].iloc[0])
            wide_rows.append(row)
    pd.DataFrame(wide_rows).to_csv(destination/'panel_b_age_area_Mha.csv', index=False)
    manifest = dict(
        schema='fig02-panel-b-age-groups-v2', source=str(age_source),
        units='Mha', stage_labels=STAGE_LABELS,
        age_boundary_rule='Age intervals include the lower bound and exclude the upper bound; primary forest has no assigned age.',
        grouping=[dict(label=label if interval == 'unaged' else label+' yr',
                       interval=interval, source_groups=source_groups, color=color)
                  for (label, source_groups), interval, color in zip(AGE_MERGED, AGE_INTERVALS, AGE_COLORS)],
        statistic='Original member phase-mean areas are summed over source age classes and then averaged over the unmodified common-four-stage members.',
        n_common_members={'VL': 158, 'LN': 127},
        preservation='No simulation, stage calendar, member ID, eligibility mask or carbon quantity is modified.')
    (destination/'panel_b_age_group_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def common_stage_sources(phases, age_source):
    """Join climate stages to the matching detailed-age member mask."""
    carbon_rows, age_rows, membership_rows, stats_rows = [], [], [], []
    with xr.open_dataset(age_source) as ds:
        source_ids = ds.source_config.values
        require(len(source_ids) == 500, 'The age source must retain the original 500 IDs')
        for bg in SCENARIOS:
            d = ds.sel(scenario=bg)
            keep = np.asarray(d.common_four_nonempty_stages.values, dtype=bool)
            positions = np.flatnonzero(keep)
            expected = {'VL': 158, 'LN': 127}[bg]
            require(len(positions) == expected, 'Unexpected common-four-stage source population')
            base = phases.loc[phases.scenario.eq(bg)].copy()
            for pos, sid in enumerate(source_ids):
                id_rows = base.loc[base.member_position.eq(pos)]
                require(len(id_rows) == 4 and np.all(id_rows.source_config.to_numpy() == sid),
                        'Climate-stage and detailed-age member IDs must match in original order')
                membership_rows.append(dict(scenario=bg, member_position=int(pos), source_config=int(sid),
                                            common_four_nonempty_stages=bool(keep[pos])))
            chosen = base.loc[base.member_position.isin(positions)].copy()
            require(len(chosen) == 4 * expected and chosen.stage_has_intervals.all(),
                    'Each common member must have all four nonempty climate stages')
            carbon_rows.append(chosen)
            for si in range(4):
                member_areas = d.age_area_Mha_mean.sel(stage_index=si).isel(config=positions)
                forest = float(member_areas.sel(detailed_age_group=[v for _, group in AGE_MERGED for v in group]).sum('detailed_age_group').mean('config'))
                for (label, group), color, interval, bounds in zip(AGE_MERGED, AGE_COLORS, AGE_INTERVALS, AGE_BOUNDS):
                    area = float(member_areas.sel(detailed_age_group=group).sum('detailed_age_group').mean('config'))
                    age_rows.append(dict(scenario=bg, stage_index=si, n_common_members=expected,
                                         stage=STAGE_LABELS[si], age_support=label,
                                         age_label=label if interval == 'unaged' else label+' yr',
                                         age_interval=interval,
                                         age_lower_inclusive_years=bounds[0],
                                         age_upper_exclusive_years=bounds[1],
                                         ensemble_mean_area_Mha=area,
                                         ensemble_mean_forest_area_Mha=forest,
                                         share_of_ensemble_mean_forest_area_pct=100 * area / forest))
                require(np.isclose(sum(row['ensemble_mean_area_Mha'] for row in age_rows[-len(AGE_MERGED):]), forest,
                                   rtol=1e-12, atol=1e-9), 'Merged age areas must preserve the total forest area')
                rows = chosen.loc[chosen.stage_index.eq(si)]
                for field in ('delta_stage_mean_net_PgC_yr', 'delta_stage_cumulative_net_PgC',
                              'on_stage_mean_net_PgC_yr', 'off_stage_mean_net_PgC_yr', 'duration_years'):
                    values = rows[field].to_numpy(dtype=float)
                    require(np.isfinite(values).all(), 'Common-stage source values must be finite')
                    stats_rows.append(dict(scenario=bg, stage_index=si, metric=field, n=expected,
                                           mean=float(values.mean()), median=float(np.median(values)),
                                           q05=float(np.quantile(values, .05)), q95=float(np.quantile(values, .95)),
                                           minimum=float(values.min()), maximum=float(values.max())))
    return (pd.concat(carbon_rows, ignore_index=True), pd.DataFrame(age_rows),
            pd.DataFrame(membership_rows), pd.DataFrame(stats_rows))


def climate_stage_navigation(fig):
    panel(fig, 4, 8, 'a', 'Overshoot phases')
    labels = ['Before\nexceedance', 'Exceedance\nto peak', 'Peak to\nreturn', 'After\nreturn']
    stage_centers = 19 + (np.arange(4) + .5) * 157 / 4
    for x, label in zip(stage_centers, labels):
        text_mm(fig, x, 16, label, ha='center', fontsize=6.2)
    ax = ax_mm(fig, 19, 26, 157, 9)
    colors = ['#EDF1F3', '#F4E9DD', '#E7EEEE', '#EAF0E8']
    for i, color in enumerate(colors):
        ax.axvspan(i, i+1, color=color, ec='none')
    x = np.linspace(0, 4, 401)
    y = np.interp(x, [0, 1, 1.4, 2, 2.5, 3, 4], [-.55, 0, .52, .8, .44, 0, -.45])
    ax.plot(x, y, color=INK, lw=1)
    ax.axhline(0, color=MUTED, lw=.5, ls=(0, (2, 2)))
    for i in (1, 2, 3):
        ax.axvline(i, color='white', lw=1)
    ax.text(-.035, 0, '1.5 °C', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.5)
    ax.set(xlim=(0, 4), ylim=(-.7, 1)); ax.axis('off')


def figure02(members, phases, annual, age_rows, common_carbon, summary):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,247/25.4))
    climate_stage_navigation(fig)

    panel(fig,4,45,'b','Forest age composition')
    max_area=age_rows.ensemble_mean_forest_area_Mha.max()/1000
    for si in range(4):
        axis=ax_mm(fig,19+si*157/4,55,34.5,19)
        for bi,bg in enumerate(SCENARIOS):
            left=0.
            for (label,_),color in zip(AGE_MERGED,AGE_COLORS):
                row=age_rows.loc[age_rows.scenario.eq(bg)&age_rows.stage_index.eq(si)&age_rows.age_support.eq(label)].iloc[0]
                width=row.ensemble_mean_area_Mha/1000
                axis.barh(1-bi,width,left=left,height=.5,color=color,edgecolor='white',linewidth=.3)
                left+=width
        axis.set(xlim=(0,max_area*1.03),ylim=(-.6,1.6),xticks=[0,2,4],yticks=[1,0])
        axis.tick_params(labelsize=5.5,pad=1,length=2)
        axis.set_yticklabels(['VL','LN'] if si==0 else [])
        for spine in ('left','top','right'):axis.spines[spine].set_visible(False)
        if si==0:
            for label,bg in zip(axis.get_yticklabels(),SCENARIOS):label.set_color(COLORS[bg])
        axis.set_xlabel('Forest area (Gha)',fontsize=5.7,labelpad=2)
    for i,((label,_),color) in enumerate(zip(AGE_MERGED,AGE_COLORS)):
        x=19+i*26.6
        fig.add_artist(Rectangle((x/WIDTH_MM,1-89/247),2.2/WIDTH_MM,1.7/247,
                                 transform=fig.transFigure,fc=color,ec='none'))
        text_mm(fig,x+3.1,87.1,label if label=='Primary (no age)' else label+' yr',fontsize=5.7,color=MUTED)

    panel(fig,4,100,'c','Uptake difference by phase')
    for metric, top, height, ylabel in [
            ('delta_stage_mean_net_PgC_yr',110,18,'Rate difference\n(PgC yr$^{-1}$)'),
            ('delta_stage_cumulative_net_PgC',137,20,'Phase total\n(PgC)')]:
        c=ax_mm(fig,19,top,157,height)
        for bi,bg in enumerate(SCENARIOS):
            values=[common_carbon.loc[common_carbon.scenario.eq(bg)&common_carbon.stage_index.eq(si),
                                      metric].to_numpy() for si in range(4)]
            positions=np.arange(4)+.5+(-.12 if bi==0 else .12)
            violins=c.violinplot(values,positions=positions,widths=.21,showextrema=False,bw_method=.28,points=180)
            for body in violins['bodies']:
                body.set_facecolor(COLORS[bg]);body.set_edgecolor(COLORS[bg]);body.set_alpha(.34);body.set_linewidth(.45)
            for x,v in zip(positions,values):
                q05,median,q95=np.quantile(v,[.05,.5,.95])
                c.plot([x,x],[q05,q95],color=COLORS[bg],lw=.7)
                c.plot(x,median,'o',ms=2.3,color=COLORS[bg],mec='white',mew=.35)
        c.axhline(0,color=MUTED,lw=.55,ls=(0,(2,2)))
        c.set(xlim=(0,4),xticks=[],ylabel=ylabel)
        c.tick_params(labelsize=5.8,pad=2)
        if metric.endswith('_yr'):
            pathway_legend(c,loc='upper center',ncol=2,fontsize=5.6,bbox_to_anchor=(.5,1.25))

    panel(fig,4,171,'d','Uptake rate and CO$_2$ differences')
    a=ax_mm(fig,18,183,69,50)
    lo=np.floor(annual.delta_net_PgC_yr_mean.min()*10)/10-.05
    hi=max(.2,np.ceil(annual.delta_net_PgC_yr_mean.max()*10)/10+.05)
    a.axvspan(0,hi,color='#E6EFEB',alpha=.8,lw=0)
    a.axvline(0,color=MUTED,lw=.6)
    for bg in SCENARIOS:
        g=annual.loc[annual.scenario.eq(bg)&annual.year.ge(2023)].sort_values('year')
        x=g.delta_net_PgC_yr_mean.to_numpy();y=g.delta_CO2_ppm_mean.to_numpy()
        a.plot(x,y,color=COLORS[bg],lw=1.1)
        for index in (9,35,61):arrow_path(a,x,y,index,COLORS[bg])
        for year in (2030,2050,2070,2099):
            row=g.loc[g.year.eq(year)].iloc[0]
            a.plot(row.delta_net_PgC_yr_mean,row.delta_CO2_ppm_mean,'o',ms=2.4,
                   color=COLORS[bg],mfc=COLORS[bg] if year==2099 else 'white',mew=.5)
        dy={'VL':-11,'LN':-23}[bg]
        a.annotate('2099 '+bg,xy=(x[-1],y[-1]),xytext=(9,dy),textcoords='offset points',fontsize=5.7,
                   color=COLORS[bg],arrowprops=dict(arrowstyle='-',color=COLORS[bg],lw=.4))
        if bg=='VL':
            for year,offset in [(2030,(-31,8)),(2050,(-24,4)),(2070,(-23,4))]:
                row=g.loc[g.year.eq(year)].iloc[0]
                a.annotate(str(year),xy=(row.delta_net_PgC_yr_mean,row.delta_CO2_ppm_mean),
                           xytext=offset,textcoords='offset points',fontsize=5.5,
                           arrowprops=dict(arrowstyle='-',color=MUTED,lw=.35))
    a.set(xlim=(lo,hi+.18),ylim=(-.2,np.ceil(annual.delta_CO2_ppm_mean.max())+.3),
          xlabel='Uptake-rate difference (PgC yr$^{-1}$)',ylabel='CO$_2$ difference (ppm)')
    a.tick_params(labelsize=5.7,pad=1.6);a.xaxis.label.set_size(5.9);a.yaxis.label.set_size(6)

    panel(fig,99,171,'e','Cumulative uptake difference')
    b=ax_mm(fig,112,183,64,50)
    for bg in SCENARIOS:
        g=annual.loc[annual.scenario.eq(bg)].sort_values('year')
        b.fill_between(g.year,g.delta_cumulative_net_PgC_q05,g.delta_cumulative_net_PgC_q95,
                       color=COLORS[bg],alpha=.05,lw=0)
        b.plot(g.year,g.delta_cumulative_net_PgC_median,color=COLORS[bg],lw=1.1)
        b.plot(2099,g.delta_cumulative_net_PgC_median.iloc[-1],'o',color=COLORS[bg],ms=2.5)
        b.annotate(f'{bg} {g.delta_cumulative_net_PgC_median.iloc[-1]:.1f}'.replace('-', '−'),
                   xy=(2099,g.delta_cumulative_net_PgC_median.iloc[-1]),
                   xytext=(-3,8 if bg=='LN' else -10),textcoords='offset points',fontsize=5.5,ha='right',color=COLORS[bg])
    b.axhline(0,color=MUTED,lw=.6,ls=(0,(2,2)))
    b.set(xlim=(2022,2101),ylim=(np.floor(annual.delta_cumulative_net_PgC_q05.min()/10)*10,5),
          xticks=[2022,2060,2099],xlabel='Calendar year',ylabel='Cumulative uptake difference (PgC)')
    b.tick_params(labelsize=5.7,pad=1.6);b.xaxis.label.set_size(5.9);b.yaxis.label.set_size(6)
    return fig


def read_caption(path):
    """Read Markdown, selecting its English-caption section when present."""
    text = Path(path).read_text()
    marker = '## English caption'
    return (text.split(marker, 1)[1] if marker in text else text).strip() + '\n'


def export(fig, output, name, caption, analysis, source_names, preserve_sources=False):
    output.mkdir(parents=True, exist_ok=True)
    data_output = output / "source_data"
    preview_output = output / "preview"
    reproduction_output = output / "reproduction"
    for folder in (data_output, preview_output, reproduction_output):
        folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (name + '.svg'), facecolor='white')
    fig.savefig(output / (name + '.pdf'), facecolor='white')
    fig.savefig(output / (name + '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(output / (name + '.png'), dpi=300)
    # The separate screen-size QA image is not a publication raster export.
    with Image.open(output / (name + '.png')) as preview:
        screen_size = (round(WIDTH_MM / 25.4 * SCREEN_PPI), round(fig.get_figheight() * SCREEN_PPI))
        preview.resize(screen_size, Image.Resampling.LANCZOS).save(
            preview_output / (name + '_final_size_183mm.png'), dpi=(SCREEN_PPI, SCREEN_PPI))
    (output / 'caption.md').write_text(caption)
    shutil.copy2(__file__, reproduction_output / 'plot_main_figures_phenomenon_carbon.py')
    provenance = reproduction_output / 'provenance'
    provenance.mkdir(exist_ok=True)
    shutil.copy2(__file__, provenance / 'plot_source_used.py')
    if not preserve_sources:
        for filename in source_names:
            shutil.copy2(analysis / filename, data_output / filename)
        shutil.copy2(analysis / 'summary.json', data_output / 'analysis_summary.json')
        shutil.copy2(analysis / 'analysis_source_used.py', reproduction_output / 'analysis_source_used.py')
    tree = ET.parse(output / (name + '.svg'))
    ns = '{http://www.w3.org/2000/svg}'
    with Image.open(output / (name + '.tiff')) as im:
        tiff = dict(pixel_dimensions=list(im.size), dpi=[float(v) for v in im.info.get('dpi', ())], mode=im.mode)
    qa = dict(created_utc=datetime.now(timezone.utc).isoformat(), width_mm=WIDTH_MM,
              height_mm=round(fig.get_figheight() * 25.4, 4),
              svg_text_elements=sum(1 for _ in tree.iter(ns + 'text')),
              svg_raster_images=sum(1 for _ in tree.iter(ns + 'image')), tiff=tiff,
              svg_text_editable=True, pdf_fonttype=42, preview_dpi=300, final_size_preview_dpi=SCREEN_PPI,
              population='Original 500 paired IDs per condition; conditions never pooled',
              displayed_pathways=list(SCENARIOS), analysis_pathways=list(ALL_SCENARIOS),
              displayed_populations=({'trajectory_endpoint_joint_state': '500 per pathway'} if 'climate' in name
                    else {'age_and_phase_rate_block': {'VL':158,'LN':127},
                          'full_calendar_rate_CO2_and_cumulative': '500 per pathway'}),
              display_selection='Upcrossing-peak-decline morphology, not effect magnitude',
              source_analysis=str(analysis), render_backend='Python matplotlib')
    if hasattr(fig, '_vertical_outer_margins_mm'):
        qa['vertical_outer_margins_mm'] = fig._vertical_outer_margins_mm
    (reproduction_output / 'export_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    (output / 'README.md').write_text(
        f'# Figure export\n\n[High-resolution PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg) · [600 dpi TIFF]({name}.tiff) · [Caption](caption.md)\n\n[96 dpi screen preview](preview/{name}_final_size_183mm.png) · [Source Data](source_data/) · [Reproduction and QA](reproduction/)\n\n'
        'The analysis uses five fully matched futures, unretimed land activity, and the common '
        'member-wise H1 end-2022 state. Temperature refers to the common member-wise 1850–1900 mean. '
        'The main display highlights VL and LN for their upcrossing-peak-decline morphology; '
        'all five pathways remain in Source Data and the companion Extended Data figures. '
        'The original 500 IDs per pathway and all event/censoring flags remain in the source tables. '
        'These are parameter-ensemble summaries, not confidence intervals or real-world probabilities.\n\n'
        'Run the canonical plot workflow with --analysis-dir pointing to the matching analysis/core '
        'and --output pointing to a new explicit figure directory. The runnable plotting source is in reproduction/; '
        'reproduction/analysis_source_used.py rebuilds derived tables from the canonical thin source. '
        'The compressed annual member source is shared by both figures.\n\n'
        'SVG/PDF preserve editable text; TIFF is 600 dpi with lossless LZW compression. '
        'The final-size PNG is rendered at 183 mm width and 96 pixels per inch. '
        'reproduction/export_QA.json records physical dimensions and export properties. Data are in source_data/.\n')
    if 'phase_carbon' in name:
        with (output/'README.md').open('a') as note:
            note.write('\nThe age-stage source is the canonical analysis/age_mechanism/actual_age_stages_all500.nc, '
                       'also included here. Supply it with --age-stages when reproducing the plot. '
                       'Panels b–c use the same unmodified complete-four-stage mask (VL 158, LN 127); '
                       'all-calendar panels d–e retain all 500 original members. The common_four_stage_* '
                       'tables are display-specific derivations and do not replace the complete climate-stage tables. '
                       'common_stage_source_manifest.json records the exact join and source paths.\n')
    plt.close(fig)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--reuse-source-data', action='store_true',
                        help='Rerender existing tables without rewriting any Source Data')
    parser.add_argument('--refresh-age-groups', action='store_true',
                        help='Reaggregate panel b from the existing detailed age-stage source and export only its age-area tables when reusing other Source Data')
    parser.add_argument('--caption-file', type=Path,
                        help='External Markdown caption for one selected figure; otherwise use each output caption.md')
    parser.add_argument('--age-stages', type=Path,
                        default=Path('outputs/age_module/main_figure_redesign/analysis/age_mechanism/actual_age_stages_all500.nc'),
                        help='Canonical detailed-age stage source, independent of figure packages')
    parser.add_argument('--figures', choices=['both', 'fig01', 'fig02'], default='both')
    args = parser.parse_args()
    source, output = args.analysis_dir, args.output
    summary = json.loads((source / 'summary.json').read_text())
    require(summary['status'] == 'complete', 'A complete matched-future core is required')
    members = pd.read_csv(source / 'member_outcomes_all500.csv')
    phases = pd.read_csv(source / 'member_stage_carbon_all500.csv')
    annual = pd.read_csv(source / 'annual_summary.csv')
    stage = pd.read_csv(source / 'stage_summary.csv')
    transitions = pd.read_csv(source / 'state_transitions.csv')
    require(len(members) == 2500 and all(len(members.loc[members.scenario == b]) == 500 for b in ALL_SCENARIOS),
            'All five original-500 conditions required')
    members=members.loc[members.scenario.isin(SCENARIOS)].copy()
    annual=annual.loc[annual.scenario.isin(SCENARIOS)].copy()
    phases=phases.loc[phases.scenario.isin(SCENARIOS)].copy()
    stage=stage.loc[stage.scenario.isin(SCENARIOS)].copy()
    transitions=transitions.loc[transitions.scenario.isin(SCENARIOS)].copy()
    output.mkdir(parents=True, exist_ok=True)
    if args.caption_file and args.figures == 'both':
        parser.error('--caption-file requires --figures fig01 or fig02')
    if args.figures in ('both', 'fig01'):
        caption = read_caption(args.caption_file or output/'fig01'/'caption.md')
        fig, bins = figure01(members, annual, transitions, summary)
        if args.preview_only:
            dest=output/'fig01'/'preview'; dest.mkdir(parents=True,exist_ok=True)
            fig.savefig(dest/'fig01_climate_outcomes_final_size_183mm.png',dpi=SCREEN_PPI)
            plt.close(fig)
            print('fig01 preview ready')
        else:
            qa = export(fig, output / 'fig01', 'fig01_climate_outcomes', caption, source,
                    ['member_outcomes_all500.csv', 'annual_summary.csv', 'state_transitions.csv'], args.reuse_source_data)
            if not args.reuse_source_data:
                bins.to_csv(output / 'fig01' / 'source_data' / 'density_bin_counts.csv', index=False)
            targets=members.copy()
            targets['off2099_meets_on2099_misses']=(targets.off_T2099_K<=1.5)&(targets.on_T2099_K>1.5)
            if not args.reuse_source_data:
                targets.to_csv(output/'fig01'/'source_data'/'target_endpoint_pairs_all500.csv',index=False)
            print('fig01', json.dumps(qa))
    if args.figures in ('both', 'fig02'):
        caption = read_caption(args.caption_file or output/'fig02'/'caption.md')
        dest=output/'fig02'/'source_data'
        if args.reuse_source_data:
            common_carbon = pd.read_csv(dest/'common_four_stage_carbon_members.csv')
            if args.refresh_age_groups:
                _, age_means, _, _ = common_stage_sources(phases, args.age_stages)
                export_age_groups(age_means, dest, args.age_stages)
            else:
                age_means = pd.read_csv(dest/'common_four_stage_age_means.csv')
                require(set(age_means.age_support) == {label for label, _ in AGE_MERGED},
                        'Existing panel b uses older age groups; rerun with --refresh-age-groups')
        else:
            common_carbon, age_means, common_members, common_stats = common_stage_sources(phases, args.age_stages)
            dest.mkdir(parents=True,exist_ok=True)
            common_carbon.to_csv(dest/'common_four_stage_carbon_members.csv',index=False)
            export_age_groups(age_means, dest, args.age_stages)
            common_members.to_csv(dest/'common_four_stage_membership_all500.csv',index=False)
            common_stats.to_csv(dest/'common_four_stage_net_summary.csv',index=False)
            (dest/'common_stage_source_manifest.json').write_text(json.dumps(dict(
                schema='main-figure-common-four-stages-v1', age_source=str(args.age_stages),
                carbon_source=str(source/'member_stage_carbon_all500.csv'),
                eligibility='Detailed-age common_four_nonempty_stages; matching climate-stage calendars and member IDs',
                n_common_members={'VL':158,'LN':127},
                displayed_age_statistic='Ensemble mean of member phase-mean age areas; additive within each phase',
                displayed_carbon_statistic='Member paired phase-mean on-minus-off net uptake',
                full_calendar_population='All original 500 per pathway in the complete climate-stage tables'),indent=2)+'\n')
        fig = figure02(members, phases, annual, age_means, common_carbon, summary)
        if args.preview_only:
            dest=output/'fig02'/'preview'; dest.mkdir(parents=True,exist_ok=True)
            fig.savefig(dest/'fig02_phase_carbon_final_size_183mm.png',dpi=SCREEN_PPI)
            plt.close(fig)
            print('fig02 preview ready')
        else:
            qa = export(fig, output / 'fig02', 'fig02_phase_carbon', caption, source,
                    ['member_outcomes_all500.csv', 'member_stage_carbon_all500.csv',
                     'annual_summary.csv', 'stage_summary.csv', 'annual_members_all500.csv.gz'], args.reuse_source_data)
            if not args.reuse_source_data:
                shutil.copy2(args.age_stages, output/'fig02'/'source_data'/'actual_age_stages_all500.nc')
            print('fig02', json.dumps(qa))


if __name__ == '__main__':
    main()
