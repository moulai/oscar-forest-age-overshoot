#!/usr/bin/env python3
"""Draw Extended Data Figure 4 from the complete matched five-pathway core."""
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
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from PIL import Image

SCENARIOS = ('H', 'HL', 'M', 'VL', 'LN')
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


def pathway_legend(ax, **kwargs):
    handles = [mpl.lines.Line2D([], [], color=COLORS[b], lw=1.3, label=b) for b in SCENARIOS]
    return ax.legend(handles=handles, fontsize=6.1, handlelength=1.2, handletextpad=.3,
                     columnspacing=.8, **kwargs)


def flow_patch(ax, x0, x1, y0, y1, z0, z1, color, alpha):
    vertices = [(x0, y0), (x0 + .28, y0), (x1 - .28, z0), (x1, z0),
                (x1, z1), (x1 - .28, z1), (x0 + .28, y1), (x0, y1), (x0, y0)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor='none', alpha=alpha))



def balance_vertical_outer_margins(fig):
    """Translate the complete content without resizing any axes or altering data."""
    def visible_bounds():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [t.get_window_extent(renderer) for t in fig.texts
                  if t.get_visible() and t.get_text()]
        bounds.extend(legend.get_window_extent(renderer) for legend in fig.legends
                      if legend.get_visible())
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
    # Apply the fixed vertical offset to the complete panel group.
    down_mm = -6.102013888888894
    delta = down_mm / (fig.get_figheight() * 25.4)
    for axis in fig.axes:
        pos = axis.get_position()
        axis.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, y - delta))
    for legend in fig.legends:
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((anchor.x0, anchor.y0 - delta, anchor.width, anchor.height),
                                  transform=fig.transFigure)
    fig._vertical_outer_margins_mm = dict(before=before, content_shift_down=down_mm,
                                          after=visible_bounds())


def figure01(members, annual, transitions, summary):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 220 / 25.4))
    panel(fig, 4, 19, 'a', 'Median warming trajectories')
    a = ax_mm(fig, 14, 27, 161, 32)
    for bg in SCENARIOS:
        g = annual.loc[annual.scenario.eq(bg)].sort_values('year')
        a.fill_between(g.year, g.on_temperature_K_q05, g.on_temperature_K_q95,
                       color=COLORS[bg], alpha=.045, lw=0)
        a.plot(g.year, g.on_temperature_K_median, color=COLORS[bg], lw=1.15)
        a.plot(g.year, g.off_temperature_K_median, color=COLORS[bg], lw=.85, ls=(0, (3, 2)))
    a.axhline(1.5, color=MUTED, lw=.65, ls=(0, (1.5, 2)))
    a.set(xlim=(2022, 2099), ylim=(.65, float(np.ceil(annual.on_temperature_K_q95.max() * 2) / 2)),
          xticks=[2022, 2040, 2060, 2080, 2099], ylabel='Warming from\n1850–1900 (°C)', xlabel='Calendar year')
    a.text(2023, 1.52, '1.5 °C', ha='left', va='bottom', fontsize=6, color=MUTED)
    pathway_legend(a, loc='upper left', ncol=5)
    a.add_artist(mpl.legend.Legend(a,
        [mpl.lines.Line2D([], [], color=INK, lw=1.1),
         mpl.lines.Line2D([], [], color=INK, lw=1, ls=(0,(3,2)))],
        ['Age on','Age off'],loc='lower right',ncol=2,fontsize=5.9,frameon=False,
        handlelength=1.4,columnspacing=1))

    panel(fig, 4, 75, 'b', 'Peak warming and years above 1.5 °C')
    xvalues = members.delta_window_max_K.to_numpy() * 1000
    yvalues = members.delta_exceedance_years_77.to_numpy()
    xlo = int(np.floor(xvalues.min() / 10) * 10 - 10)
    xhi = int(np.ceil(xvalues.max() / 10) * 10 + 10)
    ylo, yhi = int(yvalues.min()) - .5, int(yvalues.max()) + .5
    xedges = np.arange(xlo, xhi + 1, 10)
    yedges = np.arange(ylo, yhi + 1, 1)
    counts = {}
    bin_rows = []
    for bg in SCENARIOS:
        g = members.loc[members.scenario.eq(bg)]
        h, _, _ = np.histogram2d(g.delta_window_max_K * 1000, g.delta_exceedance_years_77,
                                 bins=(xedges, yedges))
        require(int(h.sum()) == 500, 'Binned distribution lost an original member')
        counts[bg] = h
        for i, j in zip(*np.nonzero(h)):
            bin_rows.append(dict(scenario=bg, delta_max_mK_left=xedges[i],
                                 delta_max_mK_right=xedges[i + 1],
                                 delta_threshold_years_center=(yedges[j] + yedges[j + 1]) / 2,
                                 original_member_count=int(h[i, j]),
                                 ensemble_share_pct=100*h[i,j]/len(g)))
    vmax = max(v.max() for v in counts.values()) / 5
    norm = LogNorm(vmin=.2, vmax=vmax)
    share_cmap = LinearSegmentedColormap.from_list('ensemble_share', ['#E1E7E8','#658A94','#153B4C'])
    share_cmap.set_bad((1,1,1,0))
    for i, bg in enumerate(SCENARIOS):
        g = members.loc[members.scenario.eq(bg)]
        b = ax_mm(fig, 14 + i * 32.8, 91, 27.2, 44)
        shares = counts[bg] / 5
        b.pcolormesh(xedges, yedges, np.ma.masked_where(shares.T == 0, shares.T),
                     cmap=share_cmap, norm=norm, shading='flat', edgecolors='none', rasterized=False)
        b.axhline(0, color=MUTED, lw=.45, ls=(0, (2, 2)), zorder=0)
        b.axvline(0, color=MUTED, lw=.45, ls=(0, (2, 2)), zorder=0)
        b.set(xlim=(xlo, xhi), ylim=(ylo - 1, yhi + 1), xticks=[0, 150, 300], yticks=[0, 10, 20, 30, 40])
        b.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v/1000:g}'.replace('-', '−')))  # mK axis, °C labels
        b.set_title(f'{bg}\n+{g.delta_window_max_K.median():.3f} °C median',
                    color=COLORS[bg], fontsize=6.7, pad=4)
        if i:
            b.set_yticklabels([])
        else:
            b.set_ylabel('Additional years above 1.5 °C')
        b.text(.5, -.22, f'{100*(g.delta_exceedance_years_77>0).mean():.1f}% add years',
               transform=b.transAxes, fontsize=6.1, ha='center', color=COLORS[bg])
    text_mm(fig, 72, 147, 'Change in peak warming (°C)', fontsize=6.6)
    cbax = ax_mm(fig, 176.7, 96, 1.3, 32)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=share_cmap)
    color_ticks = [v for v in (.2,1,5,10,20) if v <= vmax]
    cbar = fig.colorbar(sm, cax=cbax, ticks=color_ticks)
    cbar.ax.set_yticklabels([f'{v:g}' for v in color_ticks])
    cbar.ax.tick_params(labelsize=5.6, length=1.5, pad=1)
    cbar.outline.set_linewidth(.35)
    cbar.solids.set_rasterized(False)
    cbar.ax.set_title('Share\n(%)', fontsize=5.6, pad=4)

    panel(fig, 4, 163, 'c', 'Return to 1.5 °C by 2099')
    for i, bg in enumerate(SCENARIOS):
        c = ax_mm(fig, 14 + i * 32.8, 173, 27.2, 25)
        c.set(xlim=(-.16, 1.16), ylim=(0, 1))
        c.axis('off')
        tr = transitions.loc[transitions.scenario.eq(bg)]
        node_counts = {}
        bases = {}
        for mode in ('off', 'on'):
            ss = summary['scenarios'][bg][mode + '_states']
            node_counts[mode] = ss
            bases[mode] = dict(zip(STATES, np.cumsum([0] + [ss[s] / 500 for s in STATES[:-1]])))
        offsets = {mode: dict(bases[mode]) for mode in ('off', 'on')}
        for source_state in STATES:
            for target_state in STATES:
                count = int(tr.loc[(tr.off_state == source_state) & (tr.on_state == target_state), 'count'].iloc[0])
                if not count:
                    continue
                thick = count / 500
                y0, z0 = offsets['off'][source_state], offsets['on'][target_state]
                lost = source_state == 'exceeded_returned' and target_state == 'exceeded_still_above'
                color = '#AE6250' if lost else STATE_COLORS[target_state]
                flow_patch(c, .055, .945, y0, y0 + thick, z0, z0 + thick, color, .9 if lost else .36)
                offsets['off'][source_state] += thick
                offsets['on'][target_state] += thick
        for mode, x in (('off', -.015), ('on', .945)):
            for state in STATES:
                c.add_patch(Rectangle((x, bases[mode][state]), .07, node_counts[mode][state] / 500,
                                      color=STATE_COLORS[state], ec='white', lw=.2))
        c.text(.5, 1.05, bg, transform=c.transAxes, ha='center', fontsize=7, fontweight='bold', color=COLORS[bg])
        # Center labels under the node bars and retain clear inter-pathway gaps.
        c.text(.14, -.08, 'Age off', transform=c.transAxes, ha='center', fontsize=5.4)
        c.text(.86, -.08, 'Age on', transform=c.transAxes, ha='center', fontsize=5.4)
        off_r = 100 * node_counts['off']['exceeded_returned'] / 500
        on_r = 100 * node_counts['on']['exceeded_returned'] / 500
        c.text(.5, -.25, f'{off_r:.1f} → {on_r:.1f}% returned', transform=c.transAxes,
               ha='center', fontsize=6.1)
    handles = [Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in STATES]
    handles.append(Rectangle((0, 0), 1, 1, color='#AE6250'))
    fig.legend(handles, ['Never above 1.5 °C', 'Returned', 'Still above in 2099', 'Lost return'],
               loc='lower center', bbox_to_anchor=(.52, .026), ncol=4,
               fontsize=6.1, handlelength=1, columnspacing=1.1, handletextpad=.4)
    balance_vertical_outer_margins(fig)
    return fig, pd.DataFrame(bin_rows)


def read_caption(path):
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
    shutil.copy2(__file__, reproduction_output / 'plot_extended_all_pathway_climate.py')
    provenance = reproduction_output / 'provenance'
    provenance.mkdir(exist_ok=True)
    shutil.copy2(__file__, provenance / 'plot_source_used.py')
    if not preserve_sources:
        for filename in source_names:
            shutil.copy2(analysis / filename, data_output / filename)
        summary_file=analysis / 'summary.json'
        if not summary_file.exists():summary_file=analysis/'analysis_summary.json'
        shutil.copy2(summary_file, data_output/'analysis_summary.json')
        if (analysis/'analysis_source_used.py').is_file():
            shutil.copy2(analysis/'analysis_source_used.py', reproduction_output/'analysis_source_used.py')
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
              source_analysis=str(analysis), render_backend='Python matplotlib',
              density_scale='100 times each bin count / 500 within its pathway; common log share scale',
              bin_dimensions='10 mK by one threshold year; zero bins transparent',
              data_coverage='All original members, signed changes, zeros and tails retained')
    qa['vertical_outer_margins_mm'] = fig._vertical_outer_margins_mm
    (reproduction_output / 'export_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    (output / 'README.md').write_text(
        f'# Figure export\n\n[High-resolution PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg) · [600 dpi TIFF]({name}.tiff) · [Caption](caption.md)\n\n[96 dpi screen preview](preview/{name}_final_size_183mm.png) · [Source Data](source_data/) · [Reproduction and QA](reproduction/)\n\n'
        'The analysis uses five fully matched futures, unretimed land activity, and the common '
        'member-wise H1 end-2022 state. Temperature refers to the common member-wise 1850–1900 mean. '
        'The original 500 IDs per pathway and all event/censoring flags remain in the source tables. '
        'These are parameter-ensemble summaries, not confidence intervals or real-world probabilities.\n\n'
        'Run the canonical plot workflow with --analysis-dir pointing to the matching analysis/core '
        'and --output pointing to a new explicit figure directory. The runnable plotting source is in reproduction/; '
        'reproduction/analysis_source_used.py rebuilds derived tables from the canonical thin source. '
        'The compressed annual member source is shared by both figure candidates.\n\n'
        'SVG/PDF preserve editable text; TIFF is 600 dpi with lossless LZW compression. '
        'The final-size PNG is rendered at 183 mm width and 96 pixels per inch. '
        'reproduction/export_QA.json records physical dimensions and export properties. Data are in source_data/.\n')
    plt.close(fig)
    return qa



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--analysis-dir',type=Path)
    group.add_argument('--source-bundle',type=Path)
    parser.add_argument('--caption-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.analysis_dir or args.source_bundle
    summary_path=source/'summary.json'
    if not summary_path.exists():summary_path=source/'analysis_summary.json'
    summary=json.loads(summary_path.read_text())
    require(summary['status']=='complete','A complete matched-future core is required')
    members=pd.read_csv(source/'member_outcomes_all500.csv')
    annual=pd.read_csv(source/'annual_summary.csv')
    transitions=pd.read_csv(source/'state_transitions.csv')
    require(len(members)==2500 and all(len(members.loc[members.scenario==b])==500 for b in SCENARIOS),
            'All five original-500 conditions required')
    fig,bins=figure01(members,annual,transitions,summary)
    preserve=source.resolve()==(args.output/'source_data').resolve()
    qa=export(fig,args.output,'ed04_all_pathway_climate',read_caption(args.caption_file),source,
              ['member_outcomes_all500.csv','annual_summary.csv','state_transitions.csv'],preserve)
    if not preserve or not (args.output/'source_data'/'density_bin_shares.csv').exists():
        bins.to_csv(args.output/'source_data'/'density_bin_shares.csv',index=False)
    (args.output/'source_data'/'README.md').write_text(
        '# Source Data for ED4\n\nThe compact original core tables retain all five conditions and 500 paired IDs each. '
        'No whole annual member cube is copied. density_bin_shares.csv adds ensemble_share_pct = '
        '100 * original_member_count / 500 to the same 10 mK by one-year bins. Empty bins are transparent; '
        'the shared colour coordinate is logarithmic, with actual percentage ticks. '
        'H/M window-end maxima are not resolved physical peaks; all state/left-truncation/censoring flags remain. '
        'Empirical parameter distributions are not scenario-weighted probabilities or confidence intervals.\n')
    (args.output/'reproduction'/'reproduce_commands.txt').write_text(
        'python -B reproduction/plot_extended_all_pathway_climate.py --source-bundle source_data '
        '--caption-file caption.md --output regenerated\n')
    print(json.dumps(qa))


if __name__=='__main__':main()
