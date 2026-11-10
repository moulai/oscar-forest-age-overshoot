#!/usr/bin/env python3
"""Draw the native regional annual-cohort land-model architecture (ED1)."""
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
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image

WIDTH_MM = 183
HEIGHT_MM = 231
SCREEN_PPI = 96
INK, MUTED = '#253039', '#707C83'
FOREST, EARTH, CLIMATE = '#316F78', '#A07855', '#7D668A'
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.linewidth': .55,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': INK, 'mathtext.default': 'regular'})


def label(ax, x, y, text, size=6.6, color=INK, ha='center', va='center', **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, **kwargs)


def panel(ax, x, y, letter, title):
    label(ax, x, y, letter, 8, ha='left', va='top', fontweight='bold')
    label(ax, x+4.5, y+.1, title, 7, ha='left', va='top')


def box(ax, x, y, w, h, text, fill='#EDF1F2', edge='none', size=6.6, radius=1.2):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                        facecolor=fill,edgecolor=edge,linewidth=.55,zorder=3)
    ax.add_patch(patch)
    label(ax,x+w/2,y+h/2,text,size,zorder=4)
    return patch


def route(ax, points, color=INK, width=.8, dashed=False, arrow=True, zorder=2):
    path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    artist=FancyArrowPatch(path=path,arrowstyle='-|>' if arrow else '-',
                           mutation_scale=7,linewidth=width,color=color,
                           linestyle=(0,(3,2)) if dashed else '-',zorder=zorder)
    ax.add_patch(artist)
    return artist



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
    down_mm = -2.0715000000000057
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


def draw():
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH_MM),ylim=(HEIGHT_MM,0));ax.axis('off')

    panel(ax,5,7,'a','Inputs and integration')
    box(ax,8,18,49,17,'Climate and emissions\ndrivers')
    box(ax,65,18,49,17,'Land-use transitions and\nwood-harvest requests')
    box(ax,122,18,53,17,'Regional age-response curves\nand initial carbon state')
    for x in (32.5,89.5,148.5):route(ax,[(x,35),(x,42)],MUTED,.65,True,False)
    route(ax,[(32.5,42),(148.5,42)],MUTED,.65,True,False)
    label(ax,91.5,46,'OSCAR · 11 land regions · annual forest cohorts',6.6)
    route(ax,[(89.5,49),(89.5,65)],MUTED,.7,True,False)
    route(ax,[(89.5,65),(48,65),(48,69)],MUTED,.7,True)
    route(ax,[(89.5,65),(138,65),(138,70)],MUTED,.7,True)

    panel(ax,5,55,'b','Land state and events')
    ax.add_patch(FancyBboxPatch((8,69),79,41,boxstyle='round,pad=0,rounding_size=1.5',
                              fc='#F1F5F4',ec='none',zorder=0))
    box(ax,11,74,25,12,'Primary forest\nNo age',fill='#D9E2DF',size=6.4)
    box(ax,11,93,25,12,'Non-forest\nNo forest age',fill='#E7E1D7',size=6.2)
    for x,y,birth in [(43,72,'b$_1$'),(47,84,'b$_2$'),(51,96,'b$_{new}$')]:
        box(ax,x,y,32,9,f'{birth}   A · W · F',fill='white',edge='#96B7B8',size=6.4,radius=.6)
        ax.add_patch(Rectangle((x,y),1.5,9,fc=FOREST,ec='none',zorder=5))
    label(ax,49,114,'Age = time − birth time',6.5)

    label(ax,10,129,'Non-forest establishment\nor primary harvest',6.4,ha='left')
    route(ax,[(53,129),(64,129)],FOREST,.9)
    box(ax,65,123,22,12,'New cohort',fill='#E1EEEA',size=6.2)
    route(ax,[(80,123),(86,119),(86,101),(83,101)],FOREST,.7)
    label(ax,10,145,'Standing secondary\nharvest',6.4,ha='left')
    route(ax,[(53,145),(64,145)],EARTH,.9)
    box(ax,65,139,22,12,'Cohort kept\nCarbon removed',fill='#F2E9DF',size=6.0)
    label(ax,10,161,'Forest conversion',6.4,ha='left')
    route(ax,[(53,161),(64,161)],EARTH,.9)
    box(ax,65,155,22,12,'Destination\nland class',fill='#ECE9E4',size=6.1)
    label(ax,48,175,'Events use the selected source stocks',6.0,color=MUTED)

    panel(ax,98,55,'c','Production and carbon pathways')
    box(ax,103,70,70,17,'Environmental response of NPP\nCO$_2$ · land temperature · precipitation',fill='#EDF1F2',size=6.5)
    label(ax,138,97,'Fixed W/F allocation × cohort area',6.1,color=MUTED)
    route(ax,[(123,87),(117,102)],FOREST,.8)
    route(ax,[(153,87),(157,102)],FOREST,.8)
    label(ax,117,104,'× g$_{W}$(age)',6.4,color=FOREST)
    label(ax,157,104,'× g$_{F}$(age)',6.4,color=FOREST)
    route(ax,[(117,108),(117,113.5)],FOREST,.8)
    route(ax,[(157,108),(157,113.5)],FOREST,.8)
    for x,text in [(117,'Wood\nW'),(157,'Other live\nF')]:
        ax.add_patch(Circle((x,122),8.5,fc='#DCEAED',ec=FOREST,lw=.6,zorder=3))
        label(ax,x,122,text,6.6,zorder=4)
    label(ax,124.5,141,'Mortality',5.9,color=MUTED)
    label(ax,149.5,141,'Mortality',5.9,color=MUTED)
    route(ax,[(117,130.5),(117,149)],EARTH,.8)
    route(ax,[(157,130.5),(157,149)],EARTH,.8)
    box(ax,103,149,27,12,'Dead wood\nCWD',fill='#ECE6DE',size=6.3)
    box(ax,144,149,27,12,'Soil',fill='#E4DED4',size=6.5)
    route(ax,[(130,155),(144,155)],EARTH,.8)
    route(ax,[(108.5,122),(98.5,122),(98.5,174),(103,174)],EARTH,.8)
    route(ax,[(98.5,155),(103,155)],EARTH,.8)
    label(ax,96.2,149,'Harvest',5.7,color=EARTH,rotation=90)
    box(ax,103,168,27,12,'Products\nHWP',fill='#EFE6DE',size=6.3)
    route(ax,[(128,161),(132,163),(132,185)],EARTH,.65,False,False)
    route(ax,[(157,161),(157,185)],EARTH,.65,False,False)
    route(ax,[(117,180),(117,185)],EARTH,.65,False,False)
    route(ax,[(117,185),(157,185)],EARTH,.65,False,False)
    label(ax,150,174,'Respiration\nand fire',6.1,color=EARTH)
    route(ax,[(123.5,128),(131,136),(171,136)],CLIMATE,.65,True,False)
    route(ax,[(165,130),(171,137),(171,187),(119,187),(119,205)],CLIMATE,.65,True)
    label(ax,138,194,'Fire → atmospheric chemistry',5.7,color=CLIMATE)

    panel(ax,5,192,'d','Carbon–climate coupling')
    box(ax,8,205,47,14,'Net land uptake\nrelative to preindustrial',fill='#DCE9E7',size=6.5)
    box(ax,65,205,28,14,'Atmospheric\nCO$_2$',fill='#E8E5EF',size=6.6)
    box(ax,105,205,27,14,'Effective\nforcing',fill='#E8E5EF',size=6.5)
    box(ax,144,202,31,19,'Surface and\ndeep-ocean\ntemperature',fill='#E5E1EC',size=6.3)
    route(ax,[(55,212),(65,212)],FOREST,.9,True)
    route(ax,[(93,212),(105,212)],CLIMATE,.8,True)
    route(ax,[(132,212),(144,212)],CLIMATE,.8,True)
    route(ax,[(147,185),(147,190),(4,190),(4,212),(8,212)],FOREST,.7,True)
    route(ax,[(27,179),(27,183),(33,183),(33,190)],FOREST,.7,True,False)
    label(ax,36,185,'Land fluxes relative to preindustrial',5.8,color=FOREST,ha='left')
    route(ax,[(79,224),(79,219)],MUTED,.6,True)
    label(ax,79,227,'Other CO$_2$ sources and sinks',5.8,color=MUTED)
    route(ax,[(118,224),(118,219)],MUTED,.6,True)
    label(ax,123,227,'Other forcing',5.8,color=MUTED)
    route(ax,[(175,211.5),(179,211.5),(179,78.5),(173,78.5)],CLIMATE,.85,True)
    label(ax,180.7,148,'CO$_2$ and climate feedback',5.9,color=CLIMATE,rotation=90)
    balance_vertical_outer_margins(fig)
    return fig


def read_caption(path):
    text=path.read_text()
    return (text.split('## English caption',1)[1] if '## English caption' in text else text).strip()+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle',type=Path,required=True)
    parser.add_argument('--caption-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sources=args.source_bundle
    for name in ('architecture_connections.csv','implementation_excerpts.md','source_manifest.json'):
        if not (sources/name).is_file():raise FileNotFoundError(sources/name)
    out=args.output
    for folder in ('source_data','reproduction','preview'):(out/folder).mkdir(parents=True,exist_ok=True)
    if sources.resolve()!=(out/'source_data').resolve():
        for p in sources.iterdir():
            if p.is_file():shutil.copy2(p,out/'source_data'/p.name)
    fig=draw();stem='ed01_native_age_architecture'
    fig.savefig(out/(stem+'.svg'))
    fig.savefig(out/(stem+'.pdf'))
    fig.savefig(out/(stem+'.png'),dpi=300)
    fig.savefig(out/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    with Image.open(out/(stem+'.png')) as im:
        im.resize((round(WIDTH_MM/25.4*SCREEN_PPI),round(HEIGHT_MM/25.4*SCREEN_PPI)),Image.Resampling.LANCZOS).save(
            out/'preview'/(stem+'_final_size_183mm.png'),dpi=(SCREEN_PPI,SCREEN_PPI))
    (out/'caption.md').write_text(read_caption(args.caption_file))
    shutil.copy2(__file__,out/'reproduction'/'plot_extended_model_architecture.py')
    tree=ET.parse(out/(stem+'.svg'));ns='{http://www.w3.org/2000/svg}'
    with Image.open(out/(stem+'.tiff')) as im:
        tiff=dict(pixel_dimensions=list(im.size),dpi=[float(x) for x in im.info['dpi']])
    qa=dict(created_utc=datetime.now(timezone.utc).isoformat(),width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
            svg_text_elements=sum(1 for _ in tree.iter(ns+'text')),
            svg_raster_images=sum(1 for _ in tree.iter(ns+'image')),
            svg_text_editable=True,pdf_fonttype=42,png_dpi=300,tiff=tiff,preview_dpi=SCREEN_PPI,
            render_backend='Python matplotlib',figure_type='Code-grounded architecture schematic',
            quantitative_data='No invented trajectory, spatial grid or numerical flow magnitude',
            arrow_meaning='Solid: physical/event routing; dashed: process information or model coupling')
    qa['vertical_outer_margins_mm'] = fig._vertical_outer_margins_mm
    (out/'reproduction'/'export_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    (out/'reproduction'/'reproduce_commands.txt').write_text(
        'python -B reproduction/plot_extended_model_architecture.py --source-bundle source_data '
        '--caption-file caption.md --output regenerated\n')
    (out/'README.md').write_text(
        '# Extended Data 1: native regional annual-cohort architecture\n\n'
        f'[High-resolution PNG]({stem}.png) · [PDF]({stem}.pdf) · [SVG]({stem}.svg) · '
        f'[600 dpi TIFF]({stem}.tiff) · [Caption](caption.md)\n\n'
        f'[96 dpi reading preview](preview/{stem}_final_size_183mm.png) · '
        '[Code connections and excerpts](source_data/README.md) · [Reproduction and QA](reproduction/)\n\n'
        'This is an implementation schematic, not a numerical experiment, spatial map or validation result. '
        'Boxes and arrows do not encode stock sizes or flux magnitudes. Solid arrows show selected physical/event '
        'routes; dashed arrows show dependencies or native model coupling. The complete stock/flux definitions and '
        'the fixed-calibration production reference are documented in the source bundle.\n')
    print(json.dumps(qa));plt.close(fig)


if __name__=='__main__':main()
