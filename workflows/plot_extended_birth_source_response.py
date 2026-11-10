#!/usr/bin/env python3
"""Adapt completed birth-source figure arrays into Extended Data Fig. 5."""
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
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

WIDTH_MM = 183
HEIGHT_MM = 156
SCREEN_PPI = 96
NAME = 'ed05_birth_source_response'
BACKGROUND = {'VL':'#176A78','LN':'#795B86'}
FUTURE, EXISTING, INTERACTION = '#BC7B30','#739AA0','#8B7399'
INK, MUTED = '#243039','#77838B'
mpl.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Arial'],
    'font.size':7, 'axes.labelsize':6.5, 'axes.titlesize':7,
    'xtick.labelsize':6, 'ytick.labelsize':6,
    'svg.fonttype':'none', 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':.6, 'xtick.major.width':.55, 'ytick.major.width':.55,
    'xtick.major.size':2.5, 'ytick.major.size':2.5,
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
    'legend.frameon':False, 'figure.facecolor':'white', 'axes.facecolor':'white',
})


def ax_mm(fig,x,top,w,h):
    return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])


def text_mm(fig,x,top,text,**kwargs):
    return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,text,va='top',**kwargs)


def panel(fig,x,top,letter,title):
    text_mm(fig,x,top,letter,fontweight='bold',fontsize=8)
    text_mm(fig,x+4.5,top+.1,title,fontsize=7)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,default=Path('outputs/age_module/main_figure_redesign/figures/ed05_birth_source_response/source_data'))
    ap.add_argument('--output',type=Path,default=Path('outputs/age_module/main_figure_redesign/figures/ed05_birth_source_response'))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'preview').mkdir(exist_ok=True)
    members=pd.read_csv(args.source_dir/'source_member_contrasts_all500.csv')
    curves=pd.read_csv(args.source_dir/'source_event_aligned_growth_summary.csv')
    summaries=json.loads((args.source_dir/'source_display_statistics.json').read_text())
    source=json.loads((args.source_dir/'source_analysis_summary.json').read_text())
    if source['status']!='complete' or len(members)!=1000 or source['members_per_condition']!=500:
        raise ValueError('The complete matched four-arm original-member source is required.')
    # These are the original figure arrays, not a new endpoint or mechanism analysis.
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    panel(fig,4,5,'a','Factorial design')
    a=ax_mm(fig,5,22,47,40)
    a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    a.text(.39,1.03,'Existing\nin 2022',ha='center',va='bottom',fontsize=6.3)
    a.text(.78,1.03,'Established\nafter 2022',ha='center',va='bottom',fontsize=6.3)
    mode_notes={'00':'(none)','10':'(existing only)','01':'(later only)','11':'(all cohorts)'}
    for row,mode in enumerate(('00','10','01','11')):
        y=.84-row*.22
        a.text(.04,y+.035,mode,ha='left',va='center',fontsize=7,fontweight='bold')
        a.text(.04,y-.045,mode_notes[mode],ha='left',va='center',fontsize=5.2)
        for x,enabled,color in ((.39,mode[0]=='1',EXISTING),(.78,mode[1]=='1',FUTURE)):
            a.add_patch(Circle((x,y),.055,facecolor=color if enabled else 'white',edgecolor=color,lw=.8))
        a.plot([.2,.95],[y-.11,y-.11],color='#E6EAEC',lw=.4)
    text_mm(fig,5,67,'Filled: age dependence; open: multiplier = 1',fontsize=6)
    text_mm(fig,5,72,'E0 = 10 − 00; N0 = 01 − 00\nN1 = 11 − 10; I = N1 − N0',fontsize=6,linespacing=1.45)

    panel(fig,61,5,'b','Effects at the peak of simulation 00')
    b=ax_mm(fig,74,20,99,51)
    values=members[['E0_T_at_p00_K','N0_T_at_p00_K']].values*1000
    lo=float(np.floor(min(0.,values.min())/10)*10-5)
    hi=float(np.ceil(values.max()/20)*20+10)
    b.set_xscale('asinh',linear_width=20);b.set_yscale('asinh',linear_width=20)
    b.plot([lo,hi],[lo,hi],color=MUTED,lw=.65,ls=(0,(3,2)),zorder=1)
    for bg,color in BACKGROUND.items():
        g=members.loc[members.scenario.eq(bg)]
        if len(g)!=500: raise ValueError('A background has fewer than 500 original members.')
        st=summaries[bg]
        b.scatter(g.E0_T_at_p00_K*1000,g.N0_T_at_p00_K*1000,s=4.2,alpha=.30,c=color,
                  marker='o' if bg=='VL' else '^',linewidths=0,zorder=2,rasterized=False)
        b.plot(st['peak']['E0']['mean'],st['peak']['N0']['mean'],marker='D',ms=4.8,
               mfc=color,mec='white',mew=.6,zorder=4)
    ticks=[v for v in (-100,-25,0,25,50,100,200,400,800) if lo<=v<=hi]
    b.set(xlim=(lo,hi),ylim=(lo,hi),xticks=ticks,yticks=ticks,
          xlabel='Existing-cohort effect E0 (°C)',ylabel='Later-cohort effect N0 (°C)')
    # Axes stay in mK (asinh, linear width 20 mK); tick labels are shown in °C.
    deg_c=mpl.ticker.FuncFormatter(lambda v,_:f'{v/1000:g}'.replace('-', '−'))
    b.xaxis.set_major_formatter(deg_c)
    b.yaxis.set_major_formatter(deg_c)
    b.xaxis.labelpad=2;b.yaxis.labelpad=2
    b.text(.025,.97,'Later-cohort effect larger',transform=b.transAxes,va='top',fontsize=6,color=MUTED)
    for i,(bg,color) in enumerate(BACKGROUND.items()):
        text_mm(fig,75+i*51,82,f'{bg}: {summaries[bg]["future_larger_n"]/5:.1f}% above equality',fontsize=6.2,color=color)

    panel(fig,4,93,'c','Growth of the warming difference after the peak')
    legend=[Patch(fc=FUTURE,alpha=.35,label='Later cohorts N0'),Patch(fc=EXISTING,alpha=.55,label='Existing E0'),
            Patch(fc=INTERACTION,alpha=.7,label='Interaction I'),Line2D([],[],color=INK,lw=1.1,label='Total'),
            Line2D([],[],color=FUTURE,lw=.8,ls=(0,(3,2)),label='N1 (11 − 10)')]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(15/WIDTH_MM,1-99/HEIGHT_MM),
               ncol=5,fontsize=5.8,handlelength=1.2,handletextpad=.35,columnspacing=.8)
    for i,bg in enumerate(BACKGROUND):
        left=15+i*89
        c=ax_mm(fig,left,111,70,33)
        parts={key:curves.loc[curves.scenario.eq(bg)&curves.contrast.eq(key)].sort_values('years_after_p00')
               for key in ('E0','N0','N1','I','total')}
        x=parts['N0'].years_after_p00.to_numpy()
        n0=parts['N0']['mean'].to_numpy();e0=parts['E0']['mean'].to_numpy();inter=parts['I']['mean'].to_numpy()
        c.fill_between(x,0,n0,color=FUTURE,alpha=.28,linewidth=0)
        c.fill_between(x,n0,n0+e0,color=EXISTING,alpha=.55,linewidth=0)
        c.fill_between(x,n0+e0,n0+e0+inter,color=INTERACTION,alpha=.7,linewidth=0)
        c.plot(x,n0,color=FUTURE,lw=.9)
        c.plot(x,parts['N1']['mean'],color=FUTURE,lw=.7,ls=(0,(3,2)))
        c.plot(x,parts['total']['mean'],color=INK,lw=1.1)
        c.axhline(0,color=MUTED,lw=.5)
        c.set(xlim=(0,20),ylim=(-3,24),xticks=[0,10,20],yticks=[0,10,20],
              xlabel='Years after the peak of simulation 00',ylabel='Growth of difference (°C)' if i==0 else '')
        c.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f'{v/1000:g}'.replace('-', '−')))  # mK data shown in °C
        c.xaxis.labelpad=1.5;c.yaxis.labelpad=1.5
        c.text(.025,.96,bg,transform=c.transAxes,va='top',fontsize=7,color=BACKGROUND[bg])
        value=float(parts['total']['mean'].iloc[-1])
        c.annotate(f'{value/1000:.3f} °C'.replace('-', '−'),xy=(20,value),xytext=(-5,4),textcoords='offset points',
                   ha='right',va='bottom',fontsize=6.2,color=INK)

    fig.savefig(args.output/(NAME+'.png'),dpi=300)
    fig.savefig(args.output/(NAME+'.svg'))
    fig.savefig(args.output/(NAME+'.pdf'))
    fig.savefig(args.output/(NAME+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(args.output/'preview'/(NAME+'_final_size_183mm.png'),dpi=SCREEN_PPI)
    plt.close(fig)
    result=dict(status='complete',schema='ed05-birth-source-figure-v1',figure='Extended Data Fig. 5',
                created_utc=datetime.now(timezone.utc).isoformat(),width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
                source_dir=str(args.source_dir),plan_identity=source['plan_identity'],
                history_run_identity=source['history_run_identity'],scenarios=list(BACKGROUND),
                original_members_each=500,source_member_rows=1000,source_four_arm_rows=4000,
                reanalysis='None: original source arrays, centers and qualification flags reused unchanged.',
                adaptation='Removed independent interaction panel; I remains in c additive means and all Source Data.',
                peak_reference='All-500 p00 is the 00 window-maximum year, not an overshoot peak for every member.',
                followup_support={bg:{k:summaries[bg][k] for k in ('n_followup','n_unavailable','n_resolved','n_resolved_followup')} for bg in BACKGROUND},
                statistical_centers='b diamonds: means; c signed component and total means over the same qualified group.',
                limits=source['limits'],exclusions='No all-500 rows removed; c uses the original fixed support predicate.')
    (args.output/'figure_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',output=str(args.output),plan_identity=source['plan_identity'])))


if __name__=='__main__':
    main()
