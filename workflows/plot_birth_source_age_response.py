#!/usr/bin/env python3
"""Source-selective forest-age responses at a shared reference year and after it."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr

WIDTH_MM=183
HEIGHT_MM=198
SCREEN_PPI=96
NAME='birth_source_age_response'
BG={'VL':'#176A78','LN':'#795B86'}
FUTURE,EXISTING,INTERACTION='#BC7B30','#739AA0','#8B7399'
INK,MUTED='#243039','#77838B'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
 'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.55,'xtick.major.width':.55,'ytick.major.width':.55,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':INK,
 'legend.frameon':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def require(ok,message):
    if not ok:raise ValueError(message)


def jwrite(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def ax_mm(fig,x,top,w,h):
    return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])


def text_mm(fig,x,top,text,**kwargs):
    return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,text,va='top',**kwargs)


def panel(fig,x,top,letter,title):
    text_mm(fig,x,top,letter,fontweight='bold',fontsize=8)
    text_mm(fig,x+4.5,top+.1,title,fontsize=7)


def stats(v):
    v=np.asarray(v,dtype=float)
    require(len(v)>0 and np.isfinite(v).all(),'Defined population has missing values')
    q=np.quantile(v,[.05,.5,.95])
    return dict(n=len(v),mean=float(v.mean()),q05=float(q[0]),median=float(q[1]),q95=float(q[2]),
        min=float(v.min()),max=float(v.max()),positive=int((v>0).sum()),negative=int((v<0).sum()))


def prepare(analysis,out):
    s=json.loads((analysis/'summary.json').read_text())
    require(s.get('status')=='complete' and s.get('members_per_condition')==500,
        'A complete original-500 matched source analysis is required')
    members=pd.read_csv(analysis/'member_source_contrasts_all500.csv')
    arms=pd.read_csv(analysis/'arm_members_all500.csv')
    require(len(members)==1000 and len(arms)==4000,'Original four-arm populations are incomplete')
    annual=[];summaries={}
    with xr.open_dataset(analysis/'source_annual_four_arms_all500.nc') as ds:
        ids=ds.source_config.values.tolist()
        require(ds.attrs['plan_identity']==s['plan_identity'],'Thin annual source has a different identity')
        for bg in BG:
            g=members.loc[members.scenario.eq(bg)].sort_values('member_position')
            require(len(g)==500 and g.source_config.tolist()==ids,'Original ordered IDs differ')
            use=g.reference00_plus20_defined.eq(True).to_numpy()
            times=np.asarray(g.p00_year,dtype=int)[use]-2022
            trajectories={mode:ds.T_1850_1900_K.sel(scenario=bg,selector=mode).transpose('year','config').values[:,use]
                          for mode in ('00','10','01','11')}
            for tau in range(21):
                changes={m:t[times+tau,np.arange(use.sum())]-t[times,np.arange(use.sum())] for m,t in trajectories.items()}
                n0=changes['01']-changes['00'];n1=changes['11']-changes['10']
                e0=changes['10']-changes['00'];inter=n1-n0
                for label,values in dict(N0=n0,N1=n1,E0=e0,I=inter,total=changes['11']-changes['00']).items():
                    annual.append(dict(scenario=bg,years_after_p00=tau,contrast=label,**stats(values*1000)))
            summaries[bg]=dict(n_all=500,n_followup=int(use.sum()),n_unavailable=int((~use).sum()),
                n_resolved=int(g.reference00_resolved_overshoot.sum()),
                n_resolved_followup=int((g.reference00_resolved_overshoot&g.reference00_plus20_defined).sum()),
                future_larger_n=int((g.N0_T_at_p00_K>g.E0_T_at_p00_K).sum()),
                peak={k:stats(g[k+'_T_at_p00_K']*1000) for k in ('N0','N1','E0','E1','I','total')},
                growth={k:stats(g.loc[use,k+'_gap20_K']*1000) for k in ('N0','N1','E0','E1','I','total')},
                resolved_growth={k:stats(g.loc[use&g.reference00_resolved_overshoot.to_numpy(),k+'_gap20_K']*1000)
                                 for k in ('N0','N1','E0','I','total')})
    members.to_csv(out/'source_member_contrasts_all500.csv',index=False)
    arms.to_csv(out/'source_four_arm_members_all500.csv',index=False)
    curves=pd.DataFrame(annual);curves.to_csv(out/'source_event_aligned_growth_summary.csv',index=False)
    jwrite(out/'source_analysis_summary.json',s);jwrite(out/'source_display_statistics.json',summaries)
    return members,curves,s,summaries,ids


def draw(members,curves,summaries):
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    text_mm(fig,4,4,'Age responses in new forest cohorts sustain the post-peak temperature gap',fontsize=8.2,fontweight='bold')
    text_mm(fig,4,10.6,'Matched VL/LN futures  |  Shared 2022 initial state  |  Original land-activity timing',fontsize=6.2,color=MUTED)
    panel(fig,4,20,'a','Assign age-production response')
    a=ax_mm(fig,5,29,47,42);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
    a.text(.39,1.01,'Existing\nin 2022',ha='center',va='bottom',fontsize=6)
    a.text(.78,1.01,'Born\nafter 2022',ha='center',va='bottom',fontsize=6)
    for row,mode in enumerate(('00','10','01','11')):
        y=.84-row*.22
        a.text(.04,y,mode,ha='left',va='center',fontsize=7,fontweight='bold')
        for x,enabled,color in ((.39,mode[0]=='1',EXISTING),(.78,mode[1]=='1',FUTURE)):
            a.add_patch(Circle((x,y),.055,facecolor=color if enabled else 'white',edgecolor=color,lw=.8))
        a.plot([.2,.95],[y-.11,y-.11],color='#E6EAEC',lw=.4)
    text_mm(fig,5,74,'Filled: original response · open: g = 1\nN0 = 01 − 00; N1 = 11 − 10\nE0 = 10 − 00; I = N1 − N0',fontsize=5.9,linespacing=1.5)

    panel(fig,59,20,'b','At the reference maximum year, both sources contribute')
    b=ax_mm(fig,71,29,102,44)
    vals=members[['E0_T_at_p00_K','N0_T_at_p00_K']].values*1000
    lo=float(np.floor(min(0.,vals.min())/10)*10-5)
    hi=float(np.ceil(vals.max()/20)*20+10)
    b.set_xscale('asinh',linear_width=20);b.set_yscale('asinh',linear_width=20)
    b.plot([lo,hi],[lo,hi],color=MUTED,lw=.65,ls=(0,(3,2)),zorder=1)
    for bg,color in BG.items():
        g=members.loc[members.scenario.eq(bg)];st=summaries[bg]
        b.scatter(g.E0_T_at_p00_K*1000,g.N0_T_at_p00_K*1000,s=4.2,alpha=.30,c=color,
                  marker='o' if bg=='VL' else '^',linewidths=0,zorder=2)
        b.plot(st['peak']['E0']['mean'],st['peak']['N0']['mean'],marker='D',ms=4.8,mfc=color,mec='white',mew=.6,zorder=4)
    ticks=[v for v in (-100,-25,0,25,50,100,200,400,800) if lo<=v<=hi]
    b.set(xlim=(lo,hi),ylim=(lo,hi),xticks=ticks,yticks=ticks,
        xlabel='Existing-cohort response E0 (mK)',ylabel='New-cohort response N0 (mK)')
    b.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
    b.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
    b.xaxis.labelpad=1.8;b.yaxis.labelpad=1.8
    b.text(.025,.97,'New-cohort effect larger',transform=b.transAxes,va='top',fontsize=5.7,color=MUTED)
    for i,(bg,color) in enumerate(BG.items()):
        text_mm(fig,71+i*52,82,f'{bg}  {summaries[bg]["future_larger_n"]}/500 above equality',fontsize=6,color=color)
    text_mm(fig,71,86,'All 500 each · diamonds: means · asinh spacing on both axes',fontsize=5.7,color=MUTED)

    panel(fig,4,96,'c','Subsequent gap growth is sustained mainly by post-2022 cohorts')
    for i,bg in enumerate(BG):
        left=15+i*89;st=summaries[bg]
        c=ax_mm(fig,left,106,70,32)
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
            xlabel='Years after reference year p00',ylabel='Further gap (mK)' if i==0 else '')
        c.xaxis.labelpad=1.5;c.yaxis.labelpad=1.5
        c.text(.02,.95,f'{bg} · n = {st["n_followup"]}',transform=c.transAxes,va='top',fontsize=6.4,color=BG[bg])
        text_mm(fig,left,145,f'20-year means: N0 {st["growth"]["N0"]["mean"]:+.2f} · E0 {st["growth"]["E0"]["mean"]:+.2f} mK',fontsize=5.8)
        text_mm(fig,left,149,f'I {st["growth"]["I"]["mean"]:+.3f} · total {st["growth"]["total"]["mean"]:+.2f} mK',fontsize=5.8,color=MUTED)
    text_mm(fig,15,101,'Filled components: N0 (amber) + E0 (blue) + I (violet) = total (black); dashed amber: N1',fontsize=5.7,color=MUTED)

    panel(fig,4,158,'d','The two conditional new-cohort effects are close at p00')
    d=ax_mm(fig,21,167,107,15)
    all_i=members.I_T_at_p00_K.to_numpy()*1000
    low=float(np.floor(all_i.min()*10)/10-.02);high=float(np.ceil(all_i.max()*10)/10+.02)
    d.set_xscale('asinh',linear_width=.08)
    for i,(bg,color) in enumerate(BG.items()):
        g=members.loc[members.scenario.eq(bg)].sort_values('member_position');st=summaries[bg]['peak']['I'];y=1-i
        jit=(np.arange(500)%23-11)/23*.32
        d.scatter(g.I_T_at_p00_K*1000,y+jit,s=2,c=color,alpha=.3,linewidths=0)
        d.plot([st['q05'],st['q95']],[y,y],color=color,lw=1.3)
        d.plot(st['median'],y,'D',ms=3.5,mfc=color,mec='white',mew=.4)
        d.plot(st['mean'],y,'s',ms=3.6,mfc='white',mec=INK,mew=.65)
        text_mm(fig,134,167+i*7.8,f'{bg} I median {st["median"]:+.3f} mK\n5–95: {st["q05"]:+.3f} to {st["q95"]:+.3f}',fontsize=5.6,linespacing=1.4)
    d.axvline(0,color=MUTED,lw=.55,ls=(0,(3,2)))
    d.set(xlim=(low,high),ylim=(-.45,1.45),yticks=[0,1],yticklabels=['LN','VL'],
        xticks=[x for x in (-10,-1,-.1,0,.1,1,10) if low<=x<=high],
        xlabel='Interaction I = N1 − N0 (mK; asinh spacing)')
    d.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
    d.xaxis.labelpad=1.4;d.tick_params(axis='y',length=0);d.spines['left'].set_visible(False)
    text_mm(fig,21,193,'Within-condition parameter distributions; diamonds: medians, open squares: means, lines: 5–95%.',fontsize=5.6,color=MUTED)
    return fig


def caption(st):
    vl,ln=st['VL'],st['LN']
    return f'''Age-production responses in post-2022 forest cohorts sustain later temperature-gap growth.

a Source-selective production responses: 00 uses unmodulated host production (gW=gF=1); 11 retains both prescribed responses; 10 applies them only to birth_time ≤2022; 01 only to birth_time >2022. Primary forest remains unaged. N0=01−00 and N1=11−10 hold existing-cohort treatment fixed within each pair; E0=10−00 and I=N1−N0. Sources identify birth cohorts that subsequently age, not carbon-atom ancestry or exclusively restoration or clear-felling.

b Same-member E0 and N0 temperature contrasts at p00, each member’s 00 maximum year in 2022–2099. Every original member is shown (500 per pathway); diamonds are means and the dashed line is equality. N0 exceeds E0 in {vl['future_larger_n']}/500 VL and {ln['future_larger_n']}/500 LN. Both axes use asinh spacing (20-mK linear width); no tails are removed. p00 is a resolved overshoot peak for {vl['n_resolved']}/{ln['n_resolved']} members, not all 500.

c Mean additional contrast growth from p00 to p00+20 among eligible interior-maximum members ({vl['n_followup']}/{ln['n_followup']}; {vl['n_unavailable']}/{ln['n_unavailable']} lack follow-up). Filled N0, E0 and I add to the black total; dashed amber is N1. Means use the same members, not typical trajectories or normalized shares. Resolved-overshoot subsets with 20-year support ({vl['n_resolved_followup']}/{ln['n_resolved_followup']}) retain N0 growth medians {vl['resolved_growth']['N0']['median']:+.2f}/{ln['resolved_growth']['N0']['median']:+.2f} mK, and E0 {vl['resolved_growth']['E0']['median']:+.2f}/{ln['resolved_growth']['E0']['median']:+.2f} mK. Return to 1.5 K is not required.

d All-500 interaction distributions at p00. Points retain every tail; diamonds show medians, open squares means and lines empirical 5th–95th percentiles. Asinh linear width is 0.08 mK.

VL and LN retain fully matched CMIP7 drivers, original land-activity timing and the same complete 2022 state. Temperatures use the common member-wise 1850–1900 reference. Effects are relative to the specified host-production reference, not a no-forest comparison. Ranges describe joint-parameter variation, not confidence intervals or pathway probabilities. Source Data preserve all arms, statuses and flags. No four-arm radiative attribution is inferred.
'''


def export(fig,out,preview_only):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];ignore=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo<=tick.get_loc()<=hi:ignore.update((tick.label1,tick.label2))
    for item in fig.findobj(mpl.text.Text):
        if item in ignore or not item.get_visible() or not item.get_text().strip():continue
        bb=item.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:outside.append(item.get_text())
    require(not outside,'Text outside page: '+str(outside))
    stem=out/NAME
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    fig.savefig(out/'final_size_183mm.png',dpi=SCREEN_PPI)
    if not preview_only:
        fig.savefig(stem.with_suffix('.svg'));fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        qa=dict(size_mm=[WIDTH_MM,HEIGHT_MM],backend='Python matplotlib',parameter_members_per_condition=500,
            image_manipulation='None',temperature_units='mK')
        for ext in ('png','tiff'):
            with Image.open(stem.with_suffix('.'+ext)) as im:
                qa[ext]=dict(pixels=list(im.size),dpi=[float(x) for x in im.info.get('dpi',[])],compression=im.info.get('compression'))
        svg=stem.with_suffix('.svg').read_text();qa.update(svg_text_objects=svg.count('<text'),svg_raster_images=svg.count('<image'))
        require(qa['svg_text_objects']>0 and qa['svg_raster_images']==0,'Editable vector output required')
        jwrite(out/'export_QA.json',qa)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true');args=p.parse_args()
    analysis=args.analysis_dir.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    members,curves,source,summary,ids=prepare(analysis,out)
    jwrite(out/'figure_manifest.json',dict(schema='birth-source-age-response-figure-v1',
        generated_utc=datetime.now(timezone.utc).isoformat(),analysis_dir=str(analysis),
        comparison_family=source['comparison_family'],plan_identity=source['plan_identity'],
        history_run_identity=source['history_run_identity'],scenarios=list(BG),source_config=ids,
        original_members_each=500,limits=source['limits']))
    (out/'caption.txt').write_text(caption(summary))
    (out/'README.md').write_text('''# Forest birth-source responses in matched overshoot futures

Four source-selective arms distinguish the conditional effect of age-production responses in cohorts existing by 2022 and cohorts born later. Original matched VL and LN futures and the common complete 2022 state are retained. Original 00/11 results are reused. This family is separate from controlled restoration comparisons.

Reproduce: .venv/bin/python -B workflows/plot_birth_source_age_response.py --analysis-dir outputs/age_module/birth_source_age_response/analysis --output <new figure directory>. The plot integrates no model. See caption.txt and figure_manifest.json for source identity and population definitions. Source Data retain all 1000 member contrasts and 4000 four-arm member rows; event-aligned mean/parameter summaries use only defined follow-up support without changing the all-500 analyses.

Exports: 183×198 mm; editable SVG/PDF; 600-dpi LZW TIFF; 300-dpi PNG; 96-ppi physical-width preview. No radiative component or carbon-atom source allocation is inferred.
''')
    export(draw(members,curves,summary),out,args.preview_only)
    if Path(__file__).resolve()!=out/Path(__file__).name:shutil.copyfile(__file__,out/Path(__file__).name)
    print('EXPORTED',out/NAME,flush=True)


if __name__=='__main__':main()
