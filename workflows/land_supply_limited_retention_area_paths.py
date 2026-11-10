"""Discrete land-service consequences of cancelling young-secondary parcels.

All original members evolve together; no carbon or climate is integrated.
Uniform joint allocation is the same proportional native management rule.
All original same-source nonforest requests share one explicit supply factor.
Incoming land is not reusable within the annual event.
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import xarray as xr
from workflows.run_future_age_response import write_json
from workflows.run_age_peak_mechanism_diagnostics import netcdf
from workflows.run_main_age_diagnostics import load_batch


def area_paths(plan, root, out):
    _,_,_,layout,_=load_batch(plan,plan["cells"][0],slice(0,1),age_on=True)
    classes=[c.class_id for c in layout.classes]
    roles={c.class_id:c.role for c in layout.classes}
    at={s:i for i,s in enumerate(layout.slots)}
    columns={c:np.array([i for i,(s,_) in enumerate(layout.slots) if s==c]) for c in classes}
    forest_columns=np.array([i for i,(c,_) in enumerate(layout.slots) if roles[c] in ("forest_unaged","forest_cohort")])
    shape=(500,len(layout.regions),len(layout.slots))
    initial=np.zeros(shape)
    with xr.open_dataset(root/plan["history_directory"]/"final_state.nc") as d:
        if d.source_config.values.tolist()!=plan["source_config"]:raise RuntimeError("Original area member order changed")
        initial[:,:,at[(layout.primary_class,None)]]=d.rf_A_primary.transpose("config","reg_land").values
        for c in layout.nonforest_classes:
            initial[:,:,at[(c,None)]]=d.rf_A_nonforest.sel(nonforest_class=c).transpose("config","reg_land").values
        a=d.rf_A_forest.reindex(birth_time=list(layout.births),fill_value=0.)
        for c in layout.forest_classes:
            initial[:,:,columns[c]]=a.sel(forest_class=c).transpose("config","reg_land","birth_time").values
    start_class=np.stack([initial[:,:,columns[c]].sum(axis=-1) for c in classes],axis=-1)
    nonforest=[c for c in classes if roles[c]=="nonforest"]
    directory=out/"area_consequences";directory.mkdir(exist_ok=False)
    all_flags=[];summaries=[]
    for cell in plan["cells"]:
        with xr.open_dataset(root/cell["forcing"]) as f:
            demands=f.event_area.sel(year=slice(2023,2099)).transpose("year","reg_land","event").values.copy()
            if tuple(f.reg_land.values)!=layout.regions or tuple(f.event.values)!=tuple(e.event_id for e in layout.events):
                raise RuntimeError("Area forcing identity differs from the native layout")
        for ei,e in enumerate(layout.events):
            if e.event_id in ("secmf_harv","secyf_harv"):demands[:,:,ei]=0.
        flat=initial.copy();valid=np.ones(500,dtype=bool);failures=[None]*500
        baseline=start_class.copy()
        actual=np.full((77,500,len(layout.regions),len(classes)),np.nan)
        retained=np.full((77,500,len(layout.regions)),np.nan)
        cancelled=np.full_like(actual,np.nan)
        unmet_source=np.full((77,500,len(layout.regions),len(nonforest)),np.nan)
        unmet_destination=np.full_like(actual,np.nan)
        for yi,y in enumerate(range(2023,2100)):
            time_event=y-.5;raw_demand=demands[yi]
            demand=np.broadcast_to(raw_demand[None],(500,)+raw_demand.shape).copy()
            for ei,e in enumerate(layout.events):
                baseline[:,:,classes.index(e.source_class)]-=raw_demand[None,:,ei]
                baseline[:,:,classes.index(e.target_class)]+=raw_demand[None,:,ei]
            unmet_source_y=np.zeros((500,len(layout.regions),len(nonforest)))
            unmet_destination_y=np.zeros((500,len(layout.regions),len(classes)))
            for ci,c in enumerate(nonforest):
                rows=[i for i,e in enumerate(layout.events) if e.source_class==c]
                opening_source=flat[:,:,columns[c]].sum(axis=-1)
                total_request=demand[:,:,rows].sum(axis=-1)
                alpha=np.minimum(1.,np.divide(opening_source,total_request,out=np.ones_like(opening_source),where=total_request>0))
                demand[:,:,rows]*=alpha[:,:,None]
                unmet_source_y[:,:,ci]=total_request-demand[:,:,rows].sum(axis=-1)
                for ei in rows:
                    unmet_destination_y[:,:,classes.index(layout.events[ei].target_class)]+=raw_demand[None,:,ei]-demand[:,:,ei]
            available=flat.copy();incoming=np.zeros_like(flat)
            cancelled_y=np.zeros((500,len(layout.regions),len(classes)))
            for kind,source,rows,cols in layout.event_index_groups:
                d=demand[:,:,rows];total=d.sum(axis=-1)
                if not np.any(total):continue
                opening=available[:,:,cols]
                capacity=opening.sum(axis=-1)
                # The native joint-area solver uses 1e-10*max(1,capacity).
                bad=(((capacity<=0.) & (total>0.)) |
                     (total>capacity+1e-10*np.maximum(1.,capacity))) & valid[:,None]
                for mi in np.flatnonzero(bad.any(axis=-1)):
                    failures[mi]=dict(error="joint activity demands exceed eligible source area",
                        first_failure=[dict(year=y,event_time=time_event,region=layout.regions[ri],
                            source_class=source,events=[layout.events[i].event_id for i in rows],
                            requested_Mha=float(total[mi,ri]),available_Mha=float(capacity[mi,ri]),
                            shortage_Mha=float(total[mi,ri]-capacity[mi,ri])) for ri in np.flatnonzero(bad[mi])])
                valid[bad.any(axis=-1)]=False
                fraction=np.minimum(1.,np.divide(total,capacity,out=np.zeros_like(capacity),where=capacity>0))
                proposed=opening*fraction[:,:,None]
                proposed[~valid]=0.
                take=proposed.copy()
                if source=="secdf" and kind=="conversion":
                    young=np.array([max(0.,time_event-layout.slots[i][1])<cell["cutoff_years"] for i in cols])
                    take[:,:,young]=0.
                available[:,:,cols]-=take
                total_taken=take.sum(axis=-1)
                total_cancelled=(proposed-take).sum(axis=-1)
                shares=np.divide(d,total[:,:,None],out=np.zeros_like(d),where=total[:,:,None]>0)
                for j,ei in enumerate(rows):
                    e=layout.events[ei];share=shares[:,:,j]
                    if e.age_action=="keep":
                        for k,si in enumerate(cols):
                            ti=at[(e.target_class,layout.slots[si][1])]
                            incoming[:,:,ti]+=take[:,:,k]*share
                    else:
                        birth=time_event if roles[e.target_class]=="forest_cohort" else None
                        incoming[:,:,at[(e.target_class,birth)]]+=total_taken*share
                    if source=="secdf" and kind=="conversion":
                        cancelled_y[:,:,classes.index(e.target_class)]+=total_cancelled*share
            flat=available+incoming
            for ci,c in enumerate(classes):actual[yi,valid,:,ci]=flat[valid][:,:,columns[c]].sum(axis=-1)
            baseline_forest=baseline[:,:,[roles[c] in ("forest_unaged","forest_cohort") for c in classes]].sum(axis=-1)
            retained[yi,valid]=flat[valid][:,:,forest_columns].sum(axis=-1)-baseline_forest[valid]
            cancelled[yi,valid]=cancelled_y[valid]
            unmet_source[yi,valid]=unmet_source_y[valid]
            unmet_destination[yi,valid]=unmet_destination_y[valid]
            if not valid.any():break
        for mi in range(500):
            all_flags.append(dict(cell=cell["cell"],scenario=cell["scenario"],cutoff_years=cell["cutoff_years"],
                member_position=mi,source_config=plan["source_config"][mi],status="area_feasible" if valid[mi] else "area_infeasible",first_failure=failures[mi]))
        ds=xr.Dataset(coords=dict(year=np.arange(2023,2100),config=np.arange(500),
            source_config=("config",plan["source_config"]),reg_land=list(layout.regions),luh_class=classes,nonforest_class=nonforest))
        ds["actual_class_area_Mha"] = (("year","config","reg_land","luh_class"),actual)
        ds["forest_area_above_unprotected_baseline_Mha"] = (("year","config","reg_land"),retained)
        ds["cancelled_secondary_conversion_Mha"] = (("year","config","reg_land","luh_class"),cancelled)
        ds["nonforest_unmet_source_gross_Mha"] = (("year","config","reg_land","nonforest_class"),unmet_source)
        ds["nonforest_unmet_destination_gross_Mha"] = (("year","config","reg_land","luh_class"),unmet_destination)
        ds.attrs.update(schema=plan["schema"],plan_identity=plan["plan_identity"],cell=cell["cell"],
            time_support="Actual end-of-event area at year-0.5; cancellation is gross event total, not a time mean",
            area_only="500 original area states; uniform native joint proposal then cancellation; no carbon/climate integration; standing harvest retains area/birth",
            support="Original IDs all retained; one explicit same-source nonforest supply fraction; no replacement; any remaining forest-source shortage leaves later outcomes undefined")
        netcdf(ds,directory/(cell["cell"]+".nc"))
        summary=dict(cell=cell["cell"],area_feasible=bool(valid.all()),feasible_members=int(valid.sum()),infeasible_members=int((~valid).sum()))
        summaries.append(summary)
        write_json(out/"area_feasibility.json",dict(schema=plan["schema"],plan_identity=plan["plan_identity"],
            status="running" if len(summaries)<8 else "complete",cells=summaries,members=all_flags,
            method="All 500 actual area vectors, no rounding/grouping/member filtering; native uniform joint rule, explicit nonforest service limitation and native numerical supply tolerance",completed_at_utc=datetime.now(timezone.utc).isoformat()))
        print(summary,flush=True)
    return summaries
