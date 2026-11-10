"""Prepare absolute R10 LUH inputs by accumulating observed area events from 850."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import xarray as xr

CLASSES = ('primf','primn','secdf','secdn','urban','c3ann','c4ann','c3per','c4per','c3nfx','pastr','range','pltns')
HARVEST = ('primf','primn','secmf','secyf','secnf','pltns')
EPS = np.finfo(np.float64).eps


def check(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_initial(raw, land, support):
    positive = np.maximum(raw, 0.)
    total = positive.sum(axis=0)
    check(not np.any(support & (land > 0) & (total <= 0)), 'Positive land has no initial class')
    factor = np.zeros_like(total)
    np.divide(land, total, out=factor, where=support & (total > 0))
    result = positive * factor[None]
    remainder = land - result.sum(axis=0)
    largest = result.argmax(axis=0)
    np.put_along_axis(result, largest[None], np.take_along_axis(result, largest[None], axis=0) + remainder[None], axis=0)
    check(np.all(result >= 0), 'Initial normalization produced negative area')
    check(np.max(np.abs(result.sum(axis=0) - land)) <= 8 * EPS, 'Initial land partition did not close')
    return result


def region_sum(extensive, weights):
    coarse = extensive.reshape(180,4,360,4).sum(axis=(1,3), dtype=np.float64)
    result = np.einsum('ij,ijr->r', coarse, weights, optimize=False)
    bound = 64 * EPS * max(1., float(np.abs(extensive).sum()))
    check(abs(result.sum() - extensive.sum()) <= bound, 'Region reduction lost extensive quantity')
    return result


def read(ds, name, year, support):
    value = np.asarray(ds[name].isel(time=year-850).values, dtype=np.float64)
    check(np.isfinite(value[support]).all(), f'Missing {name} on land in {year}')
    check(not np.any(np.isfinite(value[~support]) & (value[~support] != 0)), f'Nonzero {name} outside land in {year}')
    return np.where(support, value, 0.)


def integrate(state, transitions, harvest, pairs):
    net = np.zeros_like(state)
    for index, (src, dst) in enumerate(pairs):
        net[:, src] -= transitions[:, index]
        net[:, dst] += transitions[:, index]
    for source, target in [('primf','secdf'),('primn','secdn')]:
        h = harvest[:, HARVEST.index(source)]
        net[:, CLASSES.index(source)] -= h
        net[:, CLASSES.index(target)] += h
    return state + net, net


def prepare(root, spec, out, report):
    # The completed acquisition and four-boundary evidence are accepted, not rehashed.
    acquisition = json.loads((root / spec['acquisition_manifest']).read_text())
    check(acquisition['status'] == 'raw_byte_integrity_passed_arrays_not_examined', 'Raw acquisition incomplete')
    paths = {role: root / next(f['path'] for f in acquisition['files'] if Path(f['path']).name.startswith('multiple-'+role+'_')) for role in ('static','states','transitions')}
    report['acquisition_run'] = acquisition['run_id']
    with xr.open_dataset(root / spec['region_weights']) as rw:
        regions = rw.reg_land.values.tolist()
        weights = np.nan_to_num(rw.region_fraction.values, nan=0.)
        check(weights.shape == (180,360,11), 'Expected accepted R10 conditional weights')
        region_lat, region_lon = rw.lat.values, rw.lon.values
    years = np.arange(850,2025); intervals = years[:-1]
    with xr.open_dataset(paths['static'],decode_times=False) as static, xr.open_dataset(paths['states'],decode_times=False) as states, xr.open_dataset(paths['transitions'],decode_times=False) as events:
        check(np.array_equal(region_lat,states.lat) and np.array_equal(region_lon,states.lon), 'Weight/native-grid alignment differs')
        area = np.asarray(static.carea.values,dtype=np.float64) / 10000.
        ice = np.asarray(static.icwtr.values,dtype=np.float64)
        support = np.isfinite(states.primf.isel(time=0).values)
        check(np.all((ice[support]>=0)&(ice[support]<=1)), 'Static ice/water fraction invalid')
        land = np.where(support,1.-ice,0.)
        transfers = sorted(n for n in events.data_vars if '_to_' in n)
        check(len(transfers)==130, 'Expected 130 original directed transitions')
        pairs = [(CLASSES.index(n.split('_to_')[0]),CLASSES.index(n.split('_to_')[1])) for n in transfers]
        nreg = len(regions)
        state_raw = np.zeros((1175,nreg,13)); state = np.zeros_like(state_raw)
        transition = np.zeros((1174,nreg,130)); transition_clip = np.zeros_like(transition)
        harvest = np.zeros((1174,nreg,6)); bioh_raw = np.zeros_like(harvest); bioh = np.zeros_like(harvest); bioh_correction = np.zeros_like(harvest)
        state_negative_count = np.zeros((1175,13),dtype=np.int64)
        transition_negative_count = np.zeros((1174,130),dtype=np.int64)
        carbon_negative_count = np.zeros((1174,6),dtype=np.int64)
        state_min_raw = np.zeros((1175,13)); transition_min_raw = np.zeros((1174,130))
        carbon_min_raw = np.zeros((1174,6)); partition_max = np.zeros(1175)
        carbon_without_area = np.zeros_like(harvest)
        initial = np.stack([read(states,c,850,support) for c in CLASSES])
        normalized = normalize_initial(initial,land,support)
        report['initial_adjustment'] = {}
        for c,name in enumerate(CLASSES):
            difference = normalized[c]-initial[c]
            state[0,:,c] = region_sum(normalized[c]*area,weights)
            report['initial_adjustment'][name] = {'affected_cells':int(np.count_nonzero(difference)),
                'max_abs_fraction':float(np.max(np.abs(difference))),
                'signed_Mha':float((difference*area).sum()),'absolute_Mha':float((np.abs(difference)*area).sum()),
                'negative_raw_cells':int((initial[c]<0).sum())}
        forest_delta = (normalized[[0,2,12]]-initial[[0,2,12]]).sum(axis=0)
        report['accepted_raw_forest850_weights_not_modified'] = True
        report['initial_forest_adjustment_Mha'] = {'signed':float((forest_delta*area).sum()),'absolute':float((np.abs(forest_delta)*area).sum()),'by_region':region_sum(forest_delta*area,weights).tolist()}
        land_region = region_sum(land*area,weights)
        check(np.max(np.abs(state[0].sum(axis=1)-land_region)) <= 128*EPS*max(1.,float(land_region.max())), 'Initial regional land total differs')
        del normalized, forest_delta
        start = time.monotonic()
        max_step_error = 0.
        for i,year in enumerate(years):
            partition = np.zeros_like(area)
            for c,name in enumerate(CLASSES):
                value = initial[c] if i==0 else read(states,name,int(year),support)
                state_raw[i,:,c] = region_sum(value*area,weights)
                partition += value
                state_negative_count[i,c] = np.count_nonzero(value[support]<0)
                state_min_raw[i,c] = np.min(value[support])
            partition_max[i] = np.max(np.abs(partition[support]-land[support]))
            if i==0:
                del initial
            if i==1174:
                break
            for t,name in enumerate(transfers):
                raw = read(events,name,int(year),support)
                negative = raw < 0
                transition_negative_count[i,t] = np.count_nonzero(negative)
                transition_min_raw[i,t] = np.min(raw[support])
                correction = np.where(negative,-raw,0.)
                clean = raw + correction
                transition[i,:,t] = region_sum(clean*area,weights)
                if negative.any():
                    transition_clip[i,:,t] = region_sum(correction*area,weights)
            for h,name in enumerate(HARVEST):
                fraction = read(events,name+'_harv',int(year),support)
                check(np.min(fraction)>=0, f'Negative harvest area requires a separate data decision: {name}, {year}')
                harvest[i,:,h] = region_sum(fraction*area,weights)
                mass = read(events,name+'_bioh',int(year),support)
                carbon_negative_count[i,h] = np.count_nonzero(mass<0)
                carbon_min_raw[i,h] = np.min(mass[support])
                bioh_raw[i,:,h] = region_sum(mass/1e12,weights)
                bioh[i,:,h] = region_sum(np.maximum(mass,0.)/1e12,weights)
                bioh_correction[i,:,h] = region_sum(np.where(mass<0,-mass,0.)/1e12,weights)
                carbon_without_area[i,:,h] = region_sum(np.where(fraction==0,np.maximum(mass,0.),0.)/1e12,weights)
            state[i+1], net = integrate(state[i],transition[i],harvest[i],pairs)
            step_error = float(np.max(np.abs(state[i+1]-state[i]-net)))
            max_step_error = max(max_step_error,step_error)
            if i%25==0 or i==1173:
                report.update(last_completed_interval_start=int(year), elapsed_seconds=time.monotonic()-start,
                    minimum_cumulative_region_state_Mha=float(state[:i+2].min()),
                    max_deviation_from_published_states_Mha=float(np.max(np.abs(state[:i+1]-state_raw[:i+1]))),
                    cumulative_transition_negative_correction_Mha=float(transition_clip[:i+1].sum()),
                    negative_bioh_correction_PgC=float(bioh_correction[:i+1].sum()))
                save(out/'report.json',report)
                print(year,'minimum area',report['minimum_cumulative_region_state_Mha'],'max published-state deviation',report['max_deviation_from_published_states_Mha'],'elapsed',round(report['elapsed_seconds'],1),flush=True)
        difference = state-state_raw
        land_error = np.abs(state.sum(axis=2)-land_region[None])
        negative_indices = np.argwhere(state<0)
        worst = np.unravel_index(np.argmax(np.abs(difference)),difference.shape)
        report.update(status='complete_pending_data_magnitude_review',elapsed_seconds=time.monotonic()-start,
            max_step_balance_error_Mha=max_step_error,max_total_land_drift_Mha=float(land_error.max()),
            cumulative_state_negative_entries=int(len(negative_indices)),minimum_cumulative_region_state_Mha=float(state.min()),
            maximum_annual_negative_area_Mha=float(np.maximum(-state,0).sum(axis=(1,2)).max()),
            first_negative_state_entries=[{'year':int(years[y]),'region':regions[r],'class':CLASSES[c],'Mha':float(state[y,r,c])} for y,r,c in negative_indices[:20]],
            worst_published_state_deviation={'year':int(years[worst[0]]),'region':regions[worst[1]],'class':CLASSES[worst[2]],'signed_Mha':float(difference[worst])},
            final_signed_difference_Mha_by_class=difference[-1].sum(axis=0).tolist(),
            final_absolute_region_class_difference_Mha=float(np.abs(difference[-1]).sum()),
            maximum_annual_absolute_region_class_difference_Mha=float(np.abs(difference).sum(axis=(1,2)).max()),
            transition_negative_field_cell_occurrences=int(transition_negative_count.sum()),
            cumulative_transition_negative_correction_Mha=float(transition_clip.sum()),
            transition_raw_min_fraction=float(transition_min_raw.min()),
            bioh_negative_field_cell_occurrences=int(carbon_negative_count.sum()),
            negative_bioh_correction_PgC=float(bioh_correction.sum()),
            max_raw_partition_fraction_residual=float(partition_max.max()),
            raw_bioh_not_used_as_model_carbon_demand=True)
        coords={'state_year':years,'interval':intervals,'reg_land':regions,'luh_class':list(CLASSES),'transition':transfers,'harvest_source':list(HARVEST)}
        ds=xr.Dataset(coords=coords)
        ds['interval_start']=('interval',intervals);ds['interval_end']=('interval',intervals+1)
        ds['transition_from']=('transition',[n.split('_to_')[0] for n in transfers]);ds['transition_to']=('transition',[n.split('_to_')[1] for n in transfers])
        variables=[
            ('state_area',state,('state_year','reg_land','luh_class'),'Mha'),
            ('published_state_area',state_raw,('state_year','reg_land','luh_class'),'Mha'),
            ('state_minus_published',difference,('state_year','reg_land','luh_class'),'Mha'),
            ('transition_area',transition,('interval','reg_land','transition'),'Mha'),
            ('transition_negative_correction',transition_clip,('interval','reg_land','transition'),'Mha'),
            ('harvest_area',harvest,('interval','reg_land','harvest_source'),'Mha'),
            ('harvest_carbon',bioh,('interval','reg_land','harvest_source'),'Pg C'),
            ('harvest_carbon_raw',bioh_raw,('interval','reg_land','harvest_source'),'Pg C'),
            ('harvest_carbon_negative_correction',bioh_correction,('interval','reg_land','harvest_source'),'Pg C'),
            ('harvest_carbon_without_positive_area',carbon_without_area,('interval','reg_land','harvest_source'),'Pg C'),
            ('state_initial_adjustment',state[0]-state_raw[0],('reg_land','luh_class'),'Mha'),
            ('raw_state_negative_count',state_negative_count,('state_year','luh_class'),'1'),
            ('raw_transition_negative_count',transition_negative_count,('interval','transition'),'1'),
            ('raw_bioh_negative_count',carbon_negative_count,('interval','harvest_source'),'1'),
            ('raw_state_min_fraction',state_min_raw,('state_year','luh_class'),'1'),
            ('raw_transition_min_fraction',transition_min_raw,('interval','transition'),'1'),
            ('raw_bioh_min_kgC',carbon_min_raw,('interval','harvest_source'),'kg C'),
            ('raw_partition_max_abs_fraction',partition_max,('state_year',),'1')]
        for name,value,dims,unit in variables:ds[name]=xr.DataArray(value,dims=dims,attrs={'units':unit})
        ds.attrs.update(source_doi='10.5281/zenodo.19261724',source_id='UofMD-landState-3-1-2',
            state_policy='850 negative fractions clipped, positive fractions normalized to static non-ice-water land; later regional states accumulated from measured area events. Published states retained independently.',
            event_policy='Negative original conversions clipped to zero with a separate additive correction variable. No residual or fabricated harvest/age-reset events.',
            carbon_policy='Negative bioh interpreted as zero following the official author guidance; raw carbon and its correction retained. Diagnostic only, not a model carbon demand.',
            region_weights=spec['region_weights'],region_assumption='Prepared conditional R10 1-degree weights shared by 0.25-degree subcells; no second land fraction.',
            source_time_semantics='Year labels; upstream time: mean does not establish instantaneous state sampling.',
            class_policy='All 13 source classes retained; five-biome rangeland mapping remains a separate semantic decision.',
            data_status=report['status'],requires_five_biome_mapping=1)
        ds.to_netcdf(out/'land_use_absolute.nc',encoding={k:{'zlib':True,'complevel':2} for k in ds.data_vars if ds[k].dtype.kind in 'fiu'})
        report['output']='land_use_absolute.nc'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config',type=Path,default=Path('manifests/land_use_preparation.json'))
    args=parser.parse_args();root=args.root.resolve();spec=json.loads((root/args.config).read_text())
    out=root/spec['output_directory'];out.mkdir(parents=True,exist_ok=False)
    report={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':spec,'status':'running',
        'raw_rehashed':False,'model_integrated':False,'source_files_modified':False}
    try:prepare(root,spec,out,report)
    except BaseException as exc:
        report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc).replace(str(root),'<project>')});raise
    finally:
        report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save(out/'report.json',report)
        print('Report:',out.relative_to(root)/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
