#!/usr/bin/env python3
"""Prepare an independent secondary gross-AG carbon request extension.

Reuses the existing regional LUH3 reduction and original ordered parameters.
No raw raster, new parameter draw, model integration or parent bundle edit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import xarray as xr

LABELS = ['secmf_harv', 'secyf_harv']
SOURCES = ['secmf', 'secyf']
TIME = 'Initial/restart anchor is zero; later label y represents (y-1,y], event at y-0.5; raw interval i maps to y=i+1.'
PROJECTION = 'native_effective_aboveground'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinates_equal(a, b, names):
    for n in names:
        if not np.array_equal(a[n], b[n]):
            raise ValueError('Coordinate/order mismatch: '+n)


def projection_parameters(primary):
    """Derive the working projection without changing any original member."""
    p = primary[['p_wood','p_root','p_leaf']].sel(bio_land='Forest', drop=True).astype(np.float64)
    k = (1-p.p_wood-p.p_root)/(1-p.p_wood)
    if not bool(np.isfinite(k).all()) or bool(((k<0)|(k>1)).any()):
        raise ValueError('Invalid effective aboveground projection')
    out = xr.Dataset({'k_F_AG':k.transpose('reg_land','config'),
                      'k_F_AG_leaf_only':(p.p_leaf/(1-p.p_wood)).transpose('reg_land','config')})
    out.k_F_AG_leaf_only.attrs = {'units':'1','projection':'native_leaf_only_boundary',
        'formula':'p_leaf/(1-p_wood)','carbon_operator':'G_leaf = W + k_F_AG_leaf_only * F',
        'status':'Passive boundary diagnostic; not the default request projection',
        'interpretation':'Excludes the operational soft contribution; does not certify actual organ boundaries'}
    out.k_F_AG.attrs = {'units':'1','projection':PROJECTION,'formula':'(1-p_wood-p_root)/(1-p_wood)',
        'carbon_operator':'G_eff = W + k_F_AG * F',
        'status':'Operational working projection, not independently observed organ AG carbon',
        'assumptions':'W treated as effective aboveground wood; F root versus leaf+soft fractions fixed across age and environment'}
    out.attrs = {'schema':'regional-secondary-harvest-carbon-projection-v1','projection':PROJECTION,
        'parameter_policy':'Original joint p_wood and p_root by original member; no sampling, fitting, filtering or reweighting'}
    return out


def phase_requests(source, parent_forcing):
    """Align the right-end annual interval and zero only the phase anchor."""
    years = parent_forcing.year.values
    if not np.array_equal(years,np.arange(years[0],years[-1]+1)):
        raise ValueError('Nonannual parent forcing')
    value = source.harvest_carbon.sel(interval=years[1:]-1).rename(interval='year',harvest_source='secondary_harvest_label')
    value = value.assign_coords(year=years[1:],secondary_harvest_label=LABELS)
    anchor = xr.zeros_like(value.isel(year=0,drop=True)).expand_dims(year=[years[0]])
    value = xr.concat([anchor,value],dim='year').transpose('year','reg_land','secondary_harvest_label')
    out = xr.Dataset({'Q_sec_harvest_by_label':value,'Q_sec_harvest':value.sum('secondary_harvest_label',skipna=False)})
    coordinates_equal(out,parent_forcing,['year','reg_land'])
    for n in out:
        out[n].attrs = {'units':'PgC','time_support':'Amount per complete annual interval; anchor has no event',
            'scope':'Secondary forest repeated harvest gross aboveground biomass carbon request',
            'conversion':'Already regional Pg C; no grid-area multiplier, slash division, product conversion or subtraction of clearing contribution'}
    out.attrs = {'schema':'regional-secondary-harvest-carbon-forcing-v1','time_contract':TIME,
        'contract':'secondary_gross_agb_request_proportional','source_semantics':'secmf and secyf are upstream allocation labels, not observed donor age strata',
        'event_scope':'Only secdf repeated harvest; primary and all other activities retain the parent area contract',
        'zero_anchor_reason':'First label is an initial/restart state; the preceding interval belongs to the preceding phase'}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    parent=a.project/'data/processed/regional_forest_age_inputs_v1'
    if a.output.resolve().is_relative_to(parent.resolve()):
        raise ValueError('Extension output must be outside the immutable parent bundle')
    original={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in parent.iterdir() if p.is_file()}
    manifest=json.loads((parent/'manifest.json').read_text())
    source_path=a.project/'data/processed/land_use/luh3_r10_v1/land_use_absolute.nc'
    report_path=source_path.with_name('report.json')
    source_report=json.loads(report_path.read_text())
    vars_=['interval_start','interval_end','harvest_area','harvest_carbon','harvest_carbon_raw',
           'harvest_carbon_negative_correction','harvest_carbon_without_positive_area','raw_bioh_negative_count','raw_bioh_min_kgC']
    with xr.open_dataset(source_path) as ds:
        source=ds[vars_].sel(interval=slice(850,2021),harvest_source=SOURCES).load()
    if source.attrs['source_id']!='UofMD-landState-3-1-2':
        raise ValueError('Unexpected LUH product; explicit source review required')
    if not np.array_equal(source.interval,np.arange(850,2022)) or not np.array_equal(source.interval_end,source.interval+1):
        raise ValueError('Missing source intervals')
    missing={n:int((~np.isfinite(v)).sum()) for n,v in source.data_vars.items()}
    if any(missing.values()):
        raise ValueError('Missing processed source values: '+str(missing))
    if source.harvest_carbon.attrs.get('units')!='Pg C' or source.harvest_area.attrs.get('units')!='Mha':
        raise ValueError('Unexpected source units')
    if bool((source.harvest_carbon<0).any()) or bool((source.harvest_area<0).any()):
        raise ValueError('Negative corrected carbon or area')
    closure=abs(source.harvest_carbon-source.harvest_carbon_raw-source.harvest_carbon_negative_correction)
    if float(closure.max())>1e-14:
        raise ValueError('Existing signed-carbon correction does not close')
    primary=xr.load_dataset(parent/'primary_parameters.nc')
    if primary.sizes['config']!=500 or primary.sizes['reg_land']!=11:
        raise ValueError('Full original ensemble/region set required')
    if primary.source_config.values.tolist()!=manifest['source_config']:
        raise ValueError('Original member identity/order changed')
    coordinates_equal(primary,source,['reg_land'])
    projection=projection_parameters(primary)
    a.output.mkdir(parents=True,exist_ok=True)
    outputs={}
    def write(name,ds):
        path=a.output/name
        ds.to_netcdf(path)
        outputs[name]={'path':str(path.resolve()),'size_bytes':path.stat().st_size,'sha256':digest(path)}
    write('projection_parameters.nc',projection)
    phases={}
    for phase in ['prehistory','historical']:
        with xr.open_dataset(parent/(phase+'_forcing.nc')) as f:
            out=phase_requests(source,f)
            # Read just the two parent event series as an area provenance diagnostic.
            area=f.event_area.sel(event=LABELS).load()
        write(phase+'_requests.nc',out)
        years=out.year.values
        rawarea=source.harvest_area.sel(interval=years[1:]-1).values
        actual=area.isel(year=slice(1,None)).transpose('year','reg_land','event').values
        phases[phase]={'first_year':int(years[0]),'last_year':int(years[-1]),'anchor_sum_PgC':float(out.Q_sec_harvest.isel(year=0).sum()),
             'request_sum_PgC':float(out.Q_sec_harvest.sum()),'parent_event_area_minus_raw_max_abs_Mha':float(np.max(np.abs(actual-rawarea)))}
    source['secondary_gross_request']=source.harvest_carbon.sum('harvest_source',skipna=False)
    source['secondary_area']=source.harvest_area.sum('harvest_source',skipna=False)
    source.secondary_gross_request.attrs={'units':'Pg C','formula':'secmf_bioh + secyf_bioh after inherited negative cleanup'}
    source.secondary_area.attrs={'units':'Mha','status':'Upstream area comparator; not a candidate area prescription'}
    source.attrs['extension_role']='Source audit only; original area and signed/corrected carbon retained separately'
    write('source_history.nc',source)
    after={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in parent.iterdir() if p.is_file()}
    if after!=original:
        raise RuntimeError('Parent bundle changed during preparation')
    receipt={'schema':'regional-secondary-harvest-carbon-v1','status':'prepared_inputs_verified','model_run':False,
        'parent_input_directory':str(parent),'parent_input_manifest_sha256':digest(parent/'manifest.json'),
        'parent_files_unchanged':True,'parent_file_count':len(original),
        'ordered_source_ids_sha256':manifest['ordered_source_ids_sha256'],'source_config':manifest['source_config'],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        'preparer_sha256':digest(__file__),'source':{'processed_file':str(source_path),'processed_report':str(report_path),
        'source_id':source.attrs['source_id'],'doi':source.attrs['source_doi'],'acquisition_run':source_report['acquisition_run'],
        'region_weights':source.attrs['region_weights'],'raw_rasters_read':False,'raw_rehashed':False,'new_negative_cleanup':False,
        'negative_cleanup_policy':source_report['policy']['harvest_carbon'],'official_guidance':source_report['policy']['negative_bioh_author_guidance']},
        'contract':'secondary_gross_agb_request_proportional','projection':PROJECTION,'time_contract':TIME,
        'secondary_harvest_label':LABELS,'phases':phases,'missing_count':missing,
        'correction_sum_PgC':float(source.harvest_carbon_negative_correction.sum()),
        'correction_sum_by_label_PgC':{s:float(source.harvest_carbon_negative_correction.sel(harvest_source=s).sum()) for s in SOURCES},
        'raw_negative_cell_occurrences':int(source.raw_bioh_negative_count.sum()),
        'raw_minimum_kgC':float(source.raw_bioh_min_kgC.min()),
        'carbon_without_positive_area_PgC':float(source.harvest_carbon_without_positive_area.sum()),
        'correction_closure_max_abs_PgC':float(closure.max()),
        'k_F_AG_min':float(projection.k_F_AG.min()),'k_F_AG_max':float(projection.k_F_AG.max()),
        'outputs':outputs,'scope':'History/prehistory only; no future extension; the operational projection is not observed aboveground-organ carbon'}
    (a.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_config','outputs']},indent=2))

if __name__=='__main__':
    main()
