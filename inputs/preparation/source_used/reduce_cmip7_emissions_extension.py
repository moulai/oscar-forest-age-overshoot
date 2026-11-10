"""Conservatively reduce CMIP7 gridded emissions to native IAMC_R10 regions.

Surface and open burning use production 1.1.1; aviation uses 1.1.2.
All source monthly intervals are integrated before annual interpolation.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/future/cmip7_background_extension_v1'
OUT=ROOT/'data/processed/future_background/extension_v1'
REGIONS=('ROWO','AFRICA','CHINA+','EUROPE','INDIA+','LATIN_AM','MIDDLE_EAST','NORTH_AM','PAC_OECD','REF_ECON','REST_ASIA')

def region_weights():
    """Native fractional polygons plus explicitly unassigned ocean in ROWO."""
    mask=xr.load_dataset(ROOT/'oscar/_core/_regions/OSCAR_region_mask_1deg.nc')
    mapping=pd.read_csv(ROOT/'oscar/_core/_regions/OSCAR_reg_dict.csv').set_index('reg_code')['IAMC_R10'].to_dict()
    frac=mask.frac_reg.fillna(0).values.astype(float)
    w=np.stack([frac[:,:,np.array([mapping[c]==r for c in mask.reg_code.values])].sum(axis=2) for r in REGIONS],axis=2)
    total=w.sum(axis=2)
    assert total.max() < 1+1e-6 and total.min() >= 0
    land=np.divide(w,total[:,:,None],out=np.zeros_like(w),where=total[:,:,None]>0)
    ocean_only=total==0
    land[ocean_only,0]=1.
    # The raw fractional map applies to spatially distributed shipping/aviation.
    w/=np.maximum(1.,total)[:,:,None]
    w[:,:,0]+=1.-w.sum(axis=2)
    def fine(values):
        return np.repeat(np.repeat(values[::-1],2,axis=0),2,axis=1)
    raw=fine(w).reshape(-1,len(REGIONS))
    conditional=fine(land).reshape(-1,len(REGIONS))
    assert np.max(np.abs(raw.sum(axis=1)-1))<1e-12
    assert np.max(np.abs(conditional.sum(axis=1)-1))<1e-12
    return raw,conditional,fine(ocean_only)


def validate_reduction(row):
    """Validate an existing small cache against its declared raw-source row."""
    target=OUT/row['scenario']/'emission_reductions'/f"{row['variable']}.nc"
    report=json.loads(target.with_suffix('.json').read_text())
    assert report['status']=='passed' and report['source']==row, 'Reduction source report differs'
    data=xr.load_dataset(target)
    assert data.attrs['source_id']==row['source_id'], 'Reduction scenario/version differs'
    assert data.attrs['source_variable']==row['variable'], 'Reduction variable differs'
    assert data.attrs['source_path']==str((RAW/row['path']).relative_to(ROOT)), 'Reduction raw path differs'
    assert data.attrs['cell_area']=='WGS84 ellipsoid exact latitude-band integration'
    assert data.attrs['spatial_allocation_version']=='sector-support-v1'
    expected_years=np.array([2022,2023,2024,*range(2025,2101,5)])
    assert np.array_equal(data.year,expected_years) and report['source_years']==expected_years.tolist()
    assert np.array_equal(data.reg_land,np.array(REGIONS)), 'Reduction region identity/order differs'
    sector_count=(25 if '_AIR_' in row['variable'] else 4 if '_openburning' in row['variable']
                  else 10 if row['variable']=='CO2_em_anthro' else 8)
    assert np.array_equal(data.source_sector,np.arange(sector_count))
    assert data.emission.dims==('year','reg_land') and data.global_emission.dims==('year',)
    assert data.all_sector_global_emission.dims==('year','source_sector')
    for name in ('emission','global_emission','all_sector_global_emission'):
        assert data[name].attrs['units']=='kg yr-1' and np.isfinite(data[name]).all()
    assert report['region_conservation_max_relative']<1e-12
    sector_global=data.all_sector_global_emission
    selected_global=(sector_global.sel(source_sector=[0,3]).sum('source_sector')
                     if '_openburning' in row['variable'] else sector_global.sum('source_sector'))
    assert np.allclose(data.global_emission,selected_global,rtol=1e-12,atol=1e-6), 'Selected sector mass differs from the model input'
    return data


def reduce_one(row):
    target=OUT/row['scenario']/'emission_reductions'/f"{row['variable']}.nc"
    if target.exists():
        validate_reduction(row)
        return str(target)
    path=RAW/row['path']
    if not path.with_suffix('.receipt.json').exists():return None
    started=time.monotonic();raw_weights,land_weights,ocean_only=region_weights()
    with netCDF4.Dataset(path) as source:
        source.set_auto_mask(False)
        assert source.source_id==row['source_id'] and source.mip_era=='CMIP7'
        variable=source[row['variable']]
        assert variable.units=='kg s-1 m-2' and variable.cell_methods=='time: mean'
        assert np.array_equal(source['lat'][:],np.arange(-89.75,90,.5))
        assert np.array_equal(source['lon'][:],np.arange(-179.75,180,.5))
        t=source['time']; dates=netCDF4.num2date(t[:],t.units,t.calendar)
        years=np.array([d.year for d in dates]); months=np.array([d.month for d in dates])
        unique=np.unique(years)
        assert np.array_equal(unique,np.array([2022,2023,2024,*range(2025,2101,5)]))
        assert np.array_equal(months,np.tile(np.arange(1,13),len(unique))) and t.calendar=='365_day'
        duration=np.diff(source['time_bnds'][:],axis=1).ravel()*86400.
        assert np.all(duration>0)
        latb=source['lat_bnds'][:];lonb=source['lon_bnds'][:]
        # Exact latitude-band integration on the WGS84 ellipsoid. The public
        # annual-by-sector diagnostic totals independently constrain this area
        # definition (sphere areas differ by up to 0.4% across latitude bands).
        equatorial_radius=6378137.; flattening=1./298.257223563
        eccentricity=np.sqrt(flattening*(2.-flattening))
        sine=np.sin(np.deg2rad(latb))
        primitive=(sine/(1.-eccentricity**2*sine**2)
                   +np.arctanh(eccentricity*sine)/eccentricity)
        area=(equatorial_radius**2*(1.-eccentricity**2)/2.
              *np.diff(primitive,axis=1)*np.deg2rad(lonb[:,1]-lonb[:,0])[None,:])
        values=variable[:]
        assert np.isfinite(values).all(), 'Missing raw emission values are not filled'
        for missing_attr in ('_FillValue','missing_value'):
            if missing_attr in variable.ncattrs():
                for sentinel in np.atleast_1d(variable.getncattr(missing_attr)):
                    if np.isfinite(sentinel):
                        assert not np.any(values==sentinel), 'Finite raw missing-value sentinel found'
        negative=float(np.minimum(values,0).sum())
        source_is_land='_AIR_' not in row['variable']
        if '_openburning' in row['variable']:
            assert variable.dimensions==('time','sector','lat','lon')
            assert source['sector'].ids=='0: Agricultural Waste Burning; 1: Forest Burning; 2: Grassland Burning; 3: Peat Burning'
            land_selected=values[:,0]+values[:,3]
            spatial_selected=np.zeros_like(land_selected)
            selection='Agricultural waste and peat burning only; forest and grassland burning are endogenous'
        elif '_AIR_' in row['variable']:
            spatial_selected=values.sum(axis=1,dtype=float)
            land_selected=np.zeros_like(spatial_selected)
            selection='All 25 aviation levels (layer fluxes; no thickness multiplier)'
        else:
            assert variable.dimensions==('time','sector','lat','lon')
            assert '7: International Shipping' in source['sector'].ids
            land_sectors=[sector for sector in range(values.shape[1]) if sector!=7]
            if row['variable']=='CO2_em_anthro':
                assert values.shape[1]==10 and '8: BECCS; 9: Other Capture and Removal' in source['sector'].ids
            else:assert values.shape[1]==8
            land_selected=values[:,land_sectors].sum(axis=1,dtype=float)
            spatial_selected=values[:,7].copy()
            selection='All non-shipping surface sectors use conditional land support, including signed CO2 BECCS/removal; shipping uses full-cell support'
        sector_global=np.einsum('tsij,ij,t->ts',values,area,duration,optimize=True)
        del values
        land_mass=land_selected*area[None,:,:]
        spatial_mass=spatial_selected*area[None,:,:]
        monthly=(land_mass.reshape(len(years),-1)@land_weights
                 +spatial_mass.reshape(len(years),-1)@raw_weights)*duration[:,None]
        global_selected=(land_mass+spatial_mass).sum(axis=(1,2))*duration
        ocean_land=land_mass[:,ocean_only].sum(axis=1)*duration
        del land_selected,spatial_selected,land_mass,spatial_mass
        annual=np.stack([monthly[years==y].sum(axis=0) for y in unique])
        annual_global=np.array([global_selected[years==y].sum() for y in unique])
        sector_annual=np.stack([sector_global[years==y].sum(axis=0) for y in unique])
        ocean_land_annual=np.array([ocean_land[years==y].sum() for y in unique])
        absolute_error=float(np.abs(annual.sum(axis=1)-annual_global).max())
        relative_error=float(np.max(np.abs(annual.sum(axis=1)-annual_global)/np.maximum(1.,np.abs(annual_global))))
        assert relative_error<1e-12
    ds=xr.Dataset({'emission':(('year','reg_land'),annual,{'units':'kg yr-1'}),
                   'global_emission':(('year',),annual_global,{'units':'kg yr-1'}),
                   'all_sector_global_emission':(('year','source_sector'),sector_annual,{'units':'kg yr-1'}),
                   'land_source_ocean_only_emission':(('year',),ocean_land_annual,{'units':'kg yr-1','destination':'ROWO; retained without clipping'})},
                  coords={'year':unique,'reg_land':list(REGIONS),'source_sector':np.arange(sector_annual.shape[1])},
                  attrs={'cell_area':'WGS84 ellipsoid exact latitude-band integration','source_id':row['source_id'],'source_variable':row['variable'],'source_path':str(path.relative_to(ROOT)),
                         'sector_selection':selection,'spatial_allocation_version':'sector-support-v1',
                         'spatial_reduction':'Land sectors use conditional land fractions; shipping/aviation use full-cell fractions; pure ocean land-source mass retained in ROWO',
                         'temporal_reduction':'Integrate monthly mean flux by exact 365-day-calendar bounds before linear interpolation between supplied years'})
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.partial.nc');ds.to_netcdf(temp);temp.replace(target)
    report={'status':'passed','source':row,'region_conservation_max_kg_yr':absolute_error,'region_conservation_max_relative':relative_error,
            'source_negative_flux_sum_kg_s_m2':negative,'elapsed_seconds':time.monotonic()-started,
            'source_years':unique.tolist(),'sector_selection':selection,
            'land_source_on_ocean_only_cells_kg_yr':ocean_land_annual.tolist(),
            'spatial_allocation_version':'sector-support-v1'}
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'reduced':row['scenario']+'/'+row['variable'],'seconds':report['elapsed_seconds']}),flush=True)
    return str(target)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'manifests/cmip7_background_extension_sources.json').read_text())
    rows=[r for r in manifest['files'] if r['scenario'] in ('hl','ln')]
    for row in rows:
        if (OUT/row['scenario']/'emission_reductions'/f"{row['variable']}.nc").exists():
            validate_reduction(row)
    while True:
        pending=[r for r in rows if not (OUT/r['scenario']/'emission_reductions'/f"{r['variable']}.nc").exists()]
        ready=[r for r in pending if (RAW/r['path']).with_suffix('.receipt.json').exists()]
        with ProcessPoolExecutor(max_workers=6) as pool:list(pool.map(reduce_one,ready))
        if not args.watch or not pending:break
        time.sleep(30)
