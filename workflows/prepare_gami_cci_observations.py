"""Reduce completed GAMI/CCI files on their measured common grid; no model projection."""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import h5py
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2010, 2020)
SUPPORTS = ('gami_native', 'gami_common', 'agb_native', 'agb_common')
BIOMASS_SUPPORTS = ('native', 'common')
LABELS = ('0-20','20-40','40-60','60-80','80-100','100-120','120-140',
          '140-160','160-180','180-200','200-299','>299')
METRICS = ('gami_count','gami_area_ha','common_gami_count','common_gami_area_ha',
           'agb_area_ha','agb_dry_mass_Mg','common_agb_area_ha','common_agb_dry_mass_Mg',
           'outside_cci_area_ha','missing_agb_area_ha','invalid_agb_area_ha',
           'sd_area_ha','sd_times_area_Mg','missing_sd_area_ha','invalid_sd_area_ha',
           'common_sd_area_ha','common_sd_times_area_Mg')
POLICY = 'v1_agb_valid_independent_of_sd_299_unresolved'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.tmp-', mode='w', delete=False) as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
        temporary = Path(f.name)
    if exclusive:
        os.link(temporary, path); temporary.unlink()
    else:
        os.replace(temporary, path)


def text_attr(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def completed_input(root, destination, source, status_key):
    """Check the receipt before any science container is opened. Do not rehash raw."""
    path = root / destination / source['name']
    receipt = Path(str(path) + '.receipt.json')
    require(receipt.is_file(), f'Completion receipt required before opening {path}')
    r = json.loads(receipt.read_text())
    require(r.get(status_key) == 'complete' and r.get('source') == source,
            f'Incomplete or wrong-source receipt: {receipt}')
    require(r.get('bytes') == source['size_bytes'] and path.is_file()
            and path.stat().st_size == source['size_bytes'], f'Raw size mismatch: {path}')
    require(re.fullmatch('[0-9a-f]{64}', r.get('sha256', '')) is not None,
            f'Missing acquisition digest: {receipt}')
    s = path.stat()
    return path, {'path': str(path.resolve()), 'receipt_sha256': digest(receipt),
                  'acquisition_sha256': r['sha256'], 'size': s.st_size,
                  'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}



def validate_completed_output(out, identity):
    """A completion record is usable only with its intact derived artifact."""
    done=json.loads((out/'completion.json').read_text())
    require(done.get('identity')==identity,'Completion identity mismatch')
    output=done.get('output',{})
    require(output.get('path')=='observations.nc','Unexpected completed output path')
    path=out/'observations.nc'
    require(path.is_file(),f'Completed derived output is missing: {path}')
    require(path.stat().st_size==output.get('bytes'),f'Completed derived output size mismatch: {path}')
    checksum=output.get('sha256')
    require(isinstance(checksum,str) and re.fullmatch('[0-9a-f]{64}',checksum) is not None,
            'Completed derived output lacks a valid recorded SHA256')
    require(digest(path)==checksum,f'Completed derived output SHA256 mismatch: {path}')
    return done


def validate_alignment(glat, glon, clat, clon, ppd=1125, tolerance=1e-8):
    require(len(glat) == 180*ppd and len(glon) == 360*ppd, 'Wrong GAMI grid length')
    require(len(clat) == 140*ppd and len(clon) == 360*ppd, 'Wrong CCI grid length')
    for actual, nominal in ((glat, 90-(np.arange(180*ppd)+.5)/ppd),
                            (glon, -180+(np.arange(360*ppd)+.5)/ppd),
                            (clat, 80-(np.arange(140*ppd)+.5)/ppd),
                            (clon, -180+(np.arange(360*ppd)+.5)/ppd)):
        require(np.all(np.isfinite(actual)) and np.max(np.abs(actual-nominal)) < tolerance,
                'Non-nominal grid, reversed order, or shifted spacing')
    lat_delta = float(np.max(np.abs(glat[10*ppd:150*ppd]-clat)))
    lon_delta = float(np.max(np.abs(glon-clon)))
    require(max(lat_delta, lon_delta) < tolerance, 'GAMI/CCI coordinates disagree')
    return {'max_lat_delta_degrees': lat_delta, 'max_lon_delta_degrees': lon_delta,
            'tolerance_degrees': tolerance, 'gami_lat_slice': [10*ppd,150*ppd]}


def row_areas_ha(rows, ppd=1125):
    """WGS84 ellipsoidal rectangle areas, not geodesic-corner polygons."""
    a, f = 6378137., 1/298.257223563
    e2 = f*(2-f); e = np.sqrt(e2)
    step=np.deg2rad(1/ppd)
    north=np.deg2rad(90-np.asarray(rows)/ppd);south=north-step
    u,v=np.sin(north),np.sin(south)
    # Analytic difference avoids cancellation of two ~4e13 primitives near poles.
    du=2*np.cos((north+south)/2)*np.sin(step/2)
    integral=a*a*(1-e2)/2*(du*(1+e2*u*v)/((1-e2*u*u)*(1-e2*v*v))
                           +np.arctanh(e*du/(1-e2*u*v))/e)
    return integral*step/10000


def bin_histogram(hist):
    """Exclude unresolved 299 from both old bins; raw histogram remains complete."""
    result = [hist[..., :21].sum(axis=-1)]
    result += [hist[..., lo+1:lo+21].sum(axis=-1) for lo in range(20,200,20)]
    result += [hist[..., 201:299].sum(axis=-1), hist[..., 300]]
    return np.stack(result, axis=-1)


def reduce_patch(ages, agb, sd, area, covered=True):
    """Independent two-year kernel; all arrays are small geographic intersections."""
    require(ages.shape == agb.shape == sd.shape and ages.shape[0] == 2, 'Patch shape mismatch')
    area = np.broadcast_to(area, ages.shape[1:])
    valid = (ages >= 0) & (ages <= 300)
    av = (agb >= 0) & (agb <= 10000) & covered
    sv = (sd >= 0) & (sd <= 10000) & covered
    common_age = valid[0] & valid[1]
    native = valid & av
    common = common_age & av[0] & av[1]
    common_sd = common & sv[0] & sv[1]
    counts = np.zeros((2,4,301), dtype=np.int64)
    areas = np.zeros((2,4,301))
    mass = np.zeros((2,2,301))
    stats = np.zeros((2,len(METRICS)))
    sd_bounds = np.full((2,2,2), np.nan)  # year, native/common, min/max
    diagnostics = []
    for y in range(2):
        supports = (valid[y],common_age,native[y],common)
        for k, mask in enumerate(supports):
            counts[y,k] = np.bincount(ages[y][mask], minlength=301)
            areas[y,k] = np.bincount(ages[y][mask], weights=area[mask], minlength=301)
        for k, mask in enumerate((native[y],common)):
            mass[y,k] = np.bincount(ages[y][mask], weights=area[mask]*agb[y][mask], minlength=301)
        ns = native[y] & sv[y]
        weights = lambda mask: float(area[mask].sum())
        stats[y] = [valid[y].sum(), weights(valid[y]), common_age.sum(), weights(common_age),
                    weights(native[y]), mass[y,0].sum(), weights(common), mass[y,1].sum(),
                    weights(valid[y]) if not covered else 0.,
                    weights(valid[y] & (agb[y] == -32768)) if covered else 0.,
                    weights(valid[y] & ~av[y] & (agb[y] != -32768)) if covered else 0.,
                    weights(ns), float((sd[y][ns]*area[ns]).sum()),
                    weights(native[y] & (sd[y] == -32768)),
                    weights(native[y] & ~sv[y] & (sd[y] != -32768)),
                    weights(common_sd), float((sd[y][common_sd]*area[common_sd]).sum())]
        for k, mask in enumerate((ns,common_sd)):
            if mask.any(): sd_bounds[y,k] = [sd[y][mask].min(),sd[y][mask].max()]
        unknown = ages[y][~valid[y] & (ages[y] != -9999) & (ages[y] != -32767)]
        codes, n = np.unique(unknown, return_counts=True)
        diagnostics.append({'missing_minus9999':int((ages[y] == -9999).sum()),
                            'default_fill_minus32767':int((ages[y] == -32767).sum()),
                            'unexpected_age_codes':{str(int(c)):int(v) for c,v in zip(codes,n)}})
    return counts, areas, mass, stats, sd_bounds, diagnostics


class ChunkCache:
    """Lazy bounded int16 CCI cache shared across all members of a GAMI block."""
    def __init__(self, sources, max_bytes=6_100_000_000):
        self.sources, self.max_bytes = sources, max_bytes
        self.cache = OrderedDict(); self.bytes = self.peak_bytes = self.reads = self.fill_hits = 0

    def window(self, year, name, y0, y1, x0, x1):
        ds = self.sources[year][name]
        sy, sx = ds.chunks[-2:]
        by, bx = y0//sy*sy, x0//sx*sx
        require(y1 <= by+sy and x1 <= bx+sx, 'Geographic patch crosses CCI chunk boundary')
        key = (year,name,by,bx)
        if key not in self.cache:
            # Only source allocation/fill metadata or an actual read proves a fill block.
            info = ds.id.get_chunk_info_by_coord((0,by,bx))
            if info.size == 0:
                self.cache[key] = np.int16(ds.fillvalue); self.fill_hits += 1
            else:
                size = min(sy,ds.shape[1]-by)*min(sx,ds.shape[2]-bx)*ds.dtype.itemsize
                require(size <= self.max_bytes, 'Cache too small for one native CCI chunk')
                while self.bytes+size > self.max_bytes:
                    _, old = self.cache.popitem(last=False)
                    if isinstance(old,np.ndarray): self.bytes -= old.nbytes
                data = ds[0,by:by+sy,bx:bx+sx]
                self.reads += 1
                if np.all(data == ds.fillvalue):
                    self.cache[key] = np.int16(ds.fillvalue)
                else:
                    self.cache[key] = data; self.bytes += data.nbytes
                    self.peak_bytes = max(self.peak_bytes,self.bytes)
        value = self.cache.pop(key); self.cache[key] = value
        if np.isscalar(value): return np.full((y1-y0,x1-x0),value,dtype=np.int16)
        return value[y0-by:y1-by,x0-bx:x1-bx]


def geographic_patches(y0,y1,x0,x1,ppd):
    for r in range(y0//ppd,(y1-1)//ppd+1):
        for c in range(x0//ppd,(x1-1)//ppd+1):
            yield r,c,max(y0,r*ppd),min(y1,(r+1)*ppd),max(x0,c*ppd),min(x1,(c+1)*ppd)


def validate_weights(weights):
    require(np.all(np.isfinite(weights) | np.isnan(weights)), 'Infinite regional weight')
    w = np.nan_to_num(weights, nan=0.)
    require(np.all(w >= 0) and np.all(w.sum(axis=-1) <= 1+1e-12), 'Invalid geographic allocation')
    residual = 1-w.sum(axis=-1)
    correction = float(max(0.,-residual.min()))
    return np.concatenate([w,np.maximum(residual,0)[...,None]],axis=-1), correction


def reduce_block(age_ds, cci, y0, x0, weights, ppd=1125, cci_row_start=11250, cci_row_end=168750, progress=None):
    """Native spatial block, then member, then small 1-degree intersections."""
    nm,ny,nx,_ = age_ds.shape
    sy,sx = age_ds.chunks[1:3]
    y1,x1 = min(y0+sy,ny),min(x0+sx,nx)
    patches = list(geographic_patches(y0,y1,x0,x1,ppd))
    nr = weights.shape[-1]
    result = {'hist_count':np.zeros((2,nm,4,nr,301)),
              'hist_area_ha':np.zeros((2,nm,4,nr,301)),
              'dry_mass_Mg':np.zeros((2,nm,2,nr,301)),
              'stats':np.zeros((2,nm,nr,len(METRICS))),
              'sd_bounds':np.full((2,nm,2,nr,2),np.nan),
              'geo':np.zeros((2,nm,len(patches),len(METRICS))),
              'cells':np.array([(r,c) for r,c,*_ in patches],dtype=np.int16)}
    diag = [[{'missing_minus9999':0,'default_fill_minus32767':0,'unexpected_age_codes':{}}
             for _ in range(nm)] for _ in range(2)]
    reads = 0
    block_areas=row_areas_ha(np.arange(y0,y1),ppd)
    for m in range(nm):
        ages=[]
        for y in range(2):
            info=age_ds.id.get_chunk_info_by_coord((m,y0,x0,y))
            if info.size == 0:
                a=np.broadcast_to(np.int16(age_ds.fillvalue),(y1-y0,x1-x0))
            else:
                a=age_ds[m,y0:y1,x0:x1,y]; reads+=1
            ages.append(a)
        # Actual read/fill metadata, never a published age-bin shortcut.
        constants=[(int(a.min()),int(a.max())) for a in ages]
        if all(lo == hi and lo in (-9999,-32767) for lo,hi in constants):
            for y,(lo,_) in enumerate(constants):
                key='missing_minus9999' if lo == -9999 else 'default_fill_minus32767'
                diag[y][m][key]=(y1-y0)*(x1-x0)
            if progress: progress(m+1)
            continue
        for cell,(r,c,top,bottom,left,right) in enumerate(patches):
            small_age=np.stack([a[top-y0:bottom-y0,left-x0:right-x0] for a in ages])
            covered=top >= cci_row_start and bottom <= cci_row_end
            require(covered or bottom <= cci_row_start or top >= cci_row_end, 'CCI boundary splits geographic patch')
            shape=(2,bottom-top,right-left)
            if covered and np.any((small_age >= 0)&(small_age <= 300)):
                agb=np.stack([cci.window(y,'agb',top-cci_row_start,bottom-cci_row_start,left,right) for y in range(2)])
                sd=np.stack([cci.window(y,'agb_sd',top-cci_row_start,bottom-cci_row_start,left,right) for y in range(2)])
            else:
                agb=np.full(shape,-32768,dtype=np.int16); sd=agb
            values=reduce_patch(small_age,agb,sd,block_areas[top-y0:bottom-y0,None],covered)
            count,area,mass,stats,bounds,ds=values
            w=weights[r,c]
            result['hist_count'][:,m] += count[:,:,None,:]*w[None,None,:,None]
            result['hist_area_ha'][:,m] += area[:,:,None,:]*w[None,None,:,None]
            result['dry_mass_Mg'][:,m] += mass[:,:,None,:]*w[None,None,:,None]
            result['stats'][:,m] += stats[:,None,:]*w[None,:,None]
            result['geo'][:,m,cell]=stats
            b=np.where(w[None,None,:,None]>0,bounds[:,:,None,:],np.nan)
            old=result['sd_bounds'][:,m]
            old[...,0]=np.fmin(old[...,0],b[...,0]); old[...,1]=np.fmax(old[...,1],b[...,1])
            for y in range(2):
                for key in ('missing_minus9999','default_fill_minus32767'):
                    diag[y][m][key]+=ds[y][key]
                for code,n in ds[y]['unexpected_age_codes'].items():
                    target=diag[y][m]['unexpected_age_codes'];target[code]=target.get(code,0)+n
        if progress: progress(m+1)
        # No annual carry or model state: both source years are independent observations.
    return result, {'source_age_codes':diag,'gami_native_reads':reads}


def publish_partial(path, identity, block, arrays, diagnostics):
    meta={'identity':identity,'block':list(block),'diagnostics':diagnostics}
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.tmp-',delete=False) as f:
        np.savez_compressed(f, metadata=np.asarray(canonical(meta)), **arrays)
        f.flush();os.fsync(f.fileno());temporary=Path(f.name)
    os.link(temporary,path);temporary.unlink()  # Never overwrite an accepted block.


def load_partial(path, identity, block):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        require(meta['identity']==identity and meta['block']==list(block),'Partial identity/block mismatch')
        arrays={k:f[k] for k in f.files if k!='metadata'}  # ZIP CRC is checked on reads.
    require(all(np.all(np.isfinite(arrays[k])) for k in arrays if k!='sd_bounds'), 'Nonfinite partial sums')
    return arrays,meta['diagnostics']


def validate_sources(gami, cci):
    ds=gami['forest_age']
    require(text_attr(gami.attrs['version'])=='v2.1','Wrong GAMI version')
    require(ds.shape==(20,202500,405000,2) and ds.chunks==(1,7789,15577,1)
            and str(ds.dtype)=='int16' and ds.fillvalue==-32767,'GAMI layout changed')
    members=gami['members'][:].astype(int)
    require(len(set(members))==20 and set(members)==set(range(20)),'GAMI member labels changed')
    require(text_attr(gami['time'].attrs['units'])=='days since 2010-01-01'
            and np.array_equal(gami['time'][:],[0,3652]),'GAMI time mapping changed')
    align=[]
    for i,f in enumerate(cci):
        require(text_attr(f.attrs['product_version'])=='7.0'
                and text_attr(f.attrs['time_coverage_start']).startswith(str(YEARS[i])), 'Wrong CCI source year/version')
        for name in ('agb','agb_sd'):
            d=f[name]
            require(d.shape==(1,157500,405000) and d.chunks==(1,11250,11250)
                    and str(d.dtype)=='int16' and text_attr(d.attrs['units'])=='Mg/ha'
                    and int(np.asarray(d.attrs['_FillValue']).ravel()[0])==-32768, 'CCI layout/units changed')
            require(float(np.asarray(d.attrs.get('scale_factor',1)).ravel()[0])==1
                    and float(np.asarray(d.attrs.get('add_offset',0)).ravel()[0])==0,'CCI packing changed')
        align.append(validate_alignment(gami['latitude'][:],gami['longitude'][:],f['lat'][:],f['lon'][:]))
    return members,align


def assemble(out, blocks, identity, members, regions, grid, weight_correction):
    target=out/'observations.nc'
    require(not target.exists(), 'Existing unreceipted final output cannot be trusted or re-signed; '
            'preserve/quarantine it and rebuild from accepted partials, without rescanning raw science arrays')
    total=None;geo=np.zeros((2,len(members),180,360,len(METRICS)))
    diagnostics=[]
    for block in blocks:
        p=out/'partials'/f'{block[0]:06d}_{block[1]:06d}.npz'
        a,d=load_partial(p,identity,block);diagnostics.append(d)
        if total is None:
            total={k:np.zeros_like(a[k]) for k in ('hist_count','hist_area_ha','dry_mass_Mg','stats')}
            total['sd_bounds']=np.full_like(a['sd_bounds'],np.nan)
        for k in ('hist_count','hist_area_ha','dry_mass_Mg','stats'):total[k]+=a[k]
        total['sd_bounds'][...,0]=np.fmin(total['sd_bounds'][...,0],a['sd_bounds'][...,0])
        total['sd_bounds'][...,1]=np.fmax(total['sd_bounds'][...,1],a['sd_bounds'][...,1])
        for j,(r,c) in enumerate(a['cells']):geo[:,:,r,c]+=a['geo'][:,:,j]
    # Geographic totals are independent of regional mixing, including UNASSIGNED.
    geographic=geo.sum(axis=(2,3));regional=total['stats'].sum(axis=2)
    delta=np.abs(geographic-regional)
    require(np.allclose(geographic,regional,rtol=2e-10,atol=1e-4),'Geographic/regional totals do not close')
    hc,ha=total['hist_count'],total['hist_area_ha']
    require(np.all(ha[:,:,1] <= ha[:,:,0]+1e-5),'Common GAMI exceeds native')
    require(np.all(ha[:,:,3] <= ha[:,:,2]+1e-5),'Common AGB exceeds native')
    require(np.allclose(bin_histogram(ha).sum(-1)+ha[...,299],ha.sum(-1),rtol=1e-12,atol=1e-5), 'Age bins plus unresolved 299 do not close')
    require(np.allclose(ha[:,:,3].sum(-1)[0],ha[:,:,3].sum(-1)[1],rtol=1e-12,atol=1e-5), 'Common biomass area differs between years')
    for support,index in ((0,1),(1,3),(2,4),(3,6)):
        require(np.allclose(ha[:,:,support].sum(-1),total['stats'][...,index],rtol=1e-12,atol=1e-5), 'Area histogram/statistic disagreement')
    require(np.allclose(total['dry_mass_Mg'][:,:,0].sum(-1),total['stats'][...,5],rtol=1e-12,atol=1e-4), 'Dry-mass histogram/statistic disagreement')
    data={
        'integer_age_count':(('year','member','support','region','integer_age'),hc),
        'integer_age_area_ha':(('year','member','support','region','integer_age'),ha),
        'age_bin_count':(('year','member','support','region','age_bin'),bin_histogram(hc)),
        'age_bin_area_ha':(('year','member','support','region','age_bin'),bin_histogram(ha)),
        'unresolved_299_count':(('year','member','support','region'),hc[...,299]),
        'unresolved_299_area_ha':(('year','member','support','region'),ha[...,299]),
        'agb_dry_mass_by_age_Mg':(('year','member','biomass_support','region','integer_age'),total['dry_mass_Mg']),
        'sd_minmax':(('year','member','biomass_support','region','bound'),total['sd_bounds'])}
    for i,name in enumerate(METRICS):
        data[name]=(('year','member','region'),total['stats'][...,i])
        data['geographic_'+name]=(('year','member','lat','lon'),geo[...,i].astype(np.int64) if name.endswith('_count') else geo[...,i])
    ds=xr.Dataset(data,coords={'year':list(YEARS),'member':members,'support':list(SUPPORTS),
        'biomass_support':list(BIOMASS_SUPPORTS),'region':regions,'integer_age':np.arange(301),
        'age_bin':list(LABELS),'bound':['min','max'],'lat':89.5-np.arange(180),'lon':-179.5+np.arange(360)},
        attrs={'run_identity':identity,'support_policy':POLICY,'status':'data_reduction_not_observational_validation',
               'age_299':'Unresolved published bin; retained in raw histogram and mass, excluded from age bins.',
               'age_300':'Mature-class code, not exact biological age.',
               'regional_allocation':'Conditional 1-degree geographic mixture; equivalent counts, not independent samples.',
               'sd_policy':'Missing SD does not discard AGB. SD times area and min/max are descriptors, not regional uncertainty.',
               'carbon_conversion':'None: AGB remains woody oven-dry mass.',
               'model_projection':'Not identified; no LUH forest denominator or coupling to model states.'})
    for prefix,area_index,mass_index in (('',4,5),('common_',6,7)):
        area=total['stats'][...,area_index];mass=total['stats'][...,mass_index]
        density=np.full_like(area,np.nan);np.divide(mass,area,out=density,where=area>0)
        ds[prefix+'agb_density_Mg_per_ha']=(('year','member','region'),density)
    for name in ds.data_vars:
        ds[name].attrs['units']='ha' if 'area_ha' in name else 'Mg dry mass' if name.endswith('_Mg') else 'Mg/ha' if name=='sd_minmax' or name.endswith('_Mg_per_ha') else 'pixels' if name.startswith('geographic_') else 'equivalent pixels'
    temporary=out/'.tmp-observations.nc'
    ds.to_netcdf(temporary,engine='h5netcdf',encoding={k:{'zlib':True,'complevel':4} for k in ds.data_vars})
    os.link(temporary,target);temporary.unlink()
    unknown=sum(sum(v['unexpected_age_codes'].values()) for d in diagnostics for yy in d['source_age_codes'] for v in yy)
    report={'identity':identity,'status':'complete_requires_source_review' if unknown or np.any(total['stats'][...,10]>0) or np.any(total['stats'][...,14]>0) else 'complete_with_source_caveats',
            'raw_checksum_repeated':False,'raw_files_complete_before_open':True,'model_run':False,
            'grid':grid,'regional_weight_roundoff_residual_correction_max':weight_correction,
            'max_absolute_geographic_regional_metric_difference':delta.max(axis=(0,1)).tolist(),
            'metric_order':list(METRICS),'unexpected_age_count':unknown,
            'unresolved_299_native_count':hc[:,:,0,:,299].sum(axis=2).tolist(),
            'source_diagnostics_by_block':diagnostics,'source_block_count':len(blocks),
            'output':{'path':target.name,'sha256':digest(target),'bytes':target.stat().st_size}}
    atomic_json(out/'completion.json',report,exclusive=True)
    return report


def audit_retained_cell(root, out):
    """One existing WA cell, cached two-year GAMI only; canonical complete CCI only."""
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from audit_gami_endpoints import CachedRanges
    gm=json.loads((root/'manifests/raw_gami_v21.json').read_text())
    cm=json.loads((root/'config/cci_observations.json').read_text())
    cp=[];identities=[]
    for year in YEARS:
        source=next(x for x in cm['files'] if x['year']==year)
        path,identity=completed_input(root,cm['destination'],source,'phase')
        cp.append(path);identities.append(identity)
    legacy=root/'results/gami_observation_preparation'
    layout=json.loads((legacy/'gami_100m_layout.json').read_text())
    require(layout['url']==gm['file_100m']['url'] and layout['file_bytes']==gm['file_100m']['size_bytes']
            and layout['head_headers']['ETag']==gm['file_100m']['etag'],'Legacy cache source mismatch')
    pilot=json.loads((legacy/'gami_100m_one_degree_pilot.json').read_text())
    cache={int(p.name):p.read_bytes() for p in (legacy/'range_cache').iterdir() if p.name.isdigit()}
    raw=CachedRanges(gm['file_100m'],cache,None,0)  # Missing bytes fail before any HTTP request.
    row,col=114,298;top,left=row*1125,col*1125;bottom,right=top+1125,left+1125
    with ExitStack() as stack:
        high=stack.enter_context(h5py.File(raw,'r',rdcc_nbytes=1024**2))
        low=[stack.enter_context(h5py.File(p,'r',rdcc_nbytes=1024**2)) for p in cp]
        source_members=high['members'][:].tolist();mi=source_members.index(0)
        require(high['forest_age'].shape==(20,202500,405000,2),'Retained GAMI schema changed')
        require(np.array_equal(high['time'][:],[0,3652]),'Retained GAMI time changed')
        ages=np.stack([high['forest_age'][mi,top:bottom,left:right,y] for y in range(2)])
        cci=ChunkCache(low,1_020_000_000)
        agb=np.stack([cci.window(y,'agb',top-11250,bottom-11250,left,right) for y in range(2)])
        sd=np.stack([cci.window(y,'agb_sd',top-11250,bottom-11250,left,right) for y in range(2)])
        area=np.broadcast_to(row_areas_ha(np.arange(top,bottom))[:,None],(1125,1125))
        count,areas,mass,stats,bounds,diag=reduce_patch(ages,agb,sd,area)
        require(int(count[0,0].sum())==pilot['valid_source_pixels'],'Retained WA source count changed')
        require(np.isclose(areas[0,0].sum()*10000,pilot['valid_forest_area_m2'],rtol=2e-9),'WA ellipsoid area changed')
        # Independently select flat indices and sum products, without the reduction masks.
        flat_age=ages.reshape(2,-1);flat_agb=agb.reshape(2,-1);flat_sd=sd.reshape(2,-1);flat_area=area.ravel()
        native=[]
        for y in range(2):
            indices=np.flatnonzero(np.isin(flat_age[y],np.arange(301)) & np.isin(flat_agb[y],np.arange(10001)))
            native.append(indices)
            require(np.isclose(np.dot(flat_agb[y,indices],flat_area[indices]),mass[y,0].sum(),rtol=1e-12),'Direct native dry-mass sum disagrees')
        common=np.intersect1d(native[0],native[1])
        for y in range(2):
            require(np.isclose(np.dot(flat_agb[y,common],flat_area[common]),mass[y,1].sum(),rtol=1e-12),'Direct common dry-mass sum disagrees')
        fraction_path=root/gm['destination']/gm['fraction_1deg']['name']
        with h5py.File(fraction_path,'r') as f:
            fi=f['members'][:].tolist().index(0)
            for y in range(2):
                require(count[y,0,299]==0,'WA now contains unresolved 299; candidate bin comparison needs review')
                expected=np.rint(f['forest_age'][fi,:,row,col,y]*count[y,0].sum()).astype(int)
                require(np.array_equal(expected,bin_histogram(count)[y,0]),'Published WA class fractions disagree')
        report={'status':'one_cell_matched_support_acceptance_not_global_validation','source':gm['file_100m'],
            'cci_complete_inputs':identities,'member_label':0,'source_member_position':mi,'fraction_member_position':fi,
            'years':list(YEARS),'latitude':-24.5,'longitude':118.5,'gami_ranges_network_bytes':raw.transferred,
            'canonical_gami_opened':False,'cci_native_chunks_read':cci.reads,'cci_cache_peak_bytes':cci.peak_bytes,
            'metric_order':list(METRICS),'metrics':stats.tolist(),'integer_age_count':count.tolist(),
            'source_diagnostics':diag,'sd_minmax':np.where(np.isfinite(bounds),bounds,None).tolist(),'independent_direct_mass_sums_passed':True,
            'published_fraction_counts_passed_both_years':True,'workflow_sha256':digest(Path(__file__))}
    raw.close()
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',report,exclusive=True)
    print(json.dumps({k:report[k] for k in ('status','gami_ranges_network_bytes','cci_native_chunks_read','cci_cache_peak_bytes')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-retained-cell',action='store_true',help='Only the retained WA two-year GAMI cache and two completed CCI files; zero network.')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache-bytes',type=int,default=6_100_000_000)
    p.add_argument('--max-new-blocks',type=int,help='Stop cleanly after this many newly completed spatial blocks; resume with identical identity.')
    p.add_argument('--preflight-only',action='store_true',help='Require complete receipts; check headers/identity only, without science blocks.')
    args=p.parse_args();root=args.root.resolve();out=root/args.out
    if args.audit_retained_cell:
        require(not args.preflight_only,'Choose either retained-cell audit or full-file preflight')
        audit_retained_cell(root,out);return
    require(args.cache_bytes>=253_125_000,'At least one CCI native block must fit')
    require(args.max_new_blocks is None or args.max_new_blocks>0,'max-new-blocks must be positive')
    gm=root/'manifests/raw_gami_v21.json';cm=root/'config/cci_observations.json'
    g=json.loads(gm.read_text());c=json.loads(cm.read_text())
    gp,gi=completed_input(root,g['destination'],g['file_100m'],'status')
    cp=[];inputs=[gi]
    for year in YEARS:
        source=next(x for x in c['files'] if x['year']==year)
        path,info=completed_input(root,c['destination'],source,'phase');cp.append(path);inputs.append(info)
    wp=root/'data/processed/cr_forest850/v1-r1/forest850_region_weights.nc'
    identity_data={'policy':POLICY,'inputs':inputs,'weights_sha256':digest(wp),
        'gami_manifest_sha256':digest(gm),'cci_config_sha256':digest(cm),
        'workflow_sha256':digest(Path(__file__)),'cache_bytes':args.cache_bytes,'years':YEARS}
    identity=hashlib.sha256(canonical(identity_data).encode()).hexdigest()
    with ExitStack() as stack:
        high=stack.enter_context(h5py.File(gp,'r',rdcc_nbytes=1024**2))
        low=[stack.enter_context(h5py.File(path,'r',rdcc_nbytes=1024**2)) for path in cp]
        members,grid=validate_sources(high,low)
        with h5py.File(wp,'r') as f:
            require(np.array_equal(f['lat_region'][:],89.5-np.arange(180)) and np.array_equal(f['lon_region'][:],-179.5+np.arange(360)),'Regional coordinate mismatch')
            weights,correction=validate_weights(f['region_fraction'][:])
            regions=[text_attr(x) for x in f['reg_land'][:]]+['UNASSIGNED']
        if args.preflight_only:
            print(json.dumps({'status':'complete_receipts_and_headers_verified','identity':identity,'grid':grid,'no_science_blocks_read':True}));return
        out.mkdir(parents=True,exist_ok=True)
        lock=stack.enter_context((out/'.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=out/'identity.json'
        if binding.exists():require(json.loads(binding.read_text())==json.loads(canonical(identity_data)),'Output identity changed; use a new directory')
        else:
            require(not any(x.name!='.lock' for x in out.iterdir()),'Unbound nonempty output directory')
            atomic_json(binding,identity_data,exclusive=True)
        if (out/'completion.json').exists():
            validate_completed_output(out,identity)
            print(json.dumps({'status':'already_complete','identity':identity}));return
        (out/'partials').mkdir(exist_ok=True)
        ds=high['forest_age'];sy,sx=ds.chunks[1:3]
        blocks=[(y,x) for y in range(0,ds.shape[1],sy) for x in range(0,ds.shape[2],sx)]
        cache=ChunkCache(low,args.cache_bytes);new=0;finished=0;started=time.monotonic()
        for block in blocks:
            path=out/'partials'/f'{block[0]:06d}_{block[1]:06d}.npz'
            if path.exists():load_partial(path,identity,block);finished+=1;continue
            def active_member(completed_members):
                atomic_json(out/'progress.json',{'identity':identity,'finished_blocks':finished,
                    'total_blocks':len(blocks),'active_block':block,'completed_members':completed_members,
                    'cci_native_reads':cache.reads,'cci_cache_peak_bytes':cache.peak_bytes,
                    'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            arrays,diag=reduce_block(ds,cache,*block,weights,progress=active_member)
            diag.update(cci_native_reads_cumulative_this_process=cache.reads,cci_cache_peak_bytes=cache.peak_bytes)
            publish_partial(path,identity,block,arrays,diag);new+=1;finished+=1
            progress={'identity':identity,'finished_blocks':finished,'total_blocks':len(blocks),'new_blocks_this_process':new,
                      'last_block':block,'elapsed_seconds_this_process':time.monotonic()-started,
                      'cci_native_reads':cache.reads,'cci_cache_peak_bytes':cache.peak_bytes,
                      'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            if args.max_new_blocks and new>=args.max_new_blocks:return
        # Refuse source mutation during reduction; no repeated raw byte digest.
        for info in inputs:
            s=Path(info['path']).stat()
            require((s.st_size,s.st_mtime_ns,s.st_ino)==(info['size'],info['mtime_ns'],info['inode']),'Raw source changed during reduction')
        report=assemble(out,blocks,identity,members,regions,grid,correction)
        print(json.dumps({k:report[k] for k in ('status','identity','source_block_count','unexpected_age_count')}))


if __name__=='__main__':main()
