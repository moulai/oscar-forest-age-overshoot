"""Build generic Hermite coefficients and a C1 mature tail from fixed training nodes."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import numpy as np
import xarray as xr
from fit_regional_continuous_curves import metrics, require

ROOT=Path(__file__).resolve().parents[1]


def coefficients(knots, stock, slope):
    """Ascending powers in local age x=a-left: c0+c1*x+c2*x^2+c3*x^3."""
    knots,stock,slope=[np.asarray(x,dtype=float) for x in (knots,stock,slope)]
    require(knots.ndim==1 and knots.shape==stock.shape==slope.shape, 'Knot shapes differ')
    h=np.diff(knots)
    require(np.all(h>0) and np.all(np.isfinite(np.r_[knots,stock,slope])), 'Invalid knot data')
    secant=np.diff(stock)/h
    return np.stack([stock[:-1],slope[:-1],(3*secant-2*slope[:-1]-slope[1:])/h,
                     (slope[:-1]+slope[1:]-2*secant)/(h*h)],axis=-1)


def minimum_derivative(knots, coefs):
    """Exact candidate extrema of the derivative quadratic on every closed interval."""
    h=np.diff(knots);c1,c2,c3=coefs[:,1],coefs[:,2],coefs[:,3]
    endpoint=c1+2*c2*h+3*c3*h*h
    minimum=np.minimum(c1,endpoint)
    vertex=np.divide(-c2,3*c3,out=np.full_like(c2,np.nan),where=c3!=0)
    inside=(vertex>0)&(vertex<h)
    minimum[inside]=np.minimum(minimum[inside],c1[inside]+2*c2[inside]*vertex[inside]+3*c3[inside]*vertex[inside]**2)
    return minimum


def tail_parameters(stock, slope):
    gap=1-float(stock)
    require(np.isfinite(gap) and gap>0 and np.isfinite(slope) and slope>0, 'Tail needs positive mature deficit and slope')
    return gap,float(slope)/gap


def evaluate(age, knots, coefs, tail_gap, tail_rate):
    """Evaluate generic stored coefficients; no raw C-R parameters or raster access."""
    age=np.asarray(age,dtype=float)
    require(np.all((age>=knots[0])&~np.isnan(age)), 'Age before first knot or NaN')
    stock=np.empty_like(age);slope=np.empty_like(age)
    tail=age>=knots[-1]
    e=np.exp(-tail_rate*(age[tail]-knots[-1]))
    stock[tail]=1-tail_gap*e;slope[tail]=tail_rate*tail_gap*e
    j=np.searchsorted(knots,age[~tail],side='right')-1
    x=age[~tail]-knots[j];c=coefs[j]
    stock[~tail]=c[:,0]+x*(c[:,1]+x*(c[:,2]+x*c[:,3]))
    slope[~tail]=c[:,1]+x*(2*c[:,2]+3*x*c[:,3])
    return stock,slope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'manifests/regional_continuous_curves.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    tail_start=cfg['hermite']['tail_start_year'];base=ROOT/cfg['output_directory'];out=base/'hermite';out.mkdir(exist_ok=False)
    with xr.open_dataset(base/'nodes/regional_cr_curves.nc') as ds: nodes=ds.load()
    with xr.open_dataset(base/'q3/wood_calibration.nc') as ds: cal=ds.load()
    age=nodes.age.values;train=np.isin(age,cfg['fit_ages']);held=np.isin(age,cfg['diagnostic_ages'])
    knot_mask=train&(age<=tail_start);knots=age[knot_mask]
    require(knots[0]==0 and knots[-1]==tail_start and not np.any(train&held), 'Registered knot or split mismatch')
    nr=nodes.sizes['reg_land'];weights=['capacity','area'];ni=len(knots)-1
    coefs_all=np.empty((2,nr,ni,4));minima=np.empty((2,nr,ni));gap_all=np.empty((2,nr));rate_all=np.empty((2,nr))
    proxy_s=np.empty((2,nr,len(age)));proxy_sp=np.empty_like(proxy_s);rows=[]
    for w,(sn,spn) in enumerate([('S','S_prime'),('S_area_weighted','S_prime_area_weighted')]):
        for r,region in enumerate(nodes.reg_land.values):
            raw_s=nodes[sn].values[r];raw_sp=nodes[spn].values[r]
            coefs=coefficients(knots,raw_s[knot_mask],raw_sp[knot_mask]);mins=minimum_derivative(knots,coefs)
            gap,rate=tail_parameters(raw_s[knot_mask][-1],raw_sp[knot_mask][-1])
            s,sp=evaluate(age,knots,coefs,gap,rate);ds=s-raw_s;dsp=sp-raw_sp
            lam=cal.lambda_w_cal.values[r];ids=cal.source_config.values
            train_m=metrics(ds,dsp,lam,age,ids,train);hold_m=metrics(ds,dsp,lam,age,ids,held)
            finite_m=metrics(ds,dsp,lam,age,ids,np.isfinite(age));tail_m=metrics(ds,dsp,lam,age,ids,held&(age>tail_start))
            # Analytical interval extrema, not a sampled monotonicity guess.
            monotone=bool(np.all(mins>=0) and raw_s[knot_mask].min()>=0 and raw_s[knot_mask].max()<=1)
            h=np.diff(knots)
            right_s=coefs[:,0]+h*(coefs[:,1]+h*(coefs[:,2]+h*coefs[:,3]))
            right_sp=coefs[:,1]+h*(2*coefs[:,2]+3*h*coefs[:,3])
            continuity=max(float(abs(right_s-raw_s[knot_mask][1:]).max()),float(abs(right_sp-raw_sp[knot_mask][1:]).max()),
                           abs(1-gap-raw_s[knot_mask][-1]),abs(gap*rate-raw_sp[knot_mask][-1]))
            accepted=monotone and continuity<1e-12 and finite_m['max_abs_delta_S']<=cfg['stock_error_budget'] and finite_m['max_abs_delta_gw']<=cfg['production_error_budget']
            rows.append({'weighting':weights[w],'region':str(region),'fit_nodes':train_m,'holdout_nodes':hold_m,
                         'all_finite_nodes':finite_m,'tail_holdout':tail_m,'minimum_interval_derivative':float(mins.min()),
                         'monotone_every_interval':monotone,'max_knot_value_or_slope_mismatch':continuity,
                         'tail_gap':gap,'tail_rate_per_year':rate,'accepted_for_candidate_test':bool(accepted)})
            coefs_all[w,r]=coefs;minima[w,r]=mins;gap_all[w,r]=gap;rate_all[w,r]=rate
            proxy_s[w,r]=s;proxy_sp[w,r]=sp
    generic=nodes[[n for n in nodes.data_vars if 'age' not in nodes[n].dims]].copy()
    generic=generic.assign_coords(weighting=weights,knot=np.arange(len(knots)),interval=np.arange(ni),power=np.arange(4))
    generic['knot_age']=(('knot',),knots)
    generic['coefficient']=(('weighting','reg_land','interval','power'),coefs_all)
    generic['minimum_interval_derivative']=(('weighting','reg_land','interval'),minima)
    generic['tail_gap']=(('weighting','reg_land'),gap_all);generic['tail_rate']=(('weighting','reg_land'),rate_all)
    generic.attrs.update(representation='Local-age cubic Hermite plus matched single-exponential mature tail',
                         evaluation='For interval j: x=age-knot_age[j], S=sum(coefficient[j,p]*x**p), derivative=sum(p*coefficient[j,p]*x**(p-1)). For age>=last knot: S=1-tail_gap*exp(-tail_rate*(age-last knot)); derivative=tail_gap*tail_rate*exp(...).',
                         tail_start_year=tail_start,tail_scope='Numerical approximation, not a new ecological timescale. Age 750 held out; training age 1000 checks error but does not estimate tail rate.',
                         calibration_environment=cfg['calibration_environment'],scope=cfg['limitations'])
    generic.knot_age.attrs['units']='yr';generic.tail_rate.attrs['units']='yr-1'
    generic.minimum_interval_derivative.attrs['units']='yr-1'
    generic.to_netcdf(out/'regional_continuous_curves.nc');cal.to_netcdf(out/'wood_calibration.nc')
    diagnostics=xr.Dataset({'S_proxy':(('weighting','reg_land','age'),proxy_s),'S_prime_proxy':(('weighting','reg_land','age'),proxy_sp),
                            'is_fit_age':(('age',),train.astype('int8')),'is_holdout_age':(('age',),held.astype('int8'))},
                           coords={'weighting':weights,'reg_land':nodes.reg_land.values,'age':age})
    diagnostics.to_netcdf(out/'diagnostic_curves.nc')
    passed=all(row['accepted_for_candidate_test'] for row in rows)
    report={'status':'passed_candidate_curve_gate' if passed else 'failed_hermite_approximation_budget',
            'created_utc':datetime.now(timezone.utc).isoformat(),'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'representation':'piecewise cubic Hermite with C1 matched exponential tail','config':cfg,
            'fit_node_count':int(train.sum()),'hermite_knot_count':len(knots),'holdout_count':int(held.sum()),'member_count':500,
            'tail_start_year':tail_start,'tail_holdout_age':age[held&(age>tail_start)].tolist(),
            'age_1000_role':'Original training membership retained; only error check, not used in tail-rate calculation.',
            'rows':rows,'failed_curve_count':sum(not row['accepted_for_candidate_test'] for row in rows),
            'scope':'Registered finite-node numerical approximation acceptance only. Exact interval derivative minima prove proxy monotonicity, not a uniform proxy-to-source error bound. Not independent ecological validation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','hermite_knot_count','holdout_count','member_count','failed_curve_count']}),flush=True)
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
