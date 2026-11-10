"""Re-evaluate accepted C-R reduction at fixed fit/holdout ages in ONE raster pass."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'manifests/regional_continuous_curves.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    spec = importlib.util.spec_from_file_location('accepted_reducer', ROOT/'workflows/reduce_cr_forest850.py')
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    np, xr = reducer.np, reducer.xr
    train, diagnostic = cfg['fit_ages'], cfg['diagnostic_ages']
    reducer.require(not set(train) & set(diagnostic), 'Fit and holdout ages overlap')
    reducer.AGES = np.array(sorted(train + diagnostic) + [np.inf], dtype=float)
    out = ROOT/cfg['output_directory']/'nodes'
    out.mkdir(parents=True, exist_ok=False)
    accepted = ROOT/cfg['accepted_reduction']
    with xr.open_dataset(accepted/'forest850_region_weights.nc') as w:
        weights = w.load()
    original = json.loads((accepted/'report.json').read_text())
    report = deepcopy(original)
    report.update(started_utc=datetime.now(timezone.utc).isoformat(),
                  execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  accepted_reduction=cfg['accepted_reduction'],
                  output_directory=str(out.relative_to(ROOT)),
                  ages_years=sorted(train + diagnostic) + ['infinity'],
                  fit_ages=train, diagnostic_ages=diagnostic,
                  purpose='Continuous-proxy approximation at fixed training and holdout ages, using the prepared forest mask and regional weights.')
    source_cfg = json.loads((ROOT/cfg['reduction_config']).read_text())
    paths = {name:ROOT/source_cfg['cr_directory']/filename for name,filename in
             [('A','cr_a.tif'),('b','cr_b.tif'),('k','cr_k.tif')]}
    reducer.save(out/'protocol.json', cfg)
    reducer.reduce(paths, out, weights.region_fraction.values, weights.forest850_area.values,
                   weights.fine_row_area_fraction.values, weights.source_forest_area.values,
                   report, out/'progress.json')
    # Compare shared support and overlapping ages with the prepared reduction.
    with xr.open_dataset(accepted/'regional_cr_curves.nc') as old, xr.open_dataset(out/'regional_cr_curves.nc') as new:
        static = [n for n in old.data_vars if 'age' not in old[n].dims]
        static_errors = {n:float(abs(new[n]-old[n]).max()) for n in static}
        common = np.intersect1d(old.age.values, new.age.values)
        curve_errors = {n:float(abs(new[n].sel(age=common)-old[n].sel(age=common)).max())
                        for n in ['S','S_prime','S_area_weighted','S_prime_area_weighted']}
    reducer.require(max(static_errors.values()) < 1e-8, 'Accepted support was changed')
    reducer.require(max(curve_errors.values()) < 1e-12, 'Existing diagnostic curves were changed')
    reducer.save(out/'reuse-checks.json', {'static_max_absolute_errors':static_errors,
                                         'shared_age_max_absolute_errors':curve_errors,
                                         'raster_passes':1})
    print(json.dumps({'status':'nodes_complete','ages':len(reducer.AGES),'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
