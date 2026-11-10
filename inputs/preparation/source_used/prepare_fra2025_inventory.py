#!/usr/bin/env python3
"""Reduce official FRA2025 country observations without filling missing stocks."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

# All source totals are in 1000 ha or million tonnes. Both need /1000 for
# the requested Mha or Pg quantities. No fixed biomass carbon fraction is used.
VARIABLES = {
    '1a_forestArea': ('forest_area_Mha', 'Mha', 'Area (1000 ha)'),
    '2c_agb_total': ('aboveground_biomass_Pg', 'Pg biomass (source mass)', 'Total forest Biomass (million tonnes)'),
    '2c_bgb_total': ('belowground_biomass_Pg', 'Pg biomass (source mass)', 'Total forest Biomass (million tonnes)'),
    '2d_carbon_agb_total': ('aboveground_carbon_PgC', 'PgC', 'Total forest carbon (million tonnes)'),
    '2d_carbon_bgb_total': ('belowground_carbon_PgC', 'PgC', 'Total forest carbon (million tonnes)'),
    '2d_carbon_dw_total': ('deadwood_carbon_PgC', 'PgC', 'Total forest carbon (million tonnes)'),
    '2d_carbon_litter_total': ('litter_carbon_PgC', 'PgC', 'Total forest carbon (million tonnes)'),
    '2d_carbon_soil_total': ('soil_carbon_PgC', 'PgC', 'Total forest carbon (million tonnes)'),
}
FLAG_DEFINITIONS = {
    'A': 'Normal value, official data reported by country/area',
    'I': 'Imputed value, desk study, data compiled by FAO',
    'M': 'Missing value, cannot exist, denominator is zero or missing',
    'O': 'Missing value',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_values(values, flags):
    """Recognize missing observations explicitly; reject unknown nonnumeric data."""
    if not set(flags).issubset(FLAG_DEFINITIONS):
        raise ValueError('Unknown source observation flag')
    numbers = pd.to_numeric(values, errors='coerce')
    if ((numbers.isna()) & (values != '')).any():
        raise ValueError('Unrecognized nonnumeric source value')
    if (numbers.notna() != flags.isin(['A', 'I'])).any():
        raise ValueError('Source value/flag availability mismatch')
    if (numbers.dropna() < 0).any() or np.isinf(numbers).any():
        raise ValueError('Negative or infinite source stock/area')
    return numbers


def derive_living(frame):
    """Pair above/belowground carbon at country level; other pools stay separate."""
    out = frame.copy()
    out['living_carbon_PgC'] = out.aboveground_carbon_PgC + out.belowground_carbon_PgC
    available = out.living_carbon_PgC.notna()
    both_reported = (out['2d_carbon_agb_total_flag'] == 'A') & (out['2d_carbon_bgb_total_flag'] == 'A')
    out['living_carbon_flag'] = np.where(available, np.where(both_reported, 'A', 'I'), 'O')
    return out


def aggregate(countries, groups):
    """Retain complete totals, available subtotals and their exact area coverage."""
    metrics = [(name, unit, source + '_flag') for source, (name, unit, _) in VARIABLES.items()]
    metrics.append(('living_carbon_PgC', 'PgC', 'living_carbon_flag'))
    rows = []
    for year, year_data in countries.groupby('year', sort=True):
        for group in groups:
            d = year_data if group == 'GLOBAL' else year_data[year_data.reg_land == group]
            for metric, unit, flag in metrics:
                available = d[metric].notna()
                area_all = d.forest_area_Mha.sum(min_count=1)
                area_covered = d.loc[available, 'forest_area_Mha'].sum(min_count=1) if available.any() else 0.
                subtotal = d[metric].sum(min_count=1)
                rows.append({
                    'year': year, 'reg_land': group, 'variable': metric, 'units': unit,
                    'country_count': len(d), 'available_country_count': int(available.sum()),
                    'missing_country_count': int((~available).sum()),
                    'available_subtotal': subtotal,
                    'complete_total': subtotal if len(d) and available.all() else np.nan,
                    'official_reported_country_count': int(((d[flag] == 'A') & available).sum()),
                    'FAO_imputed_country_count': int(((d[flag] == 'I') & available).sum()),
                    'official_reported_subtotal': d.loc[d[flag] == 'A', metric].sum(min_count=1),
                    'FAO_imputed_subtotal': d.loc[d[flag] == 'I', metric].sum(min_count=1),
                    'all_forest_area_Mha': area_all, 'covered_forest_area_Mha': area_covered,
                    'forest_area_coverage_fraction': area_covered / area_all if area_all > 0 and d.forest_area_Mha.notna().all() else np.nan,
                })
    return pd.DataFrame(rows)


def prepare(project, raw, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    archives = list(raw.glob('*.zip'))
    if len(archives) != 1:
        raise ValueError('Expected exactly one official bulk archive in the acquisition directory')
    archive_path = archives[0]
    acquisition = json.loads((raw / 'acquisition.json').read_text())
    source_record = acquisition['sources'][archive_path.name]
    if (source_record['sha256'] != digest(archive_path) or source_record['http_status'] != 200
            or 'assessmentName=fra&cycleName=2025&countryIso=WO' not in source_record['requested_url']):
        raise ValueError('Acquisition does not identify this official FRA2025 snapshot')
    mapping_path = project / 'oscar/_core/_regions/OSCAR_reg_dict.csv'
    names_path = project / 'oscar/_core/_regions/OSCAR_reg_names.csv'
    mapping = pd.read_csv(mapping_path, keep_default_na=False)[['reg_code', 'IAMC_R10']]
    if mapping.reg_code.duplicated().any():
        raise ValueError('Ambiguous ISO3-to-region mapping')
    with names_path.open(newline='') as stream:
        regions = [s.split(':', 1)[0] for row in csv.reader(stream) if row[0] == 'IAMC_R10' for s in row[1:]]
    source_units, member_checksums = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failure')
        def read(prefix):
            names = [n for n in archive.namelist() if n.startswith(prefix) and n.endswith('.csv') and ('/' in prefix or '/' not in n)]
            if len(names) != 1:
                raise ValueError('Ambiguous archive member: ' + prefix)
            content = archive.read(names[0])
            member_checksums[names[0]] = hashlib.sha256(content).hexdigest()
            return pd.read_csv(io.BytesIO(content), keep_default_na=False, dtype=str)
        all_years = read('FRA_Years_')
        if all_years.duplicated(['iso3', 'year']).any() or set(all_years.year) != {'1990', '2000', '2010', '2015', '2020', '2025'}:
            raise ValueError('Unexpected country/year structure')
        if set(all_years.Flag) - {''} != {k + ' = ' + v for k, v in FLAG_DEFINITIONS.items()}:
            raise ValueError('Official flag definitions changed')
        country = all_years[all_years.year.isin(['2010', '2020'])].copy()
        keep = ['iso3', 'iso2', 'm49', 'name', 'regions', 'subregions', 'Desk study', 'year']
        for source, (name, _, unit_text) in VARIABLES.items():
            wide = read('FRA_Years_variables/' + source + '_')
            if unit_text not in set(wide[source]):
                raise ValueError('Unreviewed source units: ' + source)
            source_units[source] = unit_text
            country[name] = parse_values(country[source], country[source + '_flag']) / 1000
            # Verify the long export and independent per-variable table agree.
            if wide.iso3.duplicated().any():
                raise ValueError('Duplicate source country')
            expected = wide.set_index('iso3')
            for year in ['2010', '2020']:
                d = country[country.year == year]
                if d[source].tolist() != expected.loc[d.iso3, year].tolist():
                    raise ValueError('Long/wide export values disagree')
            keep.extend([source, source + '_flag', name])
        country = country[keep]
        country['year'] = country.year.astype(int)
        country = country.merge(mapping, how='left', left_on='iso3', right_on='reg_code', validate='many_to_one')
        country['reg_land'] = country.IAMC_R10.fillna('UNASSIGNED')
        country = derive_living(country.drop(columns=['IAMC_R10', 'reg_code']))
        tiers, ndp = read('Tiers_'), read('NDPYear_')
        for table in [tiers, ndp]:
            if table.iso3.duplicated().any() or set(table.iso3) != set(country.iso3):
                raise ValueError('Country metadata coverage differs')
        metadata = tiers.merge(ndp[['iso3'] + [c for c in ndp if c not in tiers]], on='iso3', validate='one_to_one')
        metadata = metadata.merge(mapping, how='left', left_on='iso3', right_on='reg_code', validate='one_to_one')
        metadata['reg_land'] = metadata.IAMC_R10.fillna('UNASSIGNED')
        metadata = metadata.drop(columns=['IAMC_R10', 'reg_code'])
    regional = aggregate(country, regions + ['UNASSIGNED'])
    global_values = aggregate(country, ['GLOBAL'])
    regional_sum = regional.groupby(['year', 'variable']).available_subtotal.sum(min_count=1)
    direct_sum = global_values.set_index(['year', 'variable']).available_subtotal
    residual = (regional_sum - direct_sum).abs()
    # 1e-10 in Mha or Pg is far below the export's 0.01 source-unit precision.
    if (residual > 1e-10).any():
        raise ValueError('Regional and direct-country available subtotals do not close')
    output.mkdir(parents=True)
    outputs = {}
    for name, table in [('countries_2010_2020.csv', country), ('country_metadata.csv', metadata),
                        ('regional_2010_2020.csv', regional), ('global_2010_2020.csv', global_values)]:
        path = output / name
        table.to_csv(path, index=False, na_rep='')
        reopened = pd.read_csv(path, keep_default_na=False)
        if len(reopened) != len(table):
            raise ValueError('Written CSV row count differs')
        outputs[name] = {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest(path), 'rows': len(table)}
    summary = global_values[global_values.variable.isin(['forest_area_Mha', 'living_carbon_PgC'])]
    receipt = {
        'schema': 'fra2025-regional-inventory-v1', 'status': 'data_prepared_with_explicit_coverage',
        'model_run': False, 'model_comparison': False, 'new_parameters': False,
        'scope': 'Country-reported and FAO-imputed forest-land references; not independent validation or a claim of ecological acceptance',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        'preparer_sha256': digest(__file__), 'acquisition': acquisition,
        'archive': {'path': str(archive_path), 'sha256': source_record['sha256']},
        'archive_members_read': member_checksums, 'source_units': source_units,
        'unit_conversion': '1000 ha / 1000 = Mha; million tonnes / 1000 = Pg; biomass retains the source mass basis (dry-mass wording not independently verified in the read 2025 definitions); carbon is element C',
        'flag_definitions': FLAG_DEFINITIONS,
        'mapping': {'path': str(mapping_path), 'sha256': digest(mapping_path), 'source_column': 'reg_code',
                    'target_column': 'IAMC_R10', 'region_names_sha256': digest(names_path),
                    'policy': 'Exact existing ISO3 mapping only; unmatched countries retained as UNASSIGNED; ROWO is an explicit existing region'},
        'country_count': int(country.iso3.nunique()),
        'unassigned_countries': country.loc[country.reg_land == 'UNASSIGNED', 'iso3'].unique().tolist(),
        'living_carbon_formula': 'country aboveground carbon + belowground carbon, requiring both; excludes deadwood, litter and soil',
        'missing_policy': 'Blank observations remain missing. Available subtotals and area coverage are separate from complete totals; no zero-filling, allocation, rescaling or extrapolation.',
        'regional_global_subtotal_max_abs_difference': float(residual.max()),
        'global_summary': json.loads(summary.to_json(orient='records')),
        'definitions_url': 'https://fra-data.fao.org/definitions/fra/2025/en/tad',
        'metadata_scope': 'Desk-study flag and observation flags kept. Tiers describe forest area and biomass, not an independent carbon-stock tier. NDP dates/methods describe forest-area reporting only.',
        'limitations': [
            'FRA forest land includes qualifying young and temporarily unstocked areas; its domain differs from a current canopy-height map',
            'Country reports can include default conversion factors, interpolated/extrapolated values and remote sensing; A is not a direct-measurement certificate',
            'FRA and CCI can share NFI statistics/methods, and LUH has FAO/HYDE national-statistics lineage; overlap with this FRA2025 snapshot has not been established',
            'Reported soil/deadwood/litter columns have substantial missingness and must not be added to live carbon',
            'Year-to-year inventory stock differences are not NPP or land-use emissions',
            'Global entries are direct sums of available country rows from this dated snapshot, not imputed official global dashboard totals',
        ], 'outputs': outputs,
    }
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.project.resolve(), args.raw.resolve(), args.output.resolve())
    print(json.dumps({'status': result['status'], 'global_summary': result['global_summary']}, indent=2))
