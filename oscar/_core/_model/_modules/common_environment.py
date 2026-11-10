"""Select accepted annual environment for an explicit LUH land-use interval.

A LUH interval y -> y+1 uses common_environment year y+1. Native climate
values are annual substep-end means and CO2 is prescribed For. Holding that
annual environment constant at every land-substep midpoint is a temporal
approximation, not reconstruction of within-year climate. This module selects
and validates values only; it never recalculates climate or calibration.
"""
from numbers import Integral

import numpy as np
import xarray as xr

UNITS = {'D_CO2':'ppm','D_Tl':'K','r_Pl':'1'}
AXES = {'D_CO2':('config',),'D_Tl':('config','reg_land'),'r_Pl':('config','reg_land')}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class CommonEnvironmentAdapter:
    """A private snapshot of caller-selected, accepted original members/regions.

    Pass a single accepted common_environment dataset and matching parameters.
    Member and region selection belongs to the caller; this adapter never
    reorders, interpolates, broadcasts missing axes, or interprets scenario
    quality flags. Data/scientific-use acceptance remains an upstream decision.
    """
    def __init__(self, common_environment, parameters):
        required = set(UNITS) | {'initialization_record'}
        _require(required.issubset(common_environment.data_vars), 'Missing required common-environment fields')
        for name, dimension in [('config','config'),('source_config','config'),('reg_land','reg_land')]:
            for data in (common_environment,parameters):
                _require(name in data.coords and data[name].dims==(dimension,), 'Missing or invalid coordinate: '+name)
                values = data[name].values
                _require(len(values)>0 and len(np.unique(values))==len(values), 'Coordinate must be nonempty and unique: '+name)
            _require(np.array_equal(common_environment[name],parameters[name]), 'Coordinate order/identity mismatch: '+name)
        _require('year' in common_environment.coords and common_environment.year.dims==('year',), 'Missing annual year axis')
        years = common_environment.year.values
        _require(np.issubdtype(years.dtype,np.integer) and len(years)>0, 'Years must be nonempty integer annual labels')
        _require(np.all(np.diff(years)==1), 'Years must be unique, increasing and consecutive; no interpolation')
        flags = common_environment.initialization_record
        _require(flags.dims==('year',) and np.isin(flags.values,[0,1]).all(), 'Invalid initialization flags')
        for name, axes in AXES.items():
            value = common_environment[name]
            _require(set(value.dims)=={'year',*axes}, 'Wrong named axes: '+name)
            _require(value.attrs.get('units')==UNITS[name], 'Wrong units: '+name)
            _require(np.issubdtype(value.dtype,np.floating) and np.isfinite(value).all(), 'Missing or invalid environment values: '+name)
        _require(bool((common_environment.r_Pl>0).all()), 'Relative precipitation must be positive')
        self._data = common_environment[list(required)].copy(deep=True).load()
        self.member_ids = tuple(common_environment.source_config.values.tolist())
        self.region_ids = tuple(common_environment.reg_land.values.tolist())

    def for_interval(self, start_year, end_year):
        """Return target-year fields for exactly one annual interval.

        Compose with the existing coefficient API once per interval:
        ``coefficients = reference_adapter.prepare(adapter.for_interval(y, y+1))``.
        Reuse those coefficients for that interval's land substeps. Initialization
        rows are never annual climate means, even when their CO2 matches history.
        """
        _require(all(isinstance(v,Integral) and not isinstance(v,bool) for v in (start_year,end_year)), 'Interval bounds must be integer years')
        _require(end_year==start_year+1, 'Expected one annual interval y to y+1')
        _require(end_year in self._data.year.values, 'Requested target year is unavailable')
        selected = self._data.sel(year=end_year,drop=True)
        _require(int(selected.initialization_record)==0, 'Initialization record is not an integrated annual environment')
        result = xr.Dataset({name:selected[name].transpose(*axes).copy(deep=True) for name,axes in AXES.items()})
        result.attrs = dict(self._data.attrs)
        result.attrs.update(interval_start_year=int(start_year),interval_end_year=int(end_year),
            environment_source_year=int(end_year),
            temporal_approximation='Hold this native annual substep-end mean climate and prescribed annual CO2 constant at every land-substep midpoint; no interpolation.')
        return result
