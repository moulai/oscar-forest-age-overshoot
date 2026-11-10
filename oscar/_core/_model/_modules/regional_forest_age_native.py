"""Register regional actual-land states in OSCAR's ordinary Model loop.

Environmental/reference preparation is injected as pure functions consuming
the native Var/Par/For contract. This module contains no independent time loop
and no paired-experiment logic.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import cached_property
import numpy as np
import xarray as xr


def _over_dt(array, dt):
    """Divide a DataArray by a scalar without building an intermediate index."""
    return tuple(array.dims), array.values / dt

from ..._base.cls_main import Model
from ..._base.land_step import LandStep
from .regional_forest_age import STATE_NAMES, apply_annual_events, advance_carbon, validate_purpose_rank
from .regional_forest_secondary_carbon import (SEC_EXCHANGES, SEC_UNITS, SEC_INPUTS, CANDIDATE,
    validate_secondary_contract, initial_secondary_exchange, apply_secondary_agb_request_events)
from .regional_forest_age_diagnostics import (AGE_EXCHANGES, initial_age_exchange, area_by_age,
    HARVEST_EXCHANGES, HARVEST_UNITS, initial_harvest_exchange, harvest_source_summary)

EXCHANGES = ("D_Fland", "D_Eluc", "rf_NBP", "rf_NPP", "rf_Efire",
             "rf_HWP_input", "rf_budget_residual", "rf_event_count",
             "rf_NBP_biome", "rf_Eresp", "rf_Ehwp", "rf_Ecwd", "rf_Esoil",
             "rf_Ecrop", "rf_Egraz", "rf_NPP_W", "rf_NPP_F", "rf_forest_W_mean",
             "rf_forest_F_mean", "rf_forest_area_mean") + AGE_EXCHANGES + HARVEST_EXCHANGES
HWP_EXCHANGES = tuple("rf_HWP_" + name for name in
                      ("target", "achieved", "common", "managed", "shortfall", "excess"))


RANK_EXCHANGES = ('rf_rank_min_age', 'rf_rank_max_age')


def validate_target_rule(layout, purpose_rank, hwp_target_events, rule):
    if rule == 'weighted':
        validate_purpose_rank(layout, purpose_rank, hwp_target_events)
    elif rule == 'rank_minimum_area':
        validate_purpose_rank(layout, purpose_rank)
        if (purpose_rank is None or hwp_target_events is None
                or set(purpose_rank) != set(hwp_target_events)):
            raise ValueError('rank_minimum_area requires matching explicit rank and target event identities')
    else:
        raise ValueError('Unknown HWP target rule')


def rank_zero(layout, event_ids):
    return xr.DataArray(np.zeros((len(layout.members), len(layout.regions), len(event_ids))),
                        dims=('config', 'reg_land', 'managed_event'),
                        coords=dict(config=list(layout.members), reg_land=list(layout.regions), managed_event=list(event_ids)))


def rank_source_bounds(ledger, event_ids, dt):
    selected = ledger.selected_area.sel(event=list(event_ids)).values
    ages = ledger.source_age.values
    valid = (selected > 0) & np.isfinite(ages)[None, None, :]
    bounds = []
    for extreme, fill in ((np.min, np.inf), (np.max, -np.inf)):
        value = extreme(np.where(valid, ages[None, None, :], fill), axis=-1)
        value[~valid.any(axis=-1)] = np.nan
        bounds.append(xr.DataArray(value / dt, dims=('config', 'reg_land', 'managed_event'),
            coords=dict(config=ledger.config, reg_land=ledger.reg_land, managed_event=list(event_ids))))
    return dict(zip(RANK_EXCHANGES, bounds))


@dataclass(frozen=True)
class RegionalLandUpdate:
    layout: object
    coefficients: object
    reference_sink: object
    age_production: tuple | None = None
    preference: object = None
    hwp_target_events: tuple | None = None
    purpose_rank: dict | None = None
    hwp_target_rule: str = "weighted"
    secondary_agb_events: tuple | None = None
    harvest_projection: str | None = None
    secondary_harvest_contract: str = CANDIDATE
    age_clock: object = None

    @cached_property
    def member_layouts(self):
        """One-member layouts built once. A single member keeps the original object."""
        layout = self.layout
        built = []
        for mi, member in enumerate(layout.members):
            if len(layout.members) == 1:
                built.append(layout)
                continue
            built.append(replace(layout, members=(member,),
                                 source_members=(layout.parameter_ids[mi],)
                                 if layout.source_members is not None else None))
        return tuple(built)

    def __call__(self, variables, parameters, forcing, t0, t1, event):
        layout, dt = self.layout, t1-t0
        cohort_ages = None if self.age_clock is None else self.age_clock.ages(np.asarray(layout.births, dtype=float))
        state = variables[list(STATE_NAMES)]
        coefficients = self.coefficients(variables, parameters, forcing, t0, t1)
        product_input = np.zeros((len(layout.members), len(layout.regions)))
        event_residual = np.zeros(len(layout.members))
        age_events = {name: initial_age_exchange(name, state, layout, t0)
                      for name in AGE_EXCHANGES if name.startswith("rf_event_")}
        harvest_events = {name: initial_harvest_exchange(name, layout) for name in HARVEST_EXCHANGES}
        rank_events = self.hwp_target_events or tuple(self.purpose_rank or {})
        rank_rates = ({name: rank_zero(layout, rank_events) for name in RANK_EXCHANGES}
                      if self.purpose_rank is not None else {})
        target_rates = ({name: xr.zeros_like(state.rf_W_primary) for name in HWP_EXCHANGES}
                        if self.hwp_target_events is not None else {})
        if self.hwp_target_events is not None:
            target_rates["rf_HWP_managed_area"] = xr.zeros_like(state.rf_W_primary).expand_dims(
                managed_event=list(self.hwp_target_events), axis=-1)
        secondary_rates = ({name: initial_secondary_exchange(name, state, self.secondary_agb_events)
                            for name in SEC_EXCHANGES} if self.secondary_agb_events is not None else {})
        if event:
            if self.age_clock is not None and self.secondary_agb_events is None:
                raise RuntimeError("frozen age is only used with the secondary-harvest update")
            expected = [ev.event_id for ev in layout.events]
            if list(forcing.event.values) != expected:
                raise ValueError("annual forcing event identities differ from layout")
            demand = forcing.event_area.transpose("reg_land", "event").values
            if list(forcing.reg_land.values) != list(layout.regions):
                raise ValueError("annual forcing region identities differ from layout")
            member_states, member_transfers = [], []
            # Only the discrete event is member-wise. Continuous land, reference
            # and climate use the ordinary batched Model interval.
            for mi, member in enumerate(layout.members):
                member_layout = self.member_layouts[mi]
                member_state = state.isel(config=slice(mi, mi+1))
                member_parameters = parameters.isel(config=slice(mi, mi+1))
                member_forcing = forcing.isel(config=slice(mi, mi+1)) if "config" in forcing.dims else forcing
                weights = (np.broadcast_to(np.float64(1.), (len(layout.regions), len(layout.events), len(layout.slots)))
                           if self.preference is None else self.preference(variables.isel(config=slice(mi, mi+1)), member_parameters, member_forcing, t0))
                pwood = member_parameters.rf_p_wood.transpose("config", "reg_land", "bio_land").values
                phwp = member_parameters.rf_p_hwp.transpose("config", "reg_land", "bio_land", "box_hwp").values
                if self.secondary_agb_events is not None:
                    if tuple(member_forcing.secondary_harvest_label.values) != tuple(self.secondary_agb_events):
                        raise ValueError('secondary request label identities differ from layout')
                    for field in ('k_F_AG', 'k_F_AG_leaf_only'):
                        for coordinate, wanted in (('config', member_layout.members), ('reg_land', layout.regions)):
                            if tuple(member_parameters[field][coordinate].values) != tuple(wanted):
                                raise ValueError('secondary projection coordinate mismatch: ' + field + '/' + coordinate)
                    member_state, transfers = apply_secondary_agb_request_events(
                        member_state, member_layout, t0, demand, weights, pwood, phwp,
                        member_forcing.Q_sec_harvest.transpose('reg_land').values,
                        member_forcing.Q_sec_harvest_by_label.transpose('reg_land','secondary_harvest_label').values,
                        member_parameters.k_F_AG.transpose('config','reg_land').values,
                        secondary_events=self.secondary_agb_events, projection=self.harvest_projection,
                        k_f_ag_leaf_only=member_parameters.k_F_AG_leaf_only.transpose('config','reg_land').values,
                        secondary_harvest_contract=self.secondary_harvest_contract, cohort_ages=cohort_ages)
                elif self.hwp_target_events is None:
                    member_state, transfers = apply_annual_events(member_state, member_layout, t0, demand, weights, pwood, phwp,
                                                            purpose_rank=self.purpose_rank)
                elif self.hwp_target_rule == 'rank_minimum_area':
                    from .regional_forest_rank_management import apply_rank_hwp_target_events
                    member_state, transfers = apply_rank_hwp_target_events(member_state, member_layout, t0, demand, weights, pwood, phwp,
                        member_forcing.HWP_target.transpose("reg_land").values, managed_events=self.hwp_target_events,
                        purpose_rank=self.purpose_rank)
                else:
                    from .regional_forest_management import apply_hwp_target_events
                    member_state, transfers = apply_hwp_target_events(member_state, member_layout, t0, demand, weights, pwood, phwp,
                        member_forcing.HWP_target.transpose("reg_land").values, managed_events=self.hwp_target_events)
                member_states.append(member_state)
                # Summaries read the parcel cube in place. Do not copy it onto config.
                summary = harvest_source_summary(transfers, member_layout)
                keep = ["hwp_added", "budget_residual"] + [name.removeprefix("rf_event_") for name in age_events]
                if self.secondary_agb_events is not None:
                    keep.extend(name.removeprefix("rf_") for name in SEC_EXCHANGES)
                if self.hwp_target_events is not None:
                    keep.extend("hwp_" + name.removeprefix("rf_HWP_") for name in HWP_EXCHANGES)
                compact = xr.Dataset({name: (transfers[name].expand_dims(config=[member])
                                              if "config" not in transfers[name].dims else transfers[name])
                                      for name in keep})
                compact.update(summary)
                if self.hwp_target_events is not None:
                    compact["managed_area"] = (transfers.selected_area.sel(event=list(self.hwp_target_events))
                        .sum("source_slot").rename(event="managed_event"))
                if self.purpose_rank is not None:
                    compact.update(rank_source_bounds(transfers, rank_events, 1.))
                member_transfers.append(compact)
            if len(member_states) == 1:
                state, transfers = member_states[0], member_transfers[0]
            else:
                state = xr.concat(member_states, dim="config", data_vars="minimal", coords="minimal", compat="equals", join="exact")
                transfers = xr.concat(member_transfers, dim="config", data_vars="minimal", coords="minimal", compat="equals", join="exact")
            if self.secondary_agb_events is not None:
                secondary_rates = {name: _over_dt(transfers[name.removeprefix("rf_")], dt) for name in SEC_EXCHANGES}
            if self.hwp_target_events is not None:
                target_rates = {name: _over_dt(transfers["hwp_"+name.removeprefix("rf_HWP_")], dt) for name in HWP_EXCHANGES}
                target_rates["rf_HWP_managed_area"] = _over_dt(
                    transfers.managed_area.transpose("config", "reg_land", "managed_event"), dt)
            if self.purpose_rank is not None:
                rank_rates = {name: _over_dt(transfers[name], dt) for name in RANK_EXCHANGES}
            harvest_events = {name: _over_dt(transfers[name], dt) for name in HARVEST_EXCHANGES}
            product_input = transfers.hwp_added.sum(("event", "box_hwp")).values
            event_residual = transfers.budget_residual.values
            age_events = {name: _over_dt(transfers[name.removeprefix("rf_event_")], dt) for name in age_events}
        factors = None
        if self.age_production is not None:
            if len(self.age_production) != len(layout.regions):
                raise ValueError("one labeled age-production evaluator per region required")
            if self.age_clock is None:
                ages = np.maximum((t0+t1)/2-np.asarray(layout.births), 0.)
            else:
                ages = cohort_ages
            factors = np.empty((len(layout.members), len(layout.regions), 2,
                                len(layout.forest_classes), len(layout.births)))
            for ri, kernel in enumerate(self.age_production):
                if kernel.region_id != layout.regions[ri]:
                    raise ValueError("age-production region identity mismatch")
                factors[:, ri] = kernel.factors(ages, member_ids=layout.parameter_ids)[:, :, None, :]
            if self.age_clock is not None and self.age_clock.samples is not None:
                self.age_clock.samples.append((float(t0), float(t1), np.array(factors, copy=True)))
        if self.age_clock is None:
            next_state, flux = advance_carbon(state, layout, t0, t1, coefficients, factors)
        else:
            next_state, flux = advance_carbon(state, layout, t0, t1, coefficients, factors, cohort_ages=ages if self.age_production is not None else cohort_ages)
        nbp = flux.net_uptake.sum("bio_land") / dt
        # The provider returns the already defined reference sink anomaly,
        # weighted by current actual biome areas, never a second actual stock.
        reference = self.reference_sink(state, variables, parameters, forcing, t0, t1)
        reference, nbp = xr.align(reference, nbp, join="exact")
        baseline = parameters.rf_NBP_pi.transpose("config", "reg_land")
        reference, baseline = xr.align(reference, baseline, join="exact")
        fields = {
            "D_Fland": reference,
            "D_Eluc": reference - (nbp - baseline),
            "rf_NBP": nbp,
            "rf_NPP": _over_dt(flux.production, dt),
            "rf_Efire": _over_dt(flux.fire, dt),
            "rf_NBP_biome": _over_dt(flux.net_uptake, dt),
            "rf_Eresp": _over_dt(flux.cwd_respiration + flux.soil_respiration, dt),
            "rf_Ehwp": _over_dt(flux.hwp_respiration, dt),
            "rf_Ecwd": _over_dt(flux.cwd_respiration, dt),
            "rf_Esoil": _over_dt(flux.soil_respiration, dt),
            "rf_Ecrop": _over_dt(flux.crop_harvest, dt),
            "rf_Egraz": _over_dt(flux.grazing, dt),
            "rf_NPP_W": _over_dt(flux.forest_wood_production, dt),
            "rf_NPP_F": _over_dt(flux.forest_other_production, dt),
            "rf_forest_W_mean": _over_dt(flux.forest_wood_time_integral, dt),
            "rf_forest_F_mean": _over_dt(flux.forest_other_time_integral, dt),
            "rf_forest_area_mean": state.rf_A_primary + state.rf_A_forest.sum(("forest_class", "birth_time")),
            "rf_HWP_input": (("config", "reg_land"), product_input / dt),
            "rf_budget_residual": (("config",), (event_residual + flux.budget_residual.values) / dt),
            "rf_event_count": xr.DataArray(float(event) / dt),
            "rf_age_area_mean": area_by_age(state, layout, (t0+t1)/2) if self.age_clock is None else area_by_age(state, layout, (t0+t1)/2, cohort_ages=cohort_ages),
            "rf_age_NPP": _over_dt(flux.production_by_age, dt),
            **age_events,
            **harvest_events,
            **rank_rates,
            **target_rates,
            **secondary_rates,
        }
        # The host owns interval time. Scalar state/forcing timestamps must not
        # masquerade as coordinates of an interval-integrated exchange.
        exchanges = xr.Dataset({name: value.drop_vars("year", errors="ignore")
                                if isinstance(value, xr.DataArray) else value
                                for name, value in fields.items()})
        return next_state, exchanges


def build_regional_land_model(layout, coefficients, reference_sink, *, age_production=None,
                              preference=None, inputs=("event_area",), hwp_target_events=None, purpose_rank=None, hwp_target_rule="weighted", secondary_agb_events=None, harvest_projection=None, secondary_harvest_contract=CANDIDATE):
    """Build the native actual-land component; callers merge supported modules.

    coefficients(Var, Par, For, t0, t1) returns RegionalRates and must use the
    specified environmental convention. reference_sink returns (config,
    reg_land) PgC/yr under the declared PI baseline. Their input dependencies
    must be listed in inputs so native forcing/dependency checks remain active.
    Initialization is explicit; the first output flux row is zero because no
    interval has yet been integrated. Full fire-chemistry/reference adaptation
    is intentionally not inferred from variable names by this assembly helper.
    """
    validate_target_rule(layout, purpose_rank, hwp_target_events, hwp_target_rule)
    validate_secondary_contract(layout, secondary_agb_events, harvest_projection,
                                hwp_target_events, purpose_rank, hwp_target_rule, secondary_harvest_contract)
    model = Model("OSCAR_landC_age")
    if hwp_target_events is not None:
        inputs = tuple(dict.fromkeys(tuple(inputs) + ("HWP_target",)))
    if secondary_agb_events is not None:
        inputs = tuple(dict.fromkeys(tuple(inputs) + SEC_INPUTS))
    exchanges = (RANK_EXCHANGES if purpose_rank is not None else ()) + EXCHANGES + (HWP_EXCHANGES + ("rf_HWP_managed_area",) if hwp_target_events is not None else ())
    exchanges += SEC_EXCHANGES if secondary_agb_events is not None else ()
    template = layout.empty_state(0.)
    for name in STATE_NAMES:
        model.process(name, (name,) + tuple(inputs), Eq=lambda V, P, n=name: V[n],
                      units=template[name].attrs["units"],
                      core_dims=[d for d in template[name].dims if d != "config"])
    dependencies = tuple(STATE_NAMES) + tuple(inputs)
    def zero_flux(name, variables):
        if name in SEC_EXCHANGES:
            return initial_secondary_exchange(name, variables, secondary_agb_events)
        if name in RANK_EXCHANGES:
            return rank_zero(layout, hwp_target_events or tuple(purpose_rank))
        if name in HARVEST_EXCHANGES:
            return initial_harvest_exchange(name, layout)
        if name == "rf_HWP_managed_area":
            return xr.zeros_like(variables.rf_W_primary).expand_dims(managed_event=list(hwp_target_events), axis=-1)
        if name in HWP_EXCHANGES:
            return xr.zeros_like(variables.rf_W_primary)
        if name in AGE_EXCHANGES:
            return initial_age_exchange(name, variables, layout, float(variables.year))
        if name in ("rf_forest_W_mean", "rf_forest_F_mean"):
            pool = "W" if name == "rf_forest_W_mean" else "F"
            return variables["rf_"+pool+"_primary"] + variables["rf_"+pool+"_forest"].sum(("forest_class", "birth_time"))
        if name == "rf_forest_area_mean":
            return variables.rf_A_primary + variables.rf_A_forest.sum(("forest_class", "birth_time"))
        if name in ("D_Fland", "D_Eluc", "rf_NBP", "rf_HWP_input", "rf_NPP_W", "rf_NPP_F"):
            return 0. * variables.rf_W_primary
        if name in ("rf_NPP", "rf_Efire", "rf_NBP_biome", "rf_Eresp", "rf_Ehwp",
                    "rf_Ecwd", "rf_Esoil", "rf_Ecrop", "rf_Egraz"):
            return 0. * variables.rf_C_cwd
        if name == "rf_budget_residual":
            return 0. * variables.rf_W_primary.sum("reg_land")
        return xr.DataArray(0.)
    for name in exchanges:
        units = (SEC_UNITS[name] if name in SEC_EXCHANGES else "yr yr-1" if name in RANK_EXCHANGES else HARVEST_UNITS[name] if name in HARVEST_EXCHANGES else "Mha" if name in ("rf_forest_area_mean", "rf_age_area_mean") else
                 "Mha yr-1" if name in ("rf_event_area_by_age", "rf_HWP_managed_area") else "PgC" if name.endswith("_mean") else
                 "yr-1" if name == "rf_event_count" else "PgC yr-1")
        model.process(name, dependencies, Eq=lambda V, P, n=name: zero_flux(n, V), units=units)
    update = RegionalLandUpdate(layout, coefficients, reference_sink, age_production, preference, hwp_target_events, purpose_rank, hwp_target_rule, secondary_agb_events, harvest_projection, secondary_harvest_contract)
    model.register_land_step(LandStep(STATE_NAMES, exchanges, update))
    return model
