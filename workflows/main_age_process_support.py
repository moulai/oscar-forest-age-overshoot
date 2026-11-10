"""Passive actual-step age diagnostics and strict young-secondary eligibility.

This experiment adapter does not alter a shared OSCAR module. Event previews
are excluded. Secondary standing harvest retains area and birth in the kernel.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

EDGES = np.array([0., 5., 10., 20., 30., 40., 60., 100., np.inf])
AGE_GROUPS = ["0_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40",
              "40_to_60", "60_to_100", "100_plus", "primary_unaged", "nonforest_unaged"]
PROCESSES = ["conversion_clear", "standing_secondary_harvest", "other_removal"]
BIRTH_SOURCES = ["nonforest_establishment", "primary_harvest", "other_new_cohort"]


def strict_young_weights(layout, cutoff):
    """Hard eligibility, in actual years, for secondary clear exits and cuts."""
    from oscar._core._model._modules.land_geometry import secondary_stand_harvest
    rows = [i for i, e in enumerate(layout.events)
            if (e.source_class == "secdf" and
                ((e.kind == "conversion" and e.vegetation_action == "clear")
                 or secondary_stand_harvest(e)))]
    columns = [i for i, (c, _) in enumerate(layout.slots) if c == "secdf"]
    births = np.array([layout.slots[i][1] for i in columns])

    def preference(variables, parameters, forcing, time):
        result = np.ones((len(layout.regions), len(layout.events), len(layout.slots)))
        eligible = time - births >= float(cutoff)
        for row in rows:
            result[:, row, columns] = eligible
        return result
    return preference


class DetailedAgeObserver:
    """Observe the event actually routed and carbon step actually consumed."""
    def __init__(self, layout):
        self.layout = layout
        self.members = {v: i for i, v in enumerate(layout.members)}
        self.records = []
        self.originals = []
        ny, nm, nr, ng = 77, len(layout.members), len(layout.regions), len(AGE_GROUPS)
        self.flow_area = np.zeros((ny, nm, nr, 3, ng))
        self.flow_carbon = np.zeros((ny, nm, nr, 3, 2, ng))
        self.conversion_destination = np.zeros((ny, nm, nr, len(layout.classes), ng))
        self.birth = np.zeros((ny, nm, nr, 3))
        self.birth_by_destination = np.zeros((ny, nm, nr, len(layout.classes), 3))
        self.event_area = np.zeros((ny, nm, nr, len(layout.events)))

    def groups(self, layout, time):
        result = np.empty(len(layout.slots), dtype=int)
        classes = {c.class_id: c for c in layout.classes}
        for i, (name, birth) in enumerate(layout.slots):
            role = classes[name].role
            result[i] = (8 if role == "forest_unaged" else 9
                         if role != "forest_cohort" else
                         min(7, int(np.searchsorted(EDGES[1:], max(0., time - birth), side="right"))))
        return result

    def install(self):
        from oscar._core._model._modules import regional_forest_age_native as native
        from oscar._core._model._modules import regional_forest_secondary_carbon as secondary
        from oscar._core._model._modules.land_geometry import secondary_stand_harvest
        original_advance = native.advance_carbon

        def advance(state, layout, left, right, rates, factors=None, cohort_ages=None):
            nm, nr, ng = len(layout.members), len(layout.regions), len(AGE_GROUPS)
            area = np.zeros((nm, nr, ng))
            stock = np.zeros((nm, nr, 2, ng))
            actual = np.zeros_like(stock)
            direct = np.zeros_like(stock)
            births = np.asarray(layout.births)
            ages = np.maximum((left + right)/2 - births, 0.)
            band = np.minimum(7, np.searchsorted(EDGES[1:], ages, side="right"))
            for g in range(8):
                mask = band == g
                a = state.rf_A_forest.values[..., mask]
                area[..., g] = a.sum(axis=(2, 3))
                for pool, name in enumerate(("rf_W_forest", "rf_F_forest")):
                    stock[:, :, pool, g] = state[name].values[..., mask].sum(axis=(2, 3))
                    factor = 1. if factors is None else factors[:, :, pool, :, :][..., mask]
                    q = rates.production[:, :, pool]
                    actual[:, :, pool, g] = q * (a * factor).sum(axis=(2, 3))
                    direct[:, :, pool, g] = q * (a * (factor - 1.)).sum(axis=(2, 3))
            area[..., 8] = state.rf_A_primary.values
            for pool, name in enumerate(("rf_W_primary", "rf_F_primary")):
                stock[:, :, pool, 8] = state[name].values
                actual[:, :, pool, 8] = rates.production[:, :, pool] * state.rf_A_primary.values
            area[..., 9] = state.rf_A_nonforest.values.sum(axis=2)
            self.records.append(dict(left=float(left), right=float(right), area=area,
                                     stock=stock, actual=actual, direct=direct))
            return original_advance(state, layout, left, right, rates, factors,
                **({} if cohort_ages is None else {"cohort_ages": cohort_ages}))
        self.originals.append((native, "advance_carbon", original_advance))
        native.advance_carbon = advance

        def wrap_event(original):
            def event(state, layout, time, *args, **kwargs):
                result = original(state, layout, time, *args, **kwargs)
                if kwargs.get("selection_only", False):
                    return result
                _, ledger = result
                mi = self.members[layout.members[0]]
                yi = int(np.ceil(time)) - 2023
                groups = self.groups(layout, time)
                area = ledger.source_area.values
                selected = ledger.selected_area.values
                classes = {c.class_id: c for c in layout.classes}
                class_index = {c.class_id: i for i, c in enumerate(layout.classes)}
                density = np.stack([np.divide(ledger[name].values[0], area,
                    out=np.zeros_like(area), where=area > 0)
                    for name in ("source_wood", "source_other")], axis=1)
                for ei, spec in enumerate(layout.events):
                    take = selected[:, ei]
                    if not np.any(take):
                        continue
                    self.event_area[yi, mi, :, ei] += take.sum(axis=1)
                    standing = secondary_stand_harvest(spec)
                    process = (1 if standing else 0 if spec.kind == "conversion"
                               and spec.vegetation_action == "clear" else 2)
                    for gi in np.unique(groups):
                        mask = groups == gi
                        self.flow_area[yi, mi, :, process, gi] += take[:, mask].sum(axis=1)
                        self.flow_carbon[yi, mi, :, process, :, gi] += (
                            take[:, None, mask] * density[:, :, mask]).sum(axis=-1)
                        if process == 0:
                            self.conversion_destination[yi, mi, :, class_index[spec.target_class], gi] += take[:, mask].sum(axis=1)
                    # Standing secondary harvest is never a new birth. Retained
                    # conversions can establish new forest; primary cuts do too.
                    new = classes[spec.target_class].role == "forest_cohort" and not standing
                    if new:
                        source_role = classes[spec.source_class].role
                        bg = (1 if spec.kind == "harvest" and source_role == "forest_unaged"
                              else 0 if source_role not in ("forest_unaged", "forest_cohort") else 2)
                        amount = take.sum(axis=1)
                        self.birth[yi, mi, :, bg] += amount
                        self.birth_by_destination[yi, mi, :, class_index[spec.target_class], bg] += amount
                return result
            return event
        # The secondary request module calls its own imported event alias for
        # preview and final routing; the native alias covers other contracts.
        for module in (native, secondary):
            old = module.apply_annual_events
            self.originals.append((module, "apply_annual_events", old))
            module.apply_annual_events = wrap_event(old)

    def restore(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)

    def datasets(self):
        count = len(self.records)
        if not count or count > 154 or count % 2:
            raise RuntimeError(f"Incomplete actually consumed intervals: {count}")
        years = count // 2
        coords = dict(config=list(self.layout.members), source_config=("config", list(self.layout.parameter_ids)),
                      reg_land=list(self.layout.regions), detailed_age_group=AGE_GROUPS, tissue=["W", "F"])
        native = xr.Dataset(coords={**coords, "step": np.arange(count)})
        native["left"] = ("step", [r["left"] for r in self.records])
        native["right"] = ("step", [r["right"] for r in self.records])
        native["age_area_Mha"] = (("step", "config", "reg_land", "detailed_age_group"), np.stack([r["area"] for r in self.records]))
        for field, key, unit in [("age_live_carbon_PgC", "stock", "PgC"),
                                 ("age_actual_NPP_PgC_yr", "actual", "PgC yr-1"),
                                 ("age_direct_NPP_PgC_yr", "direct", "PgC yr-1")]:
            native[field] = (("step", "config", "reg_land", "tissue", "detailed_age_group"), np.stack([r[key] for r in self.records]))
            native[field].attrs["units"] = unit
        native.age_area_Mha.attrs["units"] = "Mha"
        annual = xr.Dataset(coords={**coords, "year": np.arange(2023, 2023+years), "event_process": PROCESSES,
            "birth_source": BIRTH_SOURCES, "destination_class": [c.class_id for c in self.layout.classes],
            "event": [e.event_id for e in self.layout.events]})
        dt = native.right.values - native.left.values
        for name in ("age_area_Mha", "age_live_carbon_PgC", "age_actual_NPP_PgC_yr", "age_direct_NPP_PgC_yr"):
            v = native[name]
            weighted = v.values * dt.reshape((-1,) + (1,) * (v.ndim - 1))
            annual[name] = (("year",) + v.dims[1:], weighted.reshape((years, 2) + weighted.shape[1:]).sum(axis=1))
            annual[name].attrs.update(v.attrs)
        annual["event_selected_area_Mha"] = (("year", "config", "reg_land", "event"), self.event_area[:years])
        annual["event_source_age_area_Mha"] = (("year", "config", "reg_land", "event_process", "detailed_age_group"), self.flow_area[:years])
        annual["event_source_age_carbon_PgC"] = (("year", "config", "reg_land", "event_process", "tissue", "detailed_age_group"), self.flow_carbon[:years])
        annual["conversion_destination_age_area_Mha"] = (("year", "config", "reg_land", "destination_class", "detailed_age_group"), self.conversion_destination[:years])
        annual["true_birth_area_Mha"] = (("year", "config", "reg_land", "birth_source"), self.birth[:years])
        annual["true_birth_destination_area_Mha"] = (("year", "config", "reg_land", "destination_class", "birth_source"), self.birth_by_destination[:years])
        attrs = dict(schema="actual-age-process-diagnostics-v1", sampling="Values used by actual native steps; event previews excluded",
            age_intervals="[0,5),[5,10),[10,20),[20,30),[30,40),[40,60),[60,100),[100,infinity); primary unaged",
            production_definition="actual=q*A*g; direct=q*A*(g-1), W/F separately; not net uptake or a climate contribution",
            standing_harvest="Secondary standing cuts remove carbon but retain birth and area; not true births",
            nonforest_note="Nonforest area is recorded but its one-pool production is not partitioned into W/F",
            time_support="Native production uses interval midpoint age; event age uses actual event time; annual duration-weighted means and annual event amounts")
        native.attrs.update(attrs); annual.attrs.update(attrs)
        return native, annual
