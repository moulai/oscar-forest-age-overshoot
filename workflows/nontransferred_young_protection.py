"""Cancel protected proposals without replacement in a named experiment.

The native optional post-selection ``cancel_mask`` interface is used only here.
Carbon routing and the host's actual-area feedback remain native.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


class NoTransferProtection:
    def __init__(self, layout, cutoff):
        if float(cutoff) not in (10., 20., 30., 40.):
            raise ValueError("Only the four prespecified protection ages are supported")
        self.layout, self.cutoff = layout, float(cutoff)
        self.originals = []
        self.members = {member: i for i, member in enumerate(layout.members)}
        self.destinations = [c.class_id for c in layout.classes]
        shape = (77, len(layout.members), len(layout.regions))
        self.conversion = np.zeros(shape + (len(self.destinations), 3))
        self.standing = np.zeros(shape + (3,))

    def protected(self, layout, time):
        return np.array([c == "secdf" and max(0., time - birth) < self.cutoff
                         for c, birth in layout.slots], dtype=bool)

    def conversion_mask(self, layout, time):
        rows = np.array([e.source_class == "secdf" and e.kind == "conversion"
                         for e in layout.events], dtype=bool)
        return np.broadcast_to(rows[None, :, None] * self.protected(layout, time)[None, None, :],
                               (len(layout.regions), len(layout.events), len(layout.slots)))

    def install(self):
        from oscar._core._model._modules import regional_forest_age_native as native
        from oscar._core._model._modules import regional_forest_secondary_carbon as secondary
        original_area = secondary.apply_annual_events
        original_secondary = native.apply_secondary_agb_request_events

        def area_events(state, layout, time, *args, **kwargs):
            return original_area(state, layout, time, *args,
                                 cancel_mask=self.conversion_mask(layout, time), **kwargs)

        def secondary_events(state, layout, time, event_area, weights, pwood, phwp,
                             target_q, label_q, k_f_ag, *, secondary_events,
                             projection, k_f_ag_leaf_only,
                             secondary_harvest_contract=secondary.CANDIDATE,
                             cohort_ages=None):
            if secondary_harvest_contract != secondary.CANDIDATE or cohort_ages is not None:
                raise ValueError("This experiment uses the existing proportional request and true ages")
            from oscar._core._model._modules.regional_forest_management import _managed_rows
            rows, source = _managed_rows(layout, secondary_events)
            columns = [i for i, (c, _) in enumerate(layout.slots) if c == source]
            common = np.array(event_area, dtype=float, copy=True)
            common[:, rows] = 0.
            # Form both original management proposals on the current state
            # before deleting any protected parcel. In the original annual
            # sequence, conversion proposals reserve their opening stocks
            # before the Q-driven standing proposal is formed.
            opening, wood, other, selected_common = original_area(
                state, layout, time, common, weights, pwood, phwp, selection_only=True)
            eligible = np.asarray(weights[:, rows[0], columns]) > 0
            if not np.array_equal(eligible, np.asarray(weights[:, rows[1], columns]) > 0):
                raise ValueError("Secondary labels have different original eligibility")
            available = np.maximum(opening - selected_common.sum(axis=1), 0.)[:, columns] * eligible
            available[:, np.array([layout.slots[i][1] >= time for i in columns])] = 0.
            density = np.divide(wood[0][:, columns] + np.asarray(k_f_ag)[0, :, None]*other[0][:, columns],
                                opening[:, columns], out=np.zeros_like(available),
                                where=opening[:, columns] > 0)
            target = np.asarray(target_q, dtype=float)
            capacity = (available*density).sum(axis=-1)
            fraction = np.minimum(1., np.divide(target, capacity, out=np.zeros_like(target), where=capacity > 0))
            proposal = available*fraction[:, None]
            selected = proposal.copy()
            protected = self.protected(layout, time)[columns]
            selected[:, protected] = 0.
            reduced_target = (selected*density).sum(axis=-1)
            labels = np.asarray(label_q, dtype=float)
            shares = np.divide(labels, target[:, None], out=np.zeros_like(labels), where=target[:, None] > 0)
            final_weights = np.array(weights, dtype=float, copy=True)
            for row in rows:
                final_weights[:, row, columns] *= ~protected
            # Native routing is performed once. Reducing Q to exactly the
            # retained original proposal and forbidding the cancelled parcels
            # gives the same selected fractions on old parcels: no refill.
            out, ledger = original_secondary(
                state, layout, time, event_area, final_weights, pwood, phwp,
                reduced_target, shares*reduced_target[:, None], k_f_ag,
                secondary_events=secondary_events, projection=projection,
                k_f_ag_leaf_only=k_f_ag_leaf_only,
                secondary_harvest_contract=secondary_harvest_contract)
            # Preserve original external demand; achieved delivery and all W/F,
            # product and dead-stock transfers are the actual native result.
            ledger.sec_agb_request.values[0, :] = target
            achieved = ledger.sec_agb_achieved.values[0]
            ledger.sec_agb_shortfall.values[0, :] = np.maximum(target-achieved, 0.)
            ledger.secondary_label_request.values[:] = labels
            ledger.secondary_label_share.values[:] = shares
            ledger.attrs["protection_rule"] = "Original current-state proposal, cancel age<cutoff, never refill"
            ledger.attrs["protection_cutoff_years"] = self.cutoff
            yi, mi = int(np.ceil(time))-2023, self.members[layout.members[0]]
            original = ledger.proposed_selected_area.values
            actual = ledger.selected_area.values
            for ei, event in enumerate(layout.events):
                if event.source_class == "secdf" and event.kind == "conversion":
                    di = self.destinations.index(event.target_class)
                    before, after = original[:, ei].sum(axis=-1), actual[:, ei].sum(axis=-1)
                    self.conversion[yi, mi, :, di, 0] += before
                    self.conversion[yi, mi, :, di, 1] += after
                    self.conversion[yi, mi, :, di, 2] += before-after
            self.standing[yi, mi, :, 0] += proposal.sum(axis=-1)
            self.standing[yi, mi, :, 1] += ledger.sec_agb_area.values[0]
            self.standing[yi, mi, :, 2] += proposal.sum(axis=-1)-ledger.sec_agb_area.values[0]
            return out, ledger

        for obj, name, replacement in ((native, "apply_annual_events", area_events),
                                        (secondary, "apply_annual_events", area_events),
                                        (native, "apply_secondary_agb_request_events", secondary_events)):
            self.originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, replacement)

    def restore(self):
        for obj, name, value in reversed(self.originals):
            setattr(obj, name, value)

    def dataset(self):
        coords = dict(year=np.arange(2023, 2100), config=list(self.layout.members),
                      source_config=("config", list(self.layout.parameter_ids)),
                      reg_land=list(self.layout.regions), destination_class=self.destinations,
                      activity_support=["original_proposal", "actual", "cancelled"])
        d = xr.Dataset(coords=coords)
        d["secondary_conversion_area_Mha"] = (("year", "config", "reg_land", "destination_class", "activity_support"), self.conversion)
        d["standing_harvest_footprint_Mha"] = (("year", "config", "reg_land", "activity_support"), self.standing)
        for name in d.data_vars:
            d[name].attrs.update(units="Mha", time_support="Gross event total over native annual interval [year-1,year]; not a time mean")
        d.attrs.update(schema="nontransferred-young-secondary-protection-v1", cutoff_years=self.cutoff,
                       rule="Original current-state parcels first, cancel protected portion, never refill",
                       forest_support="Aged secondary forest secdf only; primary has no invented age",
                       standing="Actual footprint; standing harvest preserves area and birth",
                       land_services="Cancelled conversions retain forest and reduce destination land entry")
        return d
