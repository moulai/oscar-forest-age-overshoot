"""Young-parcel cancellation with explicit nonforest land-service rationing."""
from __future__ import annotations
import numpy as np
import xarray as xr
from workflows.nontransferred_young_protection import NoTransferProtection


class SupplyLimitedProtection(NoTransferProtection):
    def __init__(self, layout, cutoff):
        super().__init__(layout,cutoff)
        self.nonforest=[c.class_id for c in layout.classes if c.role=="nonforest"]
        self.nf_events=[i for i,e in enumerate(layout.events) if e.source_class in self.nonforest]
        shape=(77,len(layout.members),len(layout.regions))
        self.supply=np.zeros(shape+(len(self.nf_events),3))
        self.opening=np.zeros(shape+(len(self.nonforest),))
        self.alpha=np.ones_like(self.opening)

    def install(self):
        from oscar._core._model._modules import regional_forest_age_native as native
        super().install()
        original=native.apply_secondary_agb_request_events

        def supplied(state,layout,time,event_area,*args,**kwargs):
            raw=np.asarray(event_area,dtype=float)
            actual_request=raw.copy()
            yi,mi=int(np.ceil(time))-2023,self.members[layout.members[0]]
            for ci,c in enumerate(self.nonforest):
                rows=[i for i,e in enumerate(layout.events) if e.source_class==c]
                area=state.rf_A_nonforest.sel(nonforest_class=c).values[0]
                total=raw[:,rows].sum(axis=-1)
                fraction=np.minimum(1.,np.divide(area,total,out=np.ones_like(area),where=total>0))
                actual_request[:,rows]*=fraction[:,None]
                self.opening[yi,mi,:,ci]=area
                self.alpha[yi,mi,:,ci]=fraction
            out,ledger=original(state,layout,time,actual_request,*args,**kwargs)
            self.supply[yi,mi,:,:,0]=raw[:,self.nf_events]
            self.supply[yi,mi,:,:,1]=ledger.executed_area.values[:,self.nf_events]
            self.supply[yi,mi,:,:,2]=raw[:,self.nf_events]-ledger.executed_area.values[:,self.nf_events]
            ledger.attrs["nonforest_service_rule"]="One opening-area fraction for all original same-source conversions and harvest; no incoming reuse or replacement"
            return out,ledger

        self.originals.append((native,"apply_secondary_agb_request_events",original))
        native.apply_secondary_agb_request_events=supplied

    def dataset(self):
        d=super().dataset()
        d=d.assign_coords(nonforest_class=self.nonforest,
            nonforest_event=[self.layout.events[i].event_id for i in self.nf_events],
            supply_accounting=["original_request","actual","unmet"])
        d["nonforest_activity_area_Mha"] = (("year","config","reg_land","nonforest_event","supply_accounting"),self.supply)
        d["nonforest_source_opening_area_Mha"] = (("year","config","reg_land","nonforest_class"),self.opening)
        d["nonforest_source_supply_fraction"] = (("year","config","reg_land","nonforest_class"),self.alpha)
        for n,attr in (("nonforest_event_source","source_class"),("nonforest_event_destination","target_class"),("nonforest_event_kind","kind")):
            d[n]=("nonforest_event",[getattr(self.layout.events[i],attr) for i in self.nf_events])
        d.nonforest_activity_area_Mha.attrs.update(units="Mha",time_support="Gross event total in [year-1,year]; original requests are retained",definition="All same-source original nonforest gross events, including forest establishment, share the same supply fraction")
        d.nonforest_source_opening_area_Mha.attrs.update(units="Mha",time_support="Opening event state at year-0.5 before any routing; current-year incoming cannot supply requests")
        d.nonforest_source_supply_fraction.attrs.update(units="1",definition="min(1,opening_area/sum_original_same_source_requests); conversions and that source's harvest pooled")
        d.attrs.update(schema="land-supply-limited-young-secondary-protection-v1",
            nonforest_services="Explicit same-source proportional limitation; includes reduced nonforest-to-forest establishment; no compensating land or activity",
            comparison="A management scenario with changed actual land and wood services, not an age-only or equal-land-service effect")
        return d
