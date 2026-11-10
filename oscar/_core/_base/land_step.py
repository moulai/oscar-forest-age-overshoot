"""One registered joint land update inside the normal Model time loop."""
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class LandStep:
    """Immutable registration; ``step`` must not retain mutable run state.

    ``step(Var, Par, For, t0, t1, event)`` returns two Datasets: all owned
    next states, and all declared interval-mean exchanges. The event is at
    the beginning of the second half of a forcing year, exactly once.
    """
    states: Tuple[str, ...]
    exchanges: Tuple[str, ...]
    step: Callable

    def __post_init__(self):
        object.__setattr__(self, 'states', tuple(self.states))
        object.__setattr__(self, 'exchanges', tuple(self.exchanges))
        if not self.states or not callable(self.step):
            raise ValueError('LandStep requires states and a callable step')
        names = self.states + self.exchanges
        if len(set(names)) != len(names):
            raise ValueError('LandStep state and exchange names must be unique')

    @property
    def variables(self):
        return set(self.states + self.exchanges)

    def validate_model(self, model):
        if not set(self.states) <= model.var_prog | model.var_node:
            raise ValueError('LandStep states must be registered prognostic or node processes')
        if not set(self.exchanges) <= model.var_diag - model.var_node:
            raise ValueError('LandStep exchanges must be registered diagnostic processes')

    def validate_run(self, model, Ini, For, time_axis):
        self.validate_model(model)
        if Ini is None or not set(self.states) <= set(Ini):
            raise ValueError('LandStep requires explicit initial values for every owned state')
        if self.variables & set(For):
            raise ValueError('For cannot override LandStep states or interval exchanges')
        years = np.asarray(For[time_axis], dtype=float)
        if (years.ndim != 1 or not len(years) or not np.isfinite(years).all()
                or not np.equal(years, np.round(years)).all()
                or not np.equal(np.diff(years), 1).all()):
            raise ValueError('LandStep requires consecutive integer annual time labels')
        if (time_axis not in Ini.coords or Ini[time_axis].ndim != 0
                or float(Ini[time_axis]) != years[0]):
            raise ValueError('LandStep Ini must carry a scalar time matching the first forcing label')

    def prepare_interval(self, Var, Par, For, t0, t1, event):
        """Return one joint update, with prescribed inputs taking precedence."""
        inputs = Var.copy(deep=False)
        # The host usually just merged this forcing. Do not merge it again
        # when the arrays are already the ones on Var.
        if any(name not in inputs._variables or inputs[name].variable._data is not For[name].variable._data
               for name in For.data_vars):
            inputs.update(For)
        states, exchanges = self.step(inputs, Par, For, t0, t1, event)
        if not isinstance(states, xr.Dataset) or set(states) != set(self.states):
            raise ValueError('LandStep must return exactly all declared next states')
        if not isinstance(exchanges, xr.Dataset) or set(exchanges) != set(self.exchanges):
            raise ValueError('LandStep must return exactly all declared interval exchanges')
        for name in self.states:
            if states[name].dims != Var[name].dims:
                raise ValueError('LandStep cannot change state dimensions during a run: ' + name)
            xr.align(states[name], Var[name], join='exact', copy=False)
        return states, exchanges
