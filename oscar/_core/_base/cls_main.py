import warnings
import numpy as np
import xarray as xr

from time import perf_counter

from oscar._core._base.fct_solve import scheme_ex, scheme_imex, adapt_nt
from oscar._core._base.land_step import LandStep


def _same_payload(left, right):
    """True when both objects already wrap one ndarray.

    Dataset item access builds a new DataArray each time, so `is` never hits.
    Shared memory means this forcing value is already on the caller dataset.
    """
    if left is right:
        return True
    try:
        return left.variable._data is right.variable._data
    except AttributeError:
        return False


def _external_land_consumers(model, land_step):
    """Non-land prognostic and node variables. They read the pre-update state."""
    return [var for var in list(model.var_prog) + list(model.var_node) if var not in land_step.states]


def _diagnostics_read_by(model, consumers, blocked, ordered):
    """Causal diagnostics actually read by consumers, excluding land-owned names."""
    pending = []
    for var in consumers:
        proc = model[var]
        pending.extend(proc.In)
        pending.extend(proc.In2)
    needed = set()
    while pending:
        var = pending.pop()
        if var in needed or var in blocked or var not in model._processes:
            continue
        proc = model[var]
        if proc.param or proc.prog:
            continue
        needed.add(var)
        pending.extend(proc.In)
        pending.extend(proc.In2)
    return [var for var in ordered if var in needed]


def _store_diagnostics(model, host, names, parameters, forcing):
    """Write equations in waves so one merge stores each independent set.

    Dataset item assignment realigns every array already stored. Names stay in
    causal order. A name that reads a pending result is evaluated only after
    that wave is stored.
    """
    pending = {}
    for name in names:
        proc = model[name]
        if pending and (set(proc.In) | set(proc.In2)) & pending.keys():
            host.update(pending)
            pending = {}
        pending[name] = proc(host, parameters, forcing)
    if pending:
        host.update(pending)




##################################################
##   1. MODELS
##################################################

class Model():
    '''
    Class defining a Model, i.e. a collection of linked Process objects.

    Init:
    ------
    (nothing)

    Options:
    --------
    name (str)      name of the model;
                    default = ''
    '''

    ## ------
    ## Basics
    ## ------

    ## initialization
    def __init__(self, name=''):
        assert type(name) is str
        self.name = name
        self._processes = {}
        self._land_step = None

    ## check if process in model
    def __contains__(self, item):
        assert type(item) == str
        return item in self.proc_all
    
    ## nice display
    def __repr__(self):
        out = '<{}: {} (Processes: {})>\n'.format(str(type(self)).replace("<class '", "").replace("'>", ""), self.name, len(self))
        for proc in self._processes.values():
            out += '{}\n'.format(proc.__repr__().split('\n')[0])
        return out

    ## number of processes
    def __len__(self): return len(self._processes)


    ## ----------
    ## Properties
    ## ----------

    ## lists of variables
    @property
    def var_param(self): return set([proc.Out for proc in self._processes.values() if proc.param])
    @property
    def var_all(self): return set([proc.Out for proc in self._processes.values()]) | set([var for proc in self._processes.values() for var in proc.In + proc.In2])
    @property
    def var_mid(self): return set([proc.Out for proc in self._processes.values()]) & set([var for proc in self._processes.values() for var in proc.In + proc.In2])
    @property
    def var_out(self): return set([proc.Out for proc in self._processes.values()]) - self.var_mid - self.var_param
    @property
    def var_in(self): return set([var for proc in self._processes.values() for var in proc.In]) - self.var_mid
    @property
    def var_in2(self): return set([var for proc in self._processes.values() for var in proc.In2]) - self.var_mid
    @property
    def var_prog(self): return set([proc.Out for proc in self._processes.values() if proc.prog])
    @property
    def var_diag(self): return set([proc.Out for proc in self._processes.values() if not proc.prog and not proc.param])
    @property
    def var_node(self): return set([var for var in self.var_diag if self._processes[var].node])
    @property
    def proc_all(self): return list(self._processes.keys())

    ## get causality tree levels
    def proc_levels(self, test_node=[]):
        ## initialize level 0 with state variables
        params, progs, nodes = self.var_param, self.var_prog, self.var_node
        levels = {-1: list(params), 0:list(progs) + list(nodes) + test_node}
        proc_remain = set(self.proc_all) - progs - nodes - params - set(test_node)
        # Earlier levels are fixed during this pass. Reuse their membership
        # instead of rescanning the complete graph for every input variable.
        available = self.var_in | set(levels[-1]) | set(levels[0])
        ## loop through levels
        while proc_remain != set():
            next_level = []
            for proc in proc_remain:
                ## check whether solvable with lower-level variables
                if all(var in available for var in self._processes[proc].In):
                    next_level.append(proc)
            levels[max(levels.keys())+1] = next_level
            proc_remain = proc_remain - set(next_level)
            available.update(next_level)
            ## break if no variables in this level
            if next_level == []:
                levels[np.inf] = list(proc_remain)
                break
        ## return dic of variables by level
        return levels


    ## ----------
    ## Operations
    ## ----------

    ## copy the model to another one
    def copy(self, add_name='_copy', new_name=None, only=None):
        ## create and name new model
        if new_name is None: new_model = Model(self.name + add_name)
        else: new_model = Model(new_name)
        ## copy processes
        for proc in self._processes.values():
            if only is None or proc.Out in only:
                new_model.process(proc.Out, proc.In, proc.In2, proc.Eq, proc.DiffEq, proc.vLin, units=proc.units, core_dims=proc.core_dims)
        if self._land_step is not None:
            selected = self._land_step.variables & set(new_model.proc_all)
            if selected and selected != self._land_step.variables:
                raise ValueError('copy cannot split a registered LandStep')
            if selected:
                new_model.register_land_step(self._land_step)
        ## return new model
        return new_model
    
    ## merge with another model
    def merge(self, other, new_name=None, overwrite=False):
        assert type(other) is Model
        if self._land_step is not None and other._land_step is not None:
            raise ValueError('merge cannot combine two registered LandSteps')
        if other._land_step is not None and not overwrite:
            if other._land_step.variables & set(self.proc_all):
                raise ValueError('merge would discard part of the incoming LandStep; use overwrite')
        ## create and name new model
        if new_name is None and not overwrite: new_model = self.copy(new_name=self.name + ' +> ' + other.name)
        elif new_name is None and overwrite: new_model = self.copy(new_name=self.name + ' <+ ' + other.name)
        else: new_model = self.copy(new_name=new_name)
        ## add processes
        for proc in other._processes.values():
            if proc.Out not in new_model or overwrite:
                new_model.process(proc.Out, proc.In, proc.In2, proc.Eq, proc.DiffEq, proc.vLin, units=proc.units, core_dims=proc.core_dims)
        if other._land_step is not None:
            new_model.register_land_step(other._land_step)
        ## return new model
        return new_model


    ## --------
    ## Defining
    ## --------

    ## define process
    def process(self, Out, *args, **kwargs):
        if self._land_step is not None and Out in self._land_step.variables:
            raise ValueError('cannot replace a process owned by a registered LandStep: ' + Out)
        self._processes[Out] = Process(Out, *args, model=self, **kwargs)

    def register_land_step(self, land_step):
        """Register the single joint land update after defining its processes."""
        if not isinstance(land_step, LandStep) or self._land_step is not None:
            raise ValueError('a Model accepts exactly one LandStep registration')
        land_step.validate_model(self)
        self._land_step = land_step

    ## get process
    def __getitem__(self, key):
        return self._processes[key]

    ## set process
    def __setitem__(self, key, val):
        assert type(key) == str and type(val) == Process
        if self._land_step is not None and key in self._land_step.variables:
            raise ValueError('cannot replace a process owned by a registered LandStep: ' + key)
        if key == val.Out: self._processes[key] = val
        else: raise KeyError('key ({0}) and process name ({1}) must be the same'.format(key, val.Out))

    ## delete process
    def __delitem__(self, key):
        if self._land_step is not None and key in self._land_step.variables:
            raise ValueError('cannot delete a process owned by a registered LandStep: ' + key)
        del self._processes[key]


    ## --------
    ## Plotting
    ## --------

    ## display a graph of the model
    def display(self, random=True):
        ## check dependencies
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except:
            raise ImportError("displaying the model requires 'matplotlib' and 'networkx' libraries installed")
        ## create graph
        DG = nx.DiGraph()
        for proc in self._processes.values():
            DG.add_edges_from([(var, proc.Out) for var in proc.In])
        ## plot
        plt.figure()
        if random: layout = nx.kamada_kawai_layout(DG, pos=nx.random_layout(DG))
        else: layout = nx.kamada_kawai_layout(DG) # nx.spring_layout(DG, k=0.8, iterations=500)
        nx.draw(DG, pos=layout, with_labels=True, alpha=0.8, font_size=8, node_size=2, edge_color='0.8')
        plt.title(self.name, fontsize='small')
        plt.show()


    ## -----------
    ## Pre-Running
    ## -----------

    ## check requested variables are in the model
    def _check_var_keep(self, var_keep=[]):
        for var in var_keep:
            if var not in self:
                raise NameError("'{0}' given in 'var_keep' is not a variable of the {1} Model".format(var, self.name))

    ## check solvability by iteration (i.e. no loops in diagnostic variables)
    def _check_solvable(self, var_node=[]):
        assert all(var in self.var_all for var in var_node)
        levels = self.proc_levels(var_node)
        if np.inf in levels.keys(): 
            raise RuntimeError("infinite loop to solve diagnostic variables! turn at least one of the following into a node variable: {0}".format([var.replace("'", "") for var in levels[np.inf]]))

    ## check initialized variables
    def _check_Ini(self, Ini):
        var_miss = (self.var_prog) - set(Ini.keys())
        if var_miss != set(): raise RuntimeError('missing initialisation value for variables: {0}'.format(str(var_miss).replace("'", "")))

    ## check forcing variables and time axis
    def _check_For(self, For, time_axis='year'):
        var_miss = self.var_in - set(For.keys())
        if var_miss != set(): raise RuntimeError("missing forcing variables: {0}".format(str(var_miss).replace("'", "")))
        if time_axis not in For.coords: raise RuntimeError("forcing dataset has no time axis: '{0}'".format(time_axis))

    ## get zeroed initial conditions
    def _get_Ini(self, Par, For):
        Ini = xr.Dataset()
        for var in list(self.var_prog) + list(self.var_node):
            if len(self[var].core_dims) == 0: 
                Ini[var] = xr.DataArray(0.)
            elif all([dim in set(Par.dims) | set(For.dims) for dim in self[var].core_dims]): 
                Ini[var] = sum([xr.zeros_like(Par[dim], dtype=float) if dim in Par.coords else xr.zeros_like(For[dim], dtype=float) for dim in self[var].core_dims])
            else:
                raise RuntimeError("cannot auto-create initial conditions")
        return Ini

    ## move forcings without time axis to parameters
    ## this overrides those provided in parameter
    def _move_For_to_Par(self, Par, For, time_axis='year'):
        Par_ = xr.merge([For[var].to_dataset() for var in For if time_axis not in For[var].dims] + [Par], join='outer', compat='override')
        For_ = For.drop_vars([var for var in For if time_axis not in For[var].dims])
        return Par_, For_

    ## get secondary parameters (provided as processes)
    def _get_Par2(self, Par, For=None, time_axis='year'):
        if For is not None: Par = self._move_For_to_Par(Par, For, time_axis=time_axis)[0]
        Par2 = xr.Dataset()
        var_list = list(self.var_param)
        while len(var_list) > 0:
            var_list_copy = var_list.copy()
            for var in var_list:
                if var in Par: out = Par[var]
                else: out = self[var].Eq(xr.merge([Par, Par2], join='outer', compat='no_conflicts'))
                if out is not None: 
                    Par2[var] = out
                    Par2[var].attrs['units'] = self[var].units
                    var_list.remove(var)
            if var_list == var_list_copy:
                raise RuntimeError("some secondary parameters cannot be computed; remove reciprocal dependency between: {0}".format(str(var_list).replace("'", "")))
        return Par2

    ## get linear speeds of differential system
    def _get_vLin(self, Par):
        return xr.Dataset({var: self[var].vLin(Par) for var in self.var_prog})


    ## -------
    ## Running
    ## -------

    ## running model
    def __call__(self, Ini, Par, For, dtype=np.float32, var_keep=[], keep_prog=True, get_final=False, time_axis='year', scheme='imex', nt=2, adapt_nt=adapt_nt, no_warnings=True):
        '''
        Input:
        ------
        Ini (xr.Dataset)        initial conditions (set to None for automatic nil values)
        Par (xr.Dataset)        parameters
        For (xr.Dataset)        forcing data

        Output:
        ------
        Var_out (xr.Dataset)    model outputs
        Var_fin (xr.Dataset)    final end-year values of state variables (if get_final is True)

        Options:
        --------
        dtype (type)            data type for computation, either np.float32 or float;
                                default = np.float32
        var_keep (list)         variables to be kept as output;
                                default = []
        keep_prog (bool)        whether prognostic and node variables should be kept as output;
                                default = True
        get_final (bool)        whether final end-year values of state variables should be kept;
                                this is necessary to initiate subsequent runs;
                                default = False
        time_axis (str)         name of the time dimension;
                                default = 'year'
        scheme (str)            solving scheme for the differential system (between 'imex' and 'ExpInt');
                                default = 'imex'
        nt (int)                number of substeps during the solving of the differential system;
                                default = 2
        adapt_nt (callable)     what heuristic rule to use to dynamically change nt;
                                the 'nt' argument is then used as a minimum number of substeps;
                                change to None or False to turn off;
                                default = default function
        no_warnings (bool)      whether warnings should be hidden during core calculations;
                                default = True
        '''

        ## load data in memory
        if Ini is not None: Ini = Ini.load()
        Par = Par.load()
        For = For.load()

        ## force data type
        if dtype is not None:
            if Ini is not None: Ini = Ini.astype(dtype)
            Par = Par.astype(dtype)
            For = For.astype(dtype)

        ## various checks and sorting
        self._check_var_keep(var_keep)
        self._check_solvable()
        self._check_For(For, time_axis)
        land_step = self._land_step
        if land_step is not None:
            land_step.validate_run(self, Ini, For, time_axis)
            if int(nt) != nt or nt < 1:
                raise ValueError('LandStep requires a positive integer nt')
        Par, For = self._move_For_to_Par(Par, For, time_axis)

        ## get secondary parameters
        ## but calculated ones are overridden if provided
        Par = xr.merge([self._get_Par2(Par), Par], join='outer', compat='override')

        ## get initial state
        if Ini is None: Ini = self._get_Ini(Par, For)
        else: self._check_Ini(Ini)

        ## get time axis and substeps
        time = For.coords[time_axis]
        steps = (0*time + nt).astype(int)

        ## get levels in causality tree, excluding parameters and non-kept outputs
        levels = self.proc_levels()
        required_outputs = set(var_keep) | (set(land_step.exchanges) if land_step is not None else set())
        levels = {lvl: list(set(levels[lvl]) - self.var_param - (self.var_out - required_outputs)) for lvl in levels.keys()}
        levels = {lvl: levels[lvl] for lvl in levels if len(levels[lvl]) > 0}

        ## lists of variables to calculate (ordered)
        list_var_prog = list(self.var_prog)
        list_var_node = list(self.var_node)
        list_var_diag = [var for lvl in np.sort(list(levels.keys())) for var in levels[lvl] if lvl > 0]
        list_var_keep = (list_var_prog + list_var_node) * keep_prog + var_keep
        # Land exchanges are recalculated by the joint step. Diagnostics that
        # read them are refreshed afterwards. The pre-step pass only needs the
        # environment chain, which does not read those exchanges.
        if land_step is not None:
            exchange_names = set(land_step.exchanges)
            # Prognostic and node values belong to the pre-update state here.
            # Refresh only declared land-input diagnostics; do not traverse a
            # node's next equation or compute unused non-land diagnostics.
            pre_land_diags = _diagnostics_read_by(
                self, list(land_step.states + land_step.exchanges),
                set(land_step.states) | exchange_names | self.var_node,
                list_var_diag)
            external = _external_land_consumers(self, land_step)
            # Kept outputs are rebuilt after the step. Before non-land equations,
            # refresh only the diagnostics those equations actually read.
            middle_diags = _diagnostics_read_by(
                self, external, set(land_step.states) | exchange_names, list_var_diag)
            # No non-land equation reads this dataset. The land step consumes
            # states and prescribed drivers, which are already on Var_step.
            if not external:
                pre_land_diags = []
        else:
            pre_land_diags = list_var_diag
            external = None
            middle_diags = None

        ## get linear speeds for solving
        vLin = self._get_vLin(Par)

        ## create quick function for solving scheme
        if scheme =='ex': f_dX = scheme_ex
        elif scheme == 'imex': f_dX = scheme_imex
        else: raise ValueError("'scheme' can only be within ['ex', 'imex']")

        ## printing and time counter
        print(self.name + ' running')
        t0 = perf_counter()

        ## catch warnings (if requested)
        with warnings.catch_warnings():
            if no_warnings: warnings.filterwarnings('ignore')

            ## INITIALIZATION
            ## initialization of all variables
            Var_old = Ini.copy(deep=True)
            _store_diagnostics(self, Var_old, list_var_diag, Par,
                               For.sel({time_axis: time[0]}, drop=True))

            if land_step is not None and get_final:
                Var_fin = Var_old[list_var_prog + list_var_node].copy(deep=True)

            ## initialization of kept variables
            Var_out = Var_old.drop([var for var in Var_old if var not in list_var_keep])
            Var_out = [Var_out.assign_coords(**{time_axis: time[0]}).expand_dims(time_axis, 0)]

            ## LOOP ON TIME-STEP
            for t in range(1, len(time)):
                print(time_axis + ' = ' + str(int(time[t])), end=' ')

                ## adapt substep size
                if adapt_nt:
                    if adapt_nt.var in For:
                        steps[t] = adapt_nt(For.isel(year=slice(t-1, t+1))[adapt_nt.var].max(), nt_min=nt)
                    elif adapt_nt.var in self._processes:
                        steps[t] = adapt_nt(Var_old[adapt_nt.var].max(), nt_min=nt)
                    elif t==1:
                        print(f'WARNING: cannot adapt nt as {adapt_nt.var} is not a variable or driver')                

                ## get time step and drivers
                dt = float(time[t] - time[t-1]) / float(steps[t])
                For_t = For.sel({time_axis: time[t]}, drop=True)

                ## anticipate new output (initialized to zero)
                Var_out.append(0 * Var_out[-1].isel({time_axis: 0}, drop=True))

                ## LOOP ON SUBSTEPS
                print('(nt = ' + str(int(steps[t])) +')', end='\n' if t+1==len(time) else '\r')
                intervals = range(int(steps[t]))
                if land_step is not None:
                    # Only the fixed annual land event can split a native substep.
                    event_time = float(time[t]) - 0.5
                    edges = np.unique(np.append(np.linspace(float(time[t-1]), float(time[t]), int(steps[t]) + 1), event_time))
                    intervals = range(len(edges) - 1)
                for tt in intervals:

                    ## solve for variables
                    Var_new = xr.Dataset()
                    Var_step = Var_old
                    if land_step is not None:
                        left, right = float(edges[tt]), float(edges[tt+1])
                        dt = right - left
                        Var_step = Var_old.copy(deep=False)
                        Var_step.update(For_t)
                        # Refresh prescribed environment before the land calculation.
                        _store_diagnostics(self, Var_step, pre_land_diags, Par, For_t)
                        land_states, land_exchanges = land_step.prepare_interval(
                            Var_step, Par, For_t, left, right, left == event_time)
                        # Non-land equations still see the pre-update land state.
                        # They need this interval's exchanges and only the diagnostics
                        # they read. Kept diagnostics are evaluated once afterwards.
                        if external:
                            Var_step.update(land_exchanges)
                            _store_diagnostics(self, Var_step, middle_diags, Par, For_t)
                        Var_new.update(land_states)
                        Var_new.update(land_exchanges)
                    for var in list_var_prog:
                        if land_step is None or var not in land_step.states:
                            Var_new[var] = self[var](Var_step, Par, For_t, f_dot=lambda dX_dt: f_dX(dX_dt, dt, vLin[var]))
                    for var in list_var_node:
                        if land_step is None or var not in land_step.states:
                            Var_new[var] = self[var](Var_step, Par, For_t)
                    diag_names = (list_var_diag if land_step is None else
                                  [var for var in list_var_diag if var not in land_step.exchanges])
                    _store_diagnostics(self, Var_new, diag_names, Par, For_t)

                    ## iterate variables
                    if land_step is None:
                        Var_old = Var_new.copy(deep=True)
                    else:
                        # The land step copies stocks before it writes. Carrying
                        # those arrays forward avoids a second full cohort copy.
                        # Exchanges are not part of this dataset.
                        Var_old = Var_new[list_var_prog + list_var_node].copy(deep=False)

                    ## get final state variables
                    if get_final:
                        if t+1 == len(time) and tt+1 == len(intervals):
                            Var_fin = Var_new.drop([var for var in Var_new if var not in list_var_prog + list_var_node])

                    ## drop unwanted variables and add to new output
                    Var_new = Var_new.drop([var for var in Var_new if var not in list_var_keep])
                    weight = float(1/steps[t]) if land_step is None else dt / float(time[t] - time[t-1])
                    Var_out[-1] += weight * Var_new # duration-weighted substep-end means
                    #Var_out[-1] = Var_new # this would give end-of-year values

                ## assign time coordinate
                Var_out[-1] = Var_out[-1].assign_coords(**{time_axis: time[t]}).expand_dims(time_axis, 0)

            ## FINALIZATION
            ## concatenate/assign time axis of final output
            Var_out = xr.concat(Var_out, dim=time_axis)
            if get_final: Var_fin = Var_fin.assign_coords(**{time_axis: time[-1]})

            ## add model info
            Var_out.attrs['model'] = self.name
            if land_step is not None:
                Var_out.attrs['time_sampling'] = 'Initial label: initial values; later labels: duration-weighted substep-end means. Land exchanges: interval means.'
            if get_final: Var_fin.attrs['model'] = self.name

            ## clean up attributes and add units to output
            for var in Var_out: 
                Var_out[var].attrs = {}
                Var_out[var].attrs['units'] = self[var].units
            if get_final: 
                for var in Var_fin: 
                    Var_fin[var].attrs = {}
                    Var_fin[var].attrs['units'] = self[var].units

        ## printing time counter
        print('total running time: {:.1f} minutes'.format((perf_counter() - t0) / 60))

        ## return
        if get_final: return Var_out, Var_fin
        else: return Var_out


##################################################
##   2. PROCESSES
##################################################

class Process():
    '''
    Class defining one single process of a Model object.

    Init:
    ------
    Out (str)           name of output variable (only one!)
    In (tuple)          names of input variables (as str); can be an empty tuple

    Options:
    --------
    In2 (tuple)         names of optional input variables (as str)
                        default = ()
    Eq (callable)       function linking In to Out
                        default = None
    DiffEq (callable)   function linking In to the first time-derivative of Out
                        default = None
    vLin (callable)     equation of the speed of the linear part of the first order time differential equation;
                        default = 1E-18
    model (Model)       model to which the process belongs;
                        default = Model()
    units (str)         units of Out;
                        default = '?'
    core_dims (list)    dims over which Out must be defined (relevant only for prognostic variables);
                        default = []
    '''

    ## initialization
    def __init__(self, Out, In=(), In2=(), Eq=None, DiffEq=None, vLin=lambda Par: 1E-18, model=Model(), units='?', core_dims=[]):
        ## check types
        assert (type(Out), type(In), type(In2), type(model), type(units), type(core_dims)) == (str, tuple, tuple, Model, str, list)
        assert (callable(Eq) or Eq is None) and (callable(DiffEq) or DiffEq is None) and (callable(vLin) or vLin is None)
        ## base attributes
        self.Out, self.In, self.In2, self.model, self.units, self.core_dims = Out, In, In2, model, units, core_dims
        self.Eq = Eq if Eq is not None else lambda Var, Par: Var[Out] 
        self.DiffEq, self.vLin = DiffEq, vLin
        self.param = self.node = self.prog = False
        ## inform if param, node or prog variable
        if In == () and Eq.__code__.co_argcount == 1: self.param = True
        if Out in In:
            if DiffEq is None: self.node = True
            else: self.prog = True

    ## nice display
    def __repr__(self):
        out = '{} {}{}: {}'.format(self.Out, str(self.In).replace("'", ""), ' ({})'.format(str(self.In2).replace("'", "")) if len(self.In2)>0 else '', self.Eq.__repr__())
        out += '\n[in model: {}]'.format(self.model.name) * (self.model.name != '')
        return out

    ## -------

    ## get process drivers (for recursive call)
    def _get_var(self, var, Var, Par, For):
        ## if prescribed
        if var in For.keys(): return For[var]
        ## if available
        elif var in Var.keys(): return Var[var]
        ## otherwise
        elif var != self.Out: return self.model[var](Var, Par, For, recursive=True)
        else: raise RuntimeError('endless recursive call of process: {0}'.format(var))

    ## solve process
    def __call__(self, Var, Par, For=xr.Dataset(), f_dot=None, recursive=False, time_axis='year'):
        ## if prescribed
        if self.Out in For: return For[self.Out]
        ## if parameter
        elif self.param: return self.Eq(Par)
        ## otherwise get drivers (ignoring secondary)
        vars_in = list(self.In) + [var for var in self.In2 if var in Var or var in For]
        # Building a fresh Dataset per diagnostic dominates short land equations.
        # Use the caller dataset when it already carries those same objects.
        reused = (not recursive and not self.prog and f_dot is None and all(
            var in Var and (var not in For or _same_payload(Var[var], For[var])) for var in vars_in))
        if reused:
            host = Var
        elif recursive:
            host = xr.Dataset({var:self._get_var(var, Var, Par, For) for var in vars_in})
        else:
            host = xr.Dataset(dict([(var, For[var]) if var in For else (var, Var[var]) for var in vars_in]))
        ## and call equation
        if not self.prog: New = self.Eq(host, Par)
        else: New = host[self.Out] + f_dot(self.DiffEq(host, Par))
        ## put time axis first if available
        if time_axis in Var.dims: New = New.transpose(*([time_axis] + [dim for dim in New.dims if dim != time_axis]))
        ## output
        return New

