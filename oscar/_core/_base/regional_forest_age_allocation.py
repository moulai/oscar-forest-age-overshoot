"""Capacity-constrained age preferences, independent of carbon integration.

Areas and demands have the same units. Weights are relative preferences per
unit available area; zero weights prohibit use. No ecological parameter is
introduced by solving the demand constraints.
"""
from math import fsum

import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.csgraph import connected_components


def _positive(value, name):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return value


def select_area(area, weight, demand, *, yield_density=None):
    """Return selected area and unmet demand, without reopening zero weights.

    Optional yield_density changes the demand unit to actual extracted carbon.
    The selection fraction still follows min(1, multiplier * weight).
    """
    area, weight = np.broadcast_arrays(_positive(area, "area"), _positive(weight, "weight"))
    if area.ndim != 1 or np.ndim(demand) or not np.isfinite(demand) or demand < 0:
        raise ValueError("one-dimensional areas and a nonnegative scalar demand required")
    density = np.ones_like(area) if yield_density is None else _positive(yield_density, "yield_density")
    if density.shape != area.shape:
        raise ValueError("yield_density must have the area shape")
    eligible = (area > 0) & (weight > 0) & (density > 0)
    capacity = area * density
    selected = np.zeros_like(area)
    target = min(float(demand), fsum(capacity[eligible]))
    missing = float(demand) - target
    if target == 0:
        return selected, missing
    remaining = target
    active = eligible.copy()
    # At least one source saturates on every nonterminal pass.
    for _ in range(area.size + 1):
        if not active.any():
            break
        relative = weight[active] / weight[active].max()
        score = capacity[active] * relative
        proposed = remaining * score / fsum(score)
        indices = np.flatnonzero(active)
        saturated = proposed >= capacity[indices]
        if not saturated.any():
            selected[indices] = proposed / density[indices]
            break
        used = indices[saturated]
        selected[used] = area[used]
        active[used] = False
        # Keep the original demand in the sum: a tiny carbon residual can
        # determine a substantial selection fraction after other sources saturate.
        remaining = fsum([target, *(-capacity[eligible & ~active])])
        if remaining <= 0:
            break
    return selected, missing


def select_joint_area(area, weights, demands, *, tolerance=1e-10):
    """Allocate several fixed-area activities against one source inventory.

    Minimize sum(h log(h/(A*w))-h), subject to activity totals and shared
    source capacities. No row order is privileged. Infeasible joint demands
    raise; a numerical failure is reported separately from infeasibility.
    """
    area, weights, demands = (_positive(area, "area"), _positive(weights, "weights"),
                              _positive(demands, "demands"))
    if area.ndim != 1 or weights.shape != (demands.size, area.size) or demands.ndim != 1:
        raise ValueError("expected source areas, activity-by-source weights, and demands")
    out = np.zeros_like(weights)
    rows = demands > 0
    cols = (area > 0) & np.any(weights[rows] > 0, axis=0)
    if not rows.any():
        return out
    scale = float(area[cols].sum())
    if scale == 0 or demands.sum() > scale + tolerance * max(scale, 1.):
        raise ValueError("joint activity demands exceed eligible source area")
    a, d, w = area[cols] / scale, demands[rows] / scale, weights[np.ix_(rows, cols)].copy()
    if np.any(w.max(axis=1) == 0):
        raise ValueError("positive activity demand has no eligible source")
    w /= w.max(axis=1)[:, None]
    # Identical preferences have an exact single-demand reduction.
    if np.all(w == w[:1]):
        chosen, gap = select_area(a, w[0], float(d.sum()))
        if gap > tolerance:
            raise ValueError("joint activity demands are infeasible")
        out[np.ix_(rows, cols)] = scale * d[:, None] / d.sum() * chosen[None, :]
        return out
    nr, nc = w.shape
    if np.any(w == 0):
        # Eligibility can make a subset infeasible despite ample total area.
        row_sum = kron(eye(nr), csr_matrix(np.ones((1, nc))), format="csr")
        col_sum = kron(csr_matrix(np.ones((1, nr))), eye(nc), format="csr")
        feasibility = linprog(np.zeros(nr * nc), A_ub=col_sum, b_ub=a,
                              A_eq=row_sum, b_eq=d,
                              bounds=[(0., None if x else 0.) for x in w.ravel()], method="highs",
                              options={"primal_feasibility_tolerance": 1e-10,
                                       "dual_feasibility_tolerance": 1e-10})
        if not feasibility.success:
            raise ValueError("activity demands and source eligibility are incompatible")
        # A tight protected-source subset can force other *eligible* edges to
        # zero. Finite entropy multipliers cannot express those boundary zeros.
        # Find them from any feasible flow. Include unused capacity as a slack
        # activity: an edge can increase precisely when a residual cycle exists.
        flow = feasibility.x.reshape(nr, nc)
        slack = np.maximum(a - flow.sum(axis=0), 0.)
        graph_weights = np.vstack((w, np.ones(nc))) if slack.any() else w
        graph_flow = np.vstack((flow, slack)) if slack.any() else flow
        nrows = graph_weights.shape[0]
        ri, ci = np.nonzero(graph_weights > 0)
        rj, cj = np.nonzero(graph_flow > 0)
        graph = csr_matrix((np.ones(ri.size + rj.size),
                            (np.r_[ri, nrows+cj], np.r_[nrows+ci, rj])),
                           shape=(nrows+nc, nrows+nc))
        _, component = connected_components(graph, directed=True, connection="strong")
        w *= component[:nr, None] == component[None, nrows:]
    def allocation(log_multiplier):
        q = w * np.exp(log_multiplier[:, None])
        return a[None, :] * q / np.maximum(1., q.sum(axis=0))[None, :]
    def residual(x):
        return allocation(x).sum(axis=1) - d
    start = np.log(d / (w * a).sum(axis=1))
    solution = least_squares(residual, start, bounds=(-700., 700.),
                             xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=500)
    selected = allocation(solution.x)
    if np.max(np.abs(selected.sum(axis=1) - d)) > tolerance:
        raise RuntimeError("joint allocation did not converge to the feasible demands")
    out[np.ix_(rows, cols)] = selected * scale
    return out


def _uniform_joint_or_none(area, weights, demands, tolerance=1e-10):
    """Same numbers as select_joint_area when every row has one weight vector.

    None tells the caller to use select_joint_area. That covers non-uniform
    rows and demands the original routine rejects.
    """
    area = np.asarray(area, dtype=float)
    weights = np.asarray(weights, dtype=float)
    demands = np.asarray(demands, dtype=float)
    if (area.ndim != 1 or demands.ndim != 1 or weights.ndim != 2
            or weights.shape != (demands.size, area.size)):
        return None
    if np.any(area < 0) or not np.all(np.isfinite(area)):
        return None
    out = np.zeros_like(weights)
    if not np.any(demands > 0):
        return out
    rows = demands > 0
    wrow = weights[rows]
    if not np.all(wrow == wrow[:1]):
        return None
    cols = (area > 0) & np.any(wrow > 0, axis=0)
    if not np.any(cols):
        return None
    scale = float(area[cols].sum())
    if scale == 0.0 or float(demands.sum()) > scale + tolerance * max(scale, 1.0):
        return None
    a = area[cols] / scale
    d = demands[rows] / scale
    w = wrow[:, cols]
    wmax = w.max(axis=1)
    if np.any(wmax == 0.0):
        return None
    w = w / wmax[:, None]
    if not np.all(w == w[:1]):
        return None
    w0 = w[0]
    # Unequal positive weights can saturate one source before another.
    # That is the general solver. Equal positive weights share one fraction.
    positive = w0[w0 > 0.0]
    if positive.size == 0 or np.any(positive != positive[0]):
        return None
    eligible = (a > 0.0) & (w0 > 0.0)
    cap = a[eligible]
    if cap.size == 0:
        return None
    demand = float(d.sum())
    total = fsum(cap)
    target = demand if demand < total else total
    if demand - target > tolerance:
        return None
    chosen = np.zeros_like(a)
    if target != 0.0:
        chosen[eligible] = target * cap / fsum(cap)
    out[np.ix_(np.flatnonzero(rows), np.flatnonzero(cols))] = scale * d[:, None] / d.sum() * chosen[None, :]
    return out


def select_joint_block(area, weights, demands, *, tolerance=1e-10):
    """select_joint_area for every region. Uniform rows skip its wrapper."""
    area = np.asarray(area, dtype=float)
    weights = np.asarray(weights, dtype=float)
    demands = np.asarray(demands, dtype=float)
    out = np.zeros(weights.shape, dtype=float)
    if area.shape[0] == 0 or demands.shape[1] == 0:
        return out
    uniform = bool(np.all(weights == weights[:, :1]))
    for ri in range(area.shape[0]):
        if not np.any(demands[ri] > 0):
            continue
        chosen = _uniform_joint_or_none(area[ri], weights[ri], demands[ri], tolerance) if uniform else None
        if chosen is None:
            chosen = select_joint_area(area[ri], weights[ri], demands[ri], tolerance=tolerance)
        out[ri] = chosen
    return out


def select_ranked_area(area, ages, eligibility, demands, directions, *, tolerance=1e-10):
    """Select two shared-pool area demands from opposite real-age ends.

    Directions are explicitly 'youngest' and 'oldest', in demand-row order.
    Equal-age sources share each selected age-group area in proportion to
    their available area. Eligibility is Boolean and must match across rows.
    No incoming area, carbon density, label inference or priority weight enters
    this selector. Material shortage raises; roundoff-scale excess shares the
    shortage proportionally, within the existing joint-area tolerance.
    """
    area, ages, demands = (_positive(area, "area"), _positive(ages, "ages"),
                           _positive(demands, "demands"))
    eligibility = np.asarray(eligibility)
    if (area.ndim != 1 or ages.shape != area.shape or demands.shape != (2,)
            or eligibility.shape != (2, area.size) or eligibility.dtype != np.dtype(bool)):
        raise ValueError("rank selection requires source ages/areas, two demands and Boolean eligibility")
    if len(directions) != 2 or set(directions) != {"youngest", "oldest"}:
        raise ValueError("rank selection requires opposite youngest/oldest directions")
    if not np.array_equal(eligibility[0], eligibility[1]):
        raise ValueError("rank selection requires shared eligibility")
    out = np.zeros((2, area.size))
    active = eligibility[0] & (area > 0)
    capacity = float(area[active].sum())
    demand = float(demands.sum())
    if demand > capacity + tolerance * max(1., capacity):
        raise ValueError("rank demands exceed eligible source area")
    if not demand or not capacity:
        return out
    # This only absorbs numerical excess; it is not a shortage scenario rule.
    targets = demands * min(1., capacity / demand)
    _, group = np.unique(ages[active], return_inverse=True)
    group_area = np.bincount(group, weights=area[active])
    for row, direction in enumerate(directions):
        ordered = group_area if direction == "youngest" else group_area[::-1]
        lower = np.r_[0., np.cumsum(ordered[:-1])]
        chosen = np.minimum(ordered, np.maximum(targets[row] - lower, 0.))
        if direction == "oldest":
            chosen = chosen[::-1]
        out[row, active] = area[active] * (chosen / group_area)[group]
    if (np.max(abs(out.sum(axis=1) - demands)) > tolerance * max(1., capacity)
            or np.any(out.sum(axis=0) > area + tolerance * max(1., capacity))):
        raise RuntimeError("rank allocation failed its area constraints")
    return out
