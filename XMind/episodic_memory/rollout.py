"""
Counterfactual verification of the actions at one decision state tau (Sec. III-B).

For every available action a and traffic condition xi_n (n = 1..J, xi_1 unperturbed):
    copy the snapshot taken before env.step at tau, apply xi_n, execute a for one decision
    interval, then let each continuation policy (IDM car following, greedy min-risk) drive
    until T_cf. All actions share the same J conditions.

    Rec(a, xi_n) = some continuation avoids collision
    q(a)   = fraction of conditions that are not recoverable
    r(a)   = mean over conditions of the lowest peak risk among collision-free continuations (1 if none)
    p(a)   = mean progress dx / (v_ref * T_cf) of the selected continuation
    A+ = {q = 0}, A- = {q >= eta}, A0 = the remaining available actions
    a* = argmax_{a in A+} [p(a) - lambda * r(a)]

Rollouts work on deep copies, never touch the live environment and are reproducible.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from XMind.scenario.directions import ACTION_NAMES
from XMind.scenario.envScenario import EnvScenario


ACTIONS = tuple(sorted(ACTION_NAMES))    # (0, 1, 2, 3, 4)
TIE_PREFERENCE = (1, 4, 3, 0, 2)         # IDLE > SLOWER > FASTER > LANE_LEFT > LANE_RIGHT
PERTURBATION_SEED = 20261104             # combined with (scene seed, frame) for reproducible conditions

LABEL_SAFE = "safe"                # A+
LABEL_DANGER = "danger"            # A-
LABEL_UNCERTAIN = "uncertain"      # A0
LABEL_UNAVAILABLE = "unavailable"


# ============================================================
# Snapshots and branch copies
# ============================================================

def _base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def take_snapshot(env):
    """Snapshot of decision state tau; call it before env.step of that step."""
    return copy.deepcopy(_base_env(env))   # AbstractEnv.__deepcopy__ skips the viewer and video wrapper


def prune(env, radius):
    """Keep only vehicles within `radius` m of the ego vehicle (radius <= 0: keep all)."""
    if radius <= 0:
        return env
    ego = env.vehicle
    x0 = float(ego.position[0])
    env.road.vehicles = [
        vehicle for vehicle in env.road.vehicles
        if vehicle is ego or abs(float(vehicle.position[0]) - x0) <= radius
    ]
    return env


def clone(snapshot, radius=0.0):
    return prune(copy.deepcopy(snapshot), radius)


def other_vehicles(env):
    ego = env.vehicle
    return [vehicle for vehicle in env.road.vehicles if vehicle is not ego]


# ============================================================
# Traffic conditions xi_n
# ============================================================

@dataclass(frozen=True)
class Perturbation:
    """Condition `index` (1 = unperturbed). One factor tuple per surrounding vehicle, in the order of
    other_vehicles(): (target speed, TIME_WANTED, DISTANCE_WANTED, COMFORT_ACC_MIN)."""

    index: int
    factors: Tuple[Tuple[float, float, float, float], ...]

    @property
    def is_identity(self):
        return self.index == 1


def make_perturbations(n_vehicles, settings, seed):
    rng = np.random.default_rng(seed)
    low, high = settings.idm_scale
    c_low, c_high = settings.comfort_decel_scale
    perturbations = [Perturbation(1, tuple((1.0, 1.0, 1.0, 1.0) for _ in range(n_vehicles)))]
    for index in range(2, settings.n_perturbations + 1):
        perturbations.append(Perturbation(index, tuple(
            (
                float(rng.uniform(low, high)),
                float(rng.uniform(low, high)),
                float(rng.uniform(low, high)),
                float(rng.uniform(c_low, c_high)),
            )
            for _ in range(n_vehicles)
        )))
    return perturbations


def apply_perturbation(env, perturbation):
    """Scale the IDM parameters of the surrounding vehicles of a copy (instance attributes only)."""
    others = other_vehicles(env)
    if len(others) != len(perturbation.factors):
        raise RuntimeError("Perturbation size does not match the pruned clone")
    if perturbation.is_identity:
        return env
    for vehicle, (f_speed, f_time, f_dist, f_decel) in zip(others, perturbation.factors):
        if not isinstance(vehicle, IDMVehicle):
            continue
        vehicle.target_speed = float(vehicle.target_speed) * f_speed
        vehicle.TIME_WANTED = float(vehicle.TIME_WANTED) * f_time
        vehicle.DISTANCE_WANTED = float(vehicle.DISTANCE_WANTED) * f_dist
        vehicle.COMFORT_ACC_MIN = float(vehicle.COMFORT_ACC_MIN) * f_decel
    return env


# ============================================================
# Stepping, risk probe and continuation policies
# ============================================================

def advance(env, action):
    """One decision interval; returns whether the ego vehicle has crashed. action=None after IDM hand-over."""
    env.step(action)
    return bool(env.vehicle.crashed)


class RiskProbe:
    """Maximum directional risk on a copy (same DRF / TTC risk as online decisions)."""

    def __init__(self, env):
        self.scenario = EnvScenario(env)

    def max_slot_risk(self):
        if self.scenario.ego.crashed:
            return 1.0
        values = [float(risk.risk) for risk in self.scenario.directional_risks().values() if risk is not None]
        return max(values) if values else 0.0


def hand_over_to_idm(env):
    """Replace the ego vehicle by an IDMVehicle without lane changes (an ongoing lane change finishes)."""
    ego = env.vehicle
    follower = IDMVehicle.create_from(ego)
    follower.enable_lane_change = False
    vehicles = env.road.vehicles
    vehicles[next(i for i, vehicle in enumerate(vehicles) if vehicle is ego)] = follower
    env.controlled_vehicles[0] = follower
    return follower


def risk_greedy_step(env):
    """One step of the greedy min-risk policy: the available action whose one-interval look-ahead has
    the lowest maximum risk. Returns (look-ahead copy of that action, crashed, action)."""
    best = None
    for action in sorted(env.get_available_actions(), key=TIE_PREFERENCE.index):
        child = copy.deepcopy(env)
        crashed = advance(child, action)
        risk = 1.0 if crashed else RiskProbe(child).max_slot_risk()
        key = (crashed, risk, TIE_PREFERENCE.index(action))
        if best is None or key < best[0]:
            best = (key, child, crashed, action)
    return best[1], best[2], best[3]


# ============================================================
# Branch(tau, a, xi, mu)
# ============================================================

@dataclass
class BranchResult:
    policy: str
    collided: bool
    crash_time: Optional[float]   # s after tau (decision-interval resolution); None without collision
    peak_risk: float              # max over decision intervals of the maximum directional risk; 1 on collision
    progress: float               # ego longitudinal displacement (m)

    def to_dict(self):
        return {
            "collided": self.collided,
            "crash_time": self.crash_time,
            "peak_risk": round(self.peak_risk, 4),
            "progress": round(self.progress, 2),
        }


def _continue(env, policy, first_risk, x0, n_steps, dt):
    """Drive with continuation policy `policy` from the second decision interval to n_steps."""
    trace = [first_risk]
    probe = RiskProbe(env)
    for k in range(1, n_steps):
        if policy == "idm":
            if k == 1:
                hand_over_to_idm(env)
            crashed = advance(env, None)
        elif policy == "risk_greedy":
            env, crashed, _ = risk_greedy_step(env)
            probe = RiskProbe(env)
        else:
            raise ValueError("Unknown continuation policy: " + policy)
        progress = float(env.vehicle.position[0]) - x0
        if crashed:
            return BranchResult(policy, True, (k + 1) * dt, 1.0, progress)
        trace.append(probe.max_slot_risk())
    return BranchResult(policy, False, None, max(trace), float(env.vehicle.position[0]) - x0)


def branch_all_policies(snapshot, action, perturbation, settings, n_steps, dt):
    """Results for every continuation policy; the first interval is simulated once and then copied."""
    first = apply_perturbation(clone(snapshot, settings.prune_radius_m), perturbation)
    x0 = float(first.vehicle.position[0])
    crashed = advance(first, action)
    progress = float(first.vehicle.position[0]) - x0
    policies = settings.continuation_policies

    if crashed:   # no continuation can recover
        return {policy: BranchResult(policy, True, dt, 1.0, progress) for policy in policies}
    first_risk = RiskProbe(first).max_slot_risk()
    if n_steps == 1:
        return {policy: BranchResult(policy, False, None, first_risk, progress) for policy in policies}

    results = {}
    for i, policy in enumerate(policies):
        env = first if i == len(policies) - 1 else copy.deepcopy(first)
        results[policy] = _continue(env, policy, first_risk, x0, n_steps, dt)
    return results


# ============================================================
# Action-outcome table
# ============================================================

@dataclass
class ActionOutcome:
    action: int
    available: bool
    label: str
    c: Optional[float] = None              # q(a): unrecoverable fraction
    r_hat: Optional[float] = None          # mean action risk
    p: Optional[float] = None              # mean normalised progress
    crash_time: Optional[float] = None     # earliest collision over unrecoverable conditions (s)
    per_perturbation: List[dict] = field(default_factory=list)

    @property
    def name(self):
        return ACTION_NAMES[self.action]

    @property
    def n_unrecoverable(self):
        return sum(not row["recoverable"] for row in self.per_perturbation)

    def to_dict(self):
        out = {"action": self.action, "name": self.name, "available": self.available, "label": self.label}
        if self.available:
            out.update({
                "c": round(self.c, 4),
                "r_hat": round(self.r_hat, 4),
                "p": round(self.p, 4),
                "crash_time": self.crash_time,
                "per_perturbation": self.per_perturbation,
            })
        return out


@dataclass
class ActionOutcomeTable:
    frame: int
    horizon_s: float
    n_perturbations: int
    outcomes: Dict[int, ActionOutcome]
    best_action: Optional[int]
    n_branches: int

    def actions_with(self, label):
        return [a for a in ACTIONS if self.outcomes[a].label == label]

    @property
    def safe(self):
        return self.actions_with(LABEL_SAFE)

    @property
    def danger(self):
        return self.actions_with(LABEL_DANGER)

    @property
    def uncertain(self):
        return self.actions_with(LABEL_UNCERTAIN)

    @property
    def available(self):
        return [a for a in ACTIONS if self.outcomes[a].available]

    @property
    def discriminativeness(self):
        """|A-| / |available actions|."""
        return len(self.danger) / len(self.available) if self.available else 0.0

    @property
    def severity(self):
        """max_a q(a)."""
        values = [self.outcomes[a].c for a in self.available]
        return max(values) if values else 0.0

    def to_dict(self):
        return {
            "frame": self.frame,
            "horizon_s": self.horizon_s,
            "n_perturbations": self.n_perturbations,
            "actions": {ACTION_NAMES[a]: self.outcomes[a].to_dict() for a in ACTIONS},
            "safe_actions": self.safe,
            "danger_actions": self.danger,
            "uncertain_actions": self.uncertain,
            "best_action": self.best_action,
            "discriminativeness": round(self.discriminativeness, 4),
            "n_branches": self.n_branches,
        }

    def summary(self):
        parts = []
        for a in ACTIONS:
            o = self.outcomes[a]
            if not o.available:
                parts.append(f"{o.name}: n/a")
            else:
                parts.append(f"{o.name}: {o.label} c={o.c:.2f} r={o.r_hat:.2f} p={o.p:.2f}")
        best = ACTION_NAMES[self.best_action] if self.best_action is not None else "none"
        return f"frame {self.frame} | " + "; ".join(parts) + f" | a*={best} | D={self.discriminativeness:.2f}"


def _label(c, settings):
    if c == 0.0:
        return LABEL_SAFE
    if c >= settings.danger_threshold:
        return LABEL_DANGER
    return LABEL_UNCERTAIN


def _best_action(outcomes, settings):
    """a* over A+; ties prefer lower risk, then TIE_PREFERENCE."""
    safe = [o for o in outcomes.values() if o.label == LABEL_SAFE]
    if not safe:
        return None
    lam = settings.efficiency_weight
    return min(safe, key=lambda o: (-(o.p - lam * o.r_hat), o.r_hat, TIE_PREFERENCE.index(o.action))).action


def _condition_row(results, perturbation, settings):
    """Outcome of one action under one traffic condition."""
    safe_policies = [r for r in results.values() if not r.collided]
    if safe_policies:
        # recoverable: the continuation with the lowest peak risk (then larger progress)
        chosen = min(safe_policies, key=lambda r: (r.peak_risk, -r.progress,
                                                   settings.continuation_policies.index(r.policy)))
        recoverable, peak_risk, crash_time = True, chosen.peak_risk, None
    else:
        # unrecoverable: risk 1; progress and crash time of the latest collision (description only)
        chosen = max(results.values(), key=lambda r: (r.crash_time, r.progress))
        recoverable, peak_risk, crash_time = False, 1.0, chosen.crash_time
    return {
        "perturbation": perturbation.index,
        "recoverable": recoverable,
        "peak_risk": peak_risk,
        "progress": chosen.progress,
        "policy": chosen.policy,
        "crash_time": crash_time,
        "branches": {p: r.to_dict() for p, r in results.items()},
    }


def evaluate_frame(snapshot, frame_index, scene_seed, settings, actions=ACTIONS):
    """Action-outcome table for every available action at the snapshot's decision state."""
    dt = 1.0 / float(snapshot.config["policy_frequency"])
    n_steps = max(1, int(round(settings.horizon_s / dt)))
    horizon = n_steps * dt

    available = set(snapshot.get_available_actions())
    n_others = len(other_vehicles(clone(snapshot, settings.prune_radius_m)))
    perturbations = make_perturbations(
        n_others, settings, seed=[abs(int(scene_seed)), int(frame_index), PERTURBATION_SEED])

    outcomes = {}
    n_branches = 0
    for action in actions:
        if action not in available:
            outcomes[action] = ActionOutcome(action, False, LABEL_UNAVAILABLE)
            continue
        rows = []
        for perturbation in perturbations:
            results = branch_all_policies(snapshot, action, perturbation, settings, n_steps, dt)
            n_branches += len(results)
            rows.append(_condition_row(results, perturbation, settings))

        c = sum(not row["recoverable"] for row in rows) / len(rows)
        crash_times = [row["crash_time"] for row in rows if row["crash_time"] is not None]
        outcomes[action] = ActionOutcome(
            action=action,
            available=True,
            label=_label(c, settings),
            c=c,
            r_hat=float(np.mean([row["peak_risk"] for row in rows])),
            p=float(np.mean([row["progress"] for row in rows])) / (settings.v_ref * horizon),
            crash_time=min(crash_times) if crash_times else None,
            per_perturbation=[
                {
                    "perturbation": row["perturbation"],
                    "recoverable": row["recoverable"],
                    "peak_risk": round(row["peak_risk"], 4),
                    "progress": round(row["progress"], 2),
                    "policy": row["policy"],
                    "crash_time": row["crash_time"],
                    "branches": row["branches"],
                }
                for row in rows
            ],
        )

    return ActionOutcomeTable(
        frame=int(frame_index),
        horizon_s=horizon,
        n_perturbations=len(perturbations),
        outcomes=outcomes,
        best_action=_best_action(outcomes, settings),
        n_branches=n_branches,
    )
