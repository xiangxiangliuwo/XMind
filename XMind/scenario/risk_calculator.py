"""
Pairwise interaction risk between the ego vehicle and one surrounding vehicle.

DRF (Eq. 2, default): the fraction of the trailing vehicle's risk field that lies in the overlap
    of the two field regions, r = sum_{overlap} F_ref / sum_{ref region} F_ref.
TTC (Eq. 3): used instead for critical longitudinal following, i.e. a same-lane leader
    (front, front_front) slower than 5 m/s with a closing speed dv above 5 m/s:
    r = 1 if gap <= 0; 0 if dv <= 0; clip((6 - gap / dv) / 3, 0, 1) otherwise.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from XMind.scenario.risk_field import GRID_RESOLUTION, compute_risk_field, look_ahead_distance


FIELD_THRESHOLD = 0.01        # a grid cell belongs to a field region if its intensity >= 0.01
TTC_MAX_LEADER_SPEED = 5.0    # m/s, leader speed below this ...
TTC_MIN_CLOSING_SPEED = 5.0   # m/s, ... and closing speed above this -> TTC
TTC_DIRECTIONS = ("front", "front_front")
FRONT_DIRECTIONS = ("front", "front_left", "front_right", "front_front", "front_left_front", "front_right_front")
REAR_DIRECTIONS = ("rear", "rear_left", "rear_right")


@dataclass(frozen=True)
class VehicleRiskState:
    """Position in m, speed in m/s, heading in rad."""

    x: float
    y: float
    speed: float
    heading_rad: float


@dataclass(frozen=True)
class PairRisk:
    risk: float
    relation: str                          # "front": the other vehicle leads; "rear": it follows
    method: str                            # "drf_reference_integral_coverage" or "ttc_piecewise"
    reference_integral: Optional[float] = None
    ttc_seconds: Optional[float] = None    # None when there is no finite TTC
    net_gap: Optional[float] = None
    closing_speed: Optional[float] = None

    @property
    def ratio_defined(self) -> bool:
        """Whether the risk is well defined (shown with a '*' in videos otherwise)."""
        if self.method == "ttc_piecewise":
            return bool(
                self.net_gap is not None and self.closing_speed is not None
                and np.isfinite([self.net_gap, self.closing_speed, self.risk]).all()
            )
        return self.reference_integral > 0.0


def risk_relation(direction: str, delta_s: float) -> str:
    if not np.isfinite(delta_s):
        raise ValueError("delta_s must be finite")
    if direction in FRONT_DIRECTIONS:
        return "front"
    if direction in REAR_DIRECTIONS:
        return "rear"
    if direction in ("left", "right"):
        return "front" if delta_s > 0.0 else "rear"
    raise ValueError("Unknown direction: " + direction)


def make_common_grid(ego: VehicleRiskState, other: VehicleRiskState) -> Tuple[np.ndarray, np.ndarray]:
    """World grid covering the full DRF regions of both vehicles."""
    states = (ego, other)
    for state in states:
        if not np.isfinite([state.x, state.y, state.speed, state.heading_rad]).all():
            raise ValueError("Vehicle state contains non-finite values")

    radii = [2.0 * look_ahead_distance(state.speed) for state in states]
    step = GRID_RESOLUTION
    x_min = min(s.x - r for s, r in zip(states, radii))
    x_max = max(s.x + r for s, r in zip(states, radii))
    y_min = min(s.y - r for s, r in zip(states, radii))
    y_max = max(s.y + r for s, r in zip(states, radii))
    return np.meshgrid(
        np.arange(np.floor(x_min / step) * step - step, np.ceil(x_max / step) * step + 2 * step, step),
        np.arange(np.floor(y_min / step) * step - step, np.ceil(y_max / step) * step + 2 * step, step),
    )


def _field(state: VehicleRiskState, grid) -> np.ndarray:
    return np.asarray(compute_risk_field(state.x, state.y, state.speed, state.heading_rad, grid), dtype=np.float64)


def drf_risk(ego: VehicleRiskState, other: VehicleRiskState, relation: str) -> Tuple[float, float]:
    """(risk, reference integral). The reference field is the trailing vehicle's field."""
    grid = make_common_grid(ego, other)
    ego_field, other_field = _field(ego, grid), _field(other, grid)
    if not np.isfinite(ego_field).all() or not np.isfinite(other_field).all():
        raise ValueError("Risk fields contain non-finite values")
    if np.any(ego_field < 0.0) or np.any(other_field < 0.0):
        raise ValueError("DRF intensities must be nonnegative")

    ego_mask = ego_field >= FIELD_THRESHOLD
    other_mask = other_field >= FIELD_THRESHOLD
    overlap_mask = ego_mask & other_mask
    field, mask = (ego_field, ego_mask) if relation == "front" else (other_field, other_mask)

    cell_area = GRID_RESOLUTION ** 2
    reference = float(np.sum(field[mask], dtype=np.float64) * cell_area)
    covered = float(np.sum(field[overlap_mask], dtype=np.float64) * cell_area)
    if not np.isfinite([reference, covered]).all():
        raise ValueError("DRF integral overflow")
    risk = float(np.clip(covered / reference, 0.0, 1.0)) if reference > 0.0 else 0.0
    return risk, reference


def ttc_risk(net_gap: float, closing_speed: float) -> Tuple[float, Optional[float]]:
    """(risk, TTC in s or None)."""
    if net_gap <= 0.0:
        return 1.0, 0.0                       # the longitudinal gap is already used up
    if closing_speed <= 0.0:
        return 0.0, None                      # not closing in
    ttc = net_gap / closing_speed
    if not np.isfinite(ttc):
        raise ValueError("TTC calculation overflow")
    return float(np.clip((6.0 - ttc) / 3.0, 0.0, 1.0)), ttc


def calculate_pair_risk(
    ego: VehicleRiskState,
    other: VehicleRiskState,
    direction: str,
    delta_s: float,
    *,
    net_gap: Optional[float] = None,
    closing_speed: Optional[float] = None,
) -> PairRisk:
    """Risk of `other` (in slot `direction`, signed longitudinal offset `delta_s`) for the ego vehicle.

    net_gap / closing_speed (longitudinal, along the lane; closing_speed > 0 when the ego vehicle
    approaches) are required for TTC_DIRECTIONS.
    """
    relation = risk_relation(direction, delta_s)
    if (net_gap is None) != (closing_speed is None):
        raise ValueError("net_gap and closing_speed must be supplied together")
    if net_gap is not None:
        net_gap, closing_speed = float(net_gap), float(closing_speed)
        if not np.isfinite([net_gap, closing_speed]).all():
            raise ValueError("TTC inputs must be finite")
    elif direction in TTC_DIRECTIONS:
        raise ValueError(direction + " requires net_gap and closing_speed")

    use_ttc = (
        direction in TTC_DIRECTIONS
        and delta_s > 0.0
        and other.speed < TTC_MAX_LEADER_SPEED
        and closing_speed > TTC_MIN_CLOSING_SPEED
    )
    if use_ttc:
        risk, ttc = ttc_risk(net_gap, closing_speed)
        return PairRisk(risk, relation, "ttc_piecewise", None, ttc, net_gap, closing_speed)

    risk, reference = drf_risk(ego, other, relation)
    return PairRisk(risk, relation, "drf_reference_integral_coverage", reference, None, net_gap, closing_speed)
