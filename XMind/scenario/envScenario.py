# Adapted from DiLu (https://github.com/PJLab-ADG/DiLu), licensed under the Apache License 2.0.
# Modified for XMind.
"""Scenario encoder for highway-v0: selects one vehicle per direction slot, computes its risk and
serialises the decision frame as JSON (the frame f_t that goes into working memory)."""

import json
import math
from typing import Any, Dict, Optional

import numpy as np
from highway_env.road.lane import StraightLane

from XMind.scenario.directions import (
    DETECTION_RANGE,
    DIRECTIONS,
    FRAME_KEY,
    SECOND_OF,
    SECOND_RANGE,
    risk_level,
)
from XMind.scenario.risk_calculator import (
    TTC_DIRECTIONS,
    PairRisk,
    VehicleRiskState,
    calculate_pair_risk,
)


ENV_TYPE = "highway-v0"

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle',
}

SIDE_HALF_LENGTH = 2.5                                         # |dx| < 2.5 m in an adjacent lane: left / right
LANE_OFFSET = {"front": 0, "front_left": -1, "front_right": 1}  # lane of the vehicle each extra slot follows


def _half_length(car, lane_unit):
    """Half extent of the car body projected onto the lane direction."""
    heading = float(car.heading)
    forward = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-np.sin(heading), np.cos(heading)])
    return 0.5 * (
        float(car.LENGTH) * abs(np.dot(forward, lane_unit))
        + float(car.WIDTH) * abs(np.dot(lateral, lane_unit))
    )


class EnvScenario:
    """Works on the live environment or on a rollout copy; the ego vehicle is always read from the
    environment because rollouts may replace it (hand-over to IDM)."""

    def __init__(self, env) -> None:
        self.env = env.unwrapped

    @property
    def ego(self):
        return self.env.vehicle

    @property
    def network(self):
        return self.env.road.network

    # ---------------- geometry ----------------

    def _ego_lane(self):
        """(ego lane, unit vector along it). Only straight, parallel lanes are supported."""
        lane = self.network.get_lane(self.ego.lane_index)
        if not isinstance(lane, StraightLane):
            raise ValueError("Directional selection and risk require straight lanes")
        vector = np.asarray(lane.end, dtype=float) - np.asarray(lane.start, dtype=float)
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("Invalid ego lane length")
        return lane, vector / length

    def _longitudinal(self, vehicle, lane_unit) -> float:
        """Signed centre offset along the ego lane (positive = ahead)."""
        return float(np.dot(np.asarray(vehicle.position) - np.asarray(self.ego.position), lane_unit))

    # ---------------- vehicles and risks ----------------

    def directional_vehicles(self) -> Dict[str, Optional[Any]]:
        """At most one vehicle per slot: the nearest one (ties keep the first in road.vehicles).

        |dx| of exactly 2.5 m belongs to the diagonal slots. The extra slots take the nearest vehicle
        ahead of front / front_left / front_right in the same lane, within SECOND_RANGE.
        """
        selected = dict.fromkeys(DIRECTIONS)
        best = {direction: float("inf") for direction in DIRECTIONS}
        ego = self.ego
        ego_index = ego.lane_index
        _, lane_unit = self._ego_lane()
        ahead = []   # (vehicle, lane offset, dx) for the extra slots

        for vehicle in self.env.road.vehicles:
            if vehicle is ego:
                continue
            lane_index = vehicle.lane_index
            if lane_index is None or lane_index[:2] != ego_index[:2]:
                continue
            offset = lane_index[2] - ego_index[2]
            if offset not in (-1, 0, 1):
                continue
            dx = self._longitudinal(vehicle, lane_unit)
            if not math.isfinite(dx):
                continue
            if 0.0 < dx <= SECOND_RANGE:
                ahead.append((vehicle, offset, dx))
            if abs(dx) > DETECTION_RANGE:
                continue

            if offset == 0:
                direction = "front" if dx >= 0 else "rear"
            else:
                side = "left" if offset == -1 else "right"
                if dx >= SIDE_HALF_LENGTH:
                    direction = "front_" + side
                elif dx <= -SIDE_HALF_LENGTH:
                    direction = "rear_" + side
                else:
                    direction = side
            if abs(dx) < best[direction]:
                selected[direction] = vehicle
                best[direction] = abs(dx)

        for second, first in SECOND_OF.items():
            if selected[first] is None:
                continue
            for vehicle, offset, dx in ahead:
                if (
                    offset == LANE_OFFSET[first]
                    and vehicle is not selected[first]
                    and best[first] < dx < best[second]
                ):
                    selected[second] = vehicle
                    best[second] = dx
        return selected

    def directional_risks(self, vehicles=None) -> Dict[str, Optional[PairRisk]]:
        """Risk per occupied slot (DRF, or TTC for critical same-lane leaders)."""
        if vehicles is None:
            vehicles = self.directional_vehicles()
        ego = self.ego
        ego_lane, lane_unit = self._ego_lane()
        ego_s, _ = ego_lane.local_coordinates(ego.position)
        ego_center = np.asarray(ego_lane.position(ego_s, 0.0), dtype=float)

        risks = dict.fromkeys(DIRECTIONS)
        for direction in DIRECTIONS:
            vehicle = vehicles[direction]
            if vehicle is None:
                continue
            dx = self._longitudinal(vehicle, lane_unit)
            same_lane = vehicle.lane_index == ego.lane_index
            if direction in TTC_DIRECTIONS and not same_lane:
                raise ValueError(direction + " slot must contain a same-lane vehicle")

            # For an adjacent-lane vehicle, the ego field is evaluated as if the ego vehicle were
            # shifted onto the centre line of that lane.
            position = np.asarray(ego.position, dtype=float).copy()
            if not same_lane:
                lane = self.network.get_lane(vehicle.lane_index)
                if not isinstance(lane, StraightLane):
                    raise ValueError("Directional risk requires straight lanes")
                target_s, _ = lane.local_coordinates(ego_center)
                position += np.asarray(lane.position(target_s, 0.0), dtype=float) - ego_center

            net_gap = closing_speed = None
            if direction in TTC_DIRECTIONS:
                net_gap = float(dx - _half_length(ego, lane_unit) - _half_length(vehicle, lane_unit))
                closing_speed = float(np.dot(
                    np.asarray(ego.velocity, dtype=float) - np.asarray(vehicle.velocity, dtype=float),
                    lane_unit,
                ))   # positive = closing in on the leader

            risks[direction] = calculate_pair_risk(
                VehicleRiskState(float(position[0]), float(position[1]), float(ego.speed), float(ego.heading)),
                VehicleRiskState(float(vehicle.position[0]), float(vehicle.position[1]),
                                 float(vehicle.speed), float(vehicle.heading)),
                direction,
                dx,
                net_gap=net_gap,
                closing_speed=closing_speed,
            )
        return risks

    # ---------------- frame JSON ----------------

    def _vehicle_state(self, vehicle, direction=None) -> Dict[str, Any]:
        velocity = vehicle.velocity
        row = {"id": "ego" if vehicle is self.ego else id(vehicle)}
        if direction is not None:
            row["direction"] = direction
        row.update({
            "x": float(vehicle.position[0]),
            "y": float(vehicle.position[1]),
            "v_x": float(velocity[0]),
            "v_y": float(velocity[1]),
            "a": float(vehicle.action["acceleration"]),
            "lane_id": int(vehicle.lane_index[2]),
        })
        return row

    def structured_state(self, frame_id: int, vehicles=None, risks=None) -> Dict[str, Any]:
        """{"Frame", "Context", "Ego", "11 Directions"}; an empty slot is None (null in JSON).

        Occupied slot: {id, direction, x, y, v_x, v_y, a, lane_id, risk: {value, level}}.
        """
        if vehicles is None:
            vehicles = self.directional_vehicles()
        if risks is None:
            risks = self.directional_risks(vehicles)

        slots = {}
        for direction in DIRECTIONS:
            vehicle = vehicles[direction]
            if vehicle is None:
                slots[direction] = None
                continue
            result = risks[direction]
            if result is None:
                raise ValueError("Missing risk for occupied direction: " + direction)
            value = float(result.risk)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Invalid risk for direction: " + direction)
            row = self._vehicle_state(vehicle, direction)
            row["risk"] = {"value": value, "level": risk_level(value)}
            slots[direction] = row

        return {
            "Frame": int(frame_id),
            "Context": ENV_TYPE,
            "Ego": self._vehicle_state(self.ego),
            FRAME_KEY: slots,
        }

    def describe(self, frame_id: int) -> str:
        """Current decision frame as JSON."""
        return json.dumps(self.structured_state(frame_id), ensure_ascii=False, allow_nan=False, indent=2)

    def lane_boundary(self) -> Dict[str, Any]:
        """Left / right road boundaries read from the road network."""
        lane_index = self.ego.lane_index
        if lane_index is None:
            raise ValueError("Missing current ego lane")
        lane_id = int(lane_index[2])
        lane_ids = {int(index[2]) for index in self.network.all_side_lanes(lane_index)}
        if lane_id not in lane_ids:
            raise ValueError("Current ego lane is absent from road network")
        return {
            "lane_id": lane_id,
            "lane_count": len(lane_ids),
            "left_blocked": lane_id - 1 not in lane_ids,
            "right_blocked": lane_id + 1 not in lane_ids,
        }

    def available_actions_description(self) -> str:
        text = 'Your available actions are: \n'
        for action in self.env.get_available_actions():
            text += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(action) + '\n'
        return text
