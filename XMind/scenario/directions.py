"""
Direction slots, the 4x3 risk layout and left-right mirroring (shared by the scenario encoder,
working memory and episodic memory).

Direction slots (11)
    Eight nearby slots: the nearest vehicle in the ego lane or an adjacent lane, within
    DETECTION_RANGE (50 m) ahead or behind.
    Three extra forward slots: the next vehicle ahead of front / front_left / front_right in the
    same lane, within SECOND_RANGE (80 m) of the ego vehicle; empty if that first vehicle is missing.

Risk layout rho(f), 4 rows x 3 columns
    Rows: second ahead / ahead / beside / behind. Columns: left lane / ego lane / right lane.
    Vehicle cells hold risk.level (0, 1, 2); the ego cell is always 0; a missing adjacent lane sets
    the left or right cell to 2. Layout similarity = 1 - ||rho_t - rho_m||^2 / RISK_NORM with
    RISK_NORM = 11 vehicle cells x 2^2 = 44.

Mirroring (the highway is left-right symmetric)
    Left and right directions, LANE_LEFT / LANE_RIGHT, layout columns and road boundaries are
    swapped; lateral positions are reflected about the ego vehicle and lateral velocities change
    sign. Lane ids keep their original values and are not used after mirroring.
"""

import copy
import json


FRAME_KEY = "11 Directions"
OLD_FRAME_KEY = "8 Directions"   # field name used by older outputs; converted when read

BASE_DIRECTIONS = (
    "front", "rear", "left", "right",
    "front_left", "rear_left", "front_right", "rear_right",
)
SECOND_DIRECTIONS = ("front_front", "front_left_front", "front_right_front")
DIRECTIONS = BASE_DIRECTIONS + SECOND_DIRECTIONS

# extra slot -> the slot of the vehicle it follows
SECOND_OF = {
    "front_front": "front",
    "front_left_front": "front_left",
    "front_right_front": "front_right",
}

DETECTION_RANGE = 50.0   # m, eight nearby slots
SECOND_RANGE = 80.0      # m, three extra forward slots

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

# risk.level: 0 if value < 0.5, 1 if 0.5 <= value < 0.7, 2 otherwise
LEVEL_THRESHOLDS = (0.5, 0.7)
TRIGGER_LEVEL = 1   # retrieval starts when a vehicle reaches this level
MAX_LEVEL = 2

LAYOUT = (
    ("front_left_front", "front_front", "front_right_front"),
    ("front_left", "front", "front_right"),
    ("left", None, "right"),
    ("rear_left", "rear", "rear_right"),
)
LAYOUT_CELLS = tuple(direction for row in LAYOUT for direction in row)
LEFT_INDEX = LAYOUT_CELLS.index("left")
RIGHT_INDEX = LAYOUT_CELLS.index("right")
RISK_NORM = float(MAX_LEVEL ** 2 * sum(direction is not None for direction in LAYOUT_CELLS))   # 44


def risk_level(value):
    low, high = LEVEL_THRESHOLDS
    if value < low:
        return 0
    if value < high:
        return 1
    return 2


# ============================================================
# Frames
# ============================================================

def upgrade_frame(frame):
    """Convert an old 8-direction frame to 11 directions (the extra slots become null)."""
    if not isinstance(frame, dict) or FRAME_KEY in frame or OLD_FRAME_KEY not in frame:
        return frame
    upgraded = {}
    for key, value in frame.items():
        if key == OLD_FRAME_KEY:
            slots = dict(value)
            upgraded[FRAME_KEY] = {direction: slots.get(direction) for direction in DIRECTIONS}
        else:
            upgraded[key] = value
    return upgraded


def parse_frame(frame):
    """JSON string or dict -> dict with the required fields."""
    raw = upgrade_frame(json.loads(frame) if isinstance(frame, str) else frame)
    for key in ("Frame", "Context", "Ego", FRAME_KEY):
        if key not in raw:
            raise ValueError("Frame is missing field: " + key)
    return raw


def frame_risk(frame):
    """(direction, value) of the highest-risk vehicle; (None, 0.0) if there is none."""
    best = (None, 0.0)
    for direction in DIRECTIONS:
        vehicle = frame[FRAME_KEY].get(direction)
        if vehicle is not None and vehicle["risk"]["value"] > best[1]:
            best = (direction, float(vehicle["risk"]["value"]))
    return best


def triggered_directions(frame):
    """Directions of real vehicles (not road boundaries) with risk level >= TRIGGER_LEVEL."""
    return [
        direction for direction in DIRECTIONS
        if frame[FRAME_KEY].get(direction) is not None
        and frame[FRAME_KEY][direction]["risk"]["level"] >= TRIGGER_LEVEL
    ]


def risk_vector(frame, lane_boundary):
    """The 4x3 risk layout as a flat list of 12 levels."""
    vector = [
        0 if direction is None or frame[FRAME_KEY].get(direction) is None
        else int(frame[FRAME_KEY][direction]["risk"]["level"])
        for direction in LAYOUT_CELLS
    ]
    if lane_boundary["left_blocked"]:
        vector[LEFT_INDEX] = MAX_LEVEL
    if lane_boundary["right_blocked"]:
        vector[RIGHT_INDEX] = MAX_LEVEL
    return vector


# ============================================================
# Left-right mirroring
# ============================================================

MIRROR_DIRECTION = {
    "front": "front", "rear": "rear", "front_front": "front_front",
    "left": "right", "right": "left",
    "front_left": "front_right", "front_right": "front_left",
    "rear_left": "rear_right", "rear_right": "rear_left",
    "front_left_front": "front_right_front", "front_right_front": "front_left_front",
}
MIRROR_ACTION = {0: 2, 1: 1, 2: 0, 3: 3, 4: 4}   # LANE_LEFT <-> LANE_RIGHT


def mirror_layout(vector):
    """Swap the left and right columns of a layout."""
    return [
        vector[i] if direction is None else vector[LAYOUT_CELLS.index(MIRROR_DIRECTION[direction])]
        for i, direction in enumerate(LAYOUT_CELLS)
    ]


def mirror_frame(frame):
    """Mirror a frame about the ego vehicle; returns a new dict and leaves the input untouched."""
    mirrored = copy.deepcopy(frame)
    axis = float(frame["Ego"]["y"])
    mirrored["Ego"]["v_y"] = -mirrored["Ego"]["v_y"]

    slots = {}
    for direction in DIRECTIONS:
        vehicle = mirrored[FRAME_KEY].get(direction)
        target = MIRROR_DIRECTION[direction]
        if vehicle is not None:
            if "direction" in vehicle:
                vehicle["direction"] = target
            vehicle["y"] = 2.0 * axis - vehicle["y"]
            vehicle["v_y"] = -vehicle["v_y"]
        slots[target] = vehicle
    mirrored[FRAME_KEY] = {direction: slots[direction] for direction in DIRECTIONS}

    if "lane_boundary" in mirrored:
        boundary = mirrored["lane_boundary"]
        boundary["left_blocked"], boundary["right_blocked"] = boundary["right_blocked"], boundary["left_blocked"]
    if "Action" in mirrored:
        mirrored["Action"] = MIRROR_ACTION[int(mirrored["Action"])]
    return mirrored
