"""
Risk-gated capsule retrieval (Sec. III-D).

Gate        retrieve only if a real vehicle has risk level >= 1.
Candidates  same scenario type, and (match_risk_source) the same direction of the highest-risk
            vehicle as the current frame. With mirror_capsules, every capsule also competes in its
            left-right mirrored form; each capsule keeps its better form (the original on a tie).
Score       s = w * s_layout + (1 - w) * s_text (Eq. 17), with
                s_layout = max(0, 1 - ||rho_t - rho_m||^2 / 44)
                s_text   = max(0, cos) of bge-m3 embeddings of a one-sentence description of the
                           highest-risk vehicle (direction, dx, dy, dvx, dvy, a, ego vx)
            The capsule with the highest score is the memory evidence B_t (ties: lower capsule id).
Prompt      action classes, preferred action, risk, reflection, layout differences and a warning when
            the recommended target area is riskier now. A mirrored capsule is reported after mirroring;
            only its explanation keeps the original wording.
"""

import copy
from collections import namedtuple
from pathlib import Path

import numpy as np

from XMind.scenario.directions import (
    ACTION_NAMES,
    FRAME_KEY,
    LAYOUT_CELLS,
    MIRROR_ACTION,
    RISK_NORM,
    frame_risk,
    mirror_frame,
    mirror_layout,
    parse_frame,
    risk_vector,
    triggered_directions,
)

ACTION_IDS = {name: action for action, name in ACTION_NAMES.items()}

# One form of a capsule taking part in retrieval: original (mirrored=False) or mirrored.
CapsuleView = namedtuple("CapsuleView", ["capsule_id", "mirrored", "key_frame", "layout"])

# Layout cells an action moves into (for applicability warnings).
TARGET_CELLS = {
    0: ("front_left", "left", "rear_left"),      # LANE_LEFT
    1: ("front", "rear"),                        # IDLE
    2: ("front_right", "right", "rear_right"),   # LANE_RIGHT
    3: ("front", "front_left", "front_right"),   # FASTER
    4: ("rear",),                                # SLOWER
}


def main_risk_direction(frame):
    return frame_risk(frame)[0]


def _num(value):
    """One decimal; -0.0 is written as 0.0 so that mirrored texts match."""
    return "{:.1f}".format(round(float(value), 1) + 0.0)


def state_text(frame):
    """Description of the highest-risk vehicle only; None if the frame has no vehicle."""
    direction = main_risk_direction(frame)
    if direction is None:
        return None
    ego = frame["Ego"]
    other = frame[FRAME_KEY][direction]
    return (
        f"The main risk vehicle is at your {direction.replace('_', ' ')}. "
        f"Its relative position is dx={_num(other['x'] - ego['x'])} m, dy={_num(other['y'] - ego['y'])} m. "
        f"Its relative velocity is dvx={_num(other['v_x'] - ego['v_x'])} m/s, "
        f"dvy={_num(other['v_y'] - ego['v_y'])} m/s. "
        f"Its acceleration is a={_num(other['a'])} m/s^2. "
        f"Your speed is vx={_num(ego['v_x'])} m/s."
    )


class TextEncoder:
    """Local bge-m3 (sentence-transformers) encoder returning unit vectors, cached per text.
    The model is loaded at construction so that a wrong path fails before the experiment starts."""

    def __init__(self, model_path, batch_size=32):
        self.model_path = Path(model_path)
        self.batch_size = int(batch_size)
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Missing local text-embedding model: {self.model_path} "
                "(set model_path in config.yaml to the bge-m3 folder)"
            )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self.model_path), local_files_only=True)
        self._cache = {}

    def encode(self, texts):
        """texts -> (n, d) float64 unit vectors."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            vectors = self._model.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float64)
        return np.stack([self._cache[text] for text in texts])


def text_similarity(encoder, query, memories):
    """max(0, cos) for each memory text; identical texts give 1. None (no vehicle) matches only None."""
    values = np.zeros(len(memories), dtype=np.float64)
    if query is None:
        values[[i for i, memory in enumerate(memories) if memory is None]] = 1.0
        return values
    present = [i for i, memory in enumerate(memories) if memory is not None]
    if present:
        vectors = encoder.encode([query] + [memories[i] for i in present])
        values[present] = np.clip(vectors[1:] @ vectors[0], 0.0, 1.0)
        values[[i for i in present if memories[i] == query]] = 1.0
    return values


def layout_similarity(query, layouts):
    diff = np.asarray(layouts, dtype=np.float64) - np.asarray(query, dtype=np.float64)
    return np.clip(1.0 - np.square(diff).sum(axis=-1) / RISK_NORM, 0.0, 1.0)


def layout_differences(query, memory, limit):
    """[(direction, level now, level in memory)] for differing cells, largest difference first."""
    rows = [
        (direction, int(now), int(then))
        for direction, now, then in zip(LAYOUT_CELLS, query, memory)
        if direction is not None and now != then
    ]
    rows.sort(key=lambda row: (-abs(row[1] - row[2]), LAYOUT_CELLS.index(row[0])))
    return rows[:limit]


def applicability_warnings(query, memory, action):
    """Target cells of `action` whose level is higher now than in memory."""
    now = dict(zip(LAYOUT_CELLS, query))
    then = dict(zip(LAYOUT_CELLS, memory))
    return [
        (direction, int(now[direction]), int(then[direction]))
        for direction in TARGET_CELLS[action]
        if now[direction] > then[direction]
    ]


def mirror_table(table):
    """Mirrored action-outcome table: the LANE_LEFT and LANE_RIGHT rows swap; action lists and a*
    are mapped with MIRROR_ACTION. Returns a new dict."""
    mirrored = copy.deepcopy(table)
    rows = {}
    for name, row in table["actions"].items():
        new_name = ACTION_NAMES[MIRROR_ACTION[ACTION_IDS[name]]]
        row = copy.deepcopy(row)
        if "action" in row:
            row["action"] = ACTION_IDS[new_name]
        if "name" in row:
            row["name"] = new_name
        rows[new_name] = row
    mirrored["actions"] = {ACTION_NAMES[a]: rows[ACTION_NAMES[a]] for a in sorted(ACTION_NAMES) if ACTION_NAMES[a] in rows}
    for key in ("safe_actions", "danger_actions", "uncertain_actions"):
        if key in table:
            mirrored[key] = sorted(MIRROR_ACTION[a] for a in table[key])
    if table.get("best_action") is not None:
        mirrored["best_action"] = MIRROR_ACTION[table["best_action"]]
    return mirrored


class CapsuleRetriever:
    def __init__(self, bank, settings, encoder=None):
        """encoder: anything with encode(texts) -> unit vectors; None loads bge-m3 from settings.model_path."""
        self.bank = bank
        self.settings = settings
        self.encoder = encoder if encoder is not None else TextEncoder(settings.model_path)

    def retrieve(self, frame, *, lane_boundary):
        current = parse_frame(frame)
        if lane_boundary["lane_id"] != current["Ego"]["lane_id"]:
            raise ValueError("Lane boundary and frame snapshot disagree")

        query_layout = risk_vector(current, lane_boundary)
        triggered = triggered_directions(current)
        source = main_risk_direction(current)
        result = {
            "enabled": bool(triggered),
            "reason": "no_vehicle_risk",
            "Context": current["Context"],
            "query_frame": current["Frame"],
            "trigger_directions": triggered,
            "risk_source": source,
            "risk_layout": query_layout,
            "candidates": 0,
            "matches": [],
            "capsule_ids": [],
            "capsules": [],
            "best_action": None,
        }
        if not triggered:
            return result

        ids = sorted(cid for cid, c in self.bank.capsules.items() if c["Context"] == current["Context"])
        if not ids:
            result["reason"] = "empty_bank" if not self.bank.capsules else "no_context_match"
            return result

        views = [view for cid in ids for view in self._views(cid, source)]
        if not views:
            result["reason"] = "no_risk_source_match"
            return result

        row, view = self._best(current, query_layout, views)
        entry = self._entry(row, view, query_layout)
        result.update({
            "reason": "retrieved",
            "candidates": len({view.capsule_id for view in views}),
            "matches": [row],
            "capsule_ids": [row["capsule_id"]],
            "capsules": [entry],
            "best_action": entry["best_action"],
        })
        return result

    def _best(self, current, query_layout, views):
        """(score row, form) of the highest-scoring capsule; each capsule competes with its better form."""
        s_layout = layout_similarity(query_layout, [view.layout for view in views])
        s_text = text_similarity(self.encoder, state_text(current), [state_text(view.key_frame) for view in views])
        w = self.settings.risk_weight
        chosen = {}
        for view, layout_value, text_value in zip(views, s_layout, s_text):
            row = {
                "capsule_id": view.capsule_id,
                "mirrored": view.mirrored,
                "text_similarity": float(text_value),
                "layout_similarity": float(layout_value),
                "score": float((1.0 - w) * text_value + w * float(layout_value)),
            }
            if view.capsule_id not in chosen or row["score"] > chosen[view.capsule_id][0]["score"]:
                chosen[view.capsule_id] = (row, view)
        return min(chosen.values(), key=lambda item: (-item[0]["score"], item[0]["capsule_id"]))

    def _views(self, capsule_id, source):
        capsule = self.bank.capsules[capsule_id]
        views = [CapsuleView(capsule_id, False, capsule["key_frame"], list(capsule["risk_layout"]))]
        if self.settings.mirror_capsules:
            views.append(CapsuleView(capsule_id, True, mirror_frame(capsule["key_frame"]),
                                     mirror_layout(capsule["risk_layout"])))
        if self.settings.match_risk_source:
            views = [view for view in views if main_risk_direction(view.key_frame) == source]
        return views

    def _entry(self, row, view, query_layout):
        """Everything the prompt needs about one capsule, in the orientation of the current frame."""
        capsule = self.bank.capsules[row["capsule_id"]]
        mirrored = view.mirrored
        best = MIRROR_ACTION[capsule["best_action"]] if mirrored else capsule["best_action"]
        return {
            "capsule_id": capsule["capsule_id"],
            "score": row["score"],
            "mirrored": mirrored,
            "key_frame": view.key_frame,
            "collision": capsule["source"]["collision"],
            "table": mirror_table(capsule["table"]) if mirrored else capsule["table"],
            "best_action": best,
            "original_best_action": capsule["best_action"],
            "explanation": capsule["explanation"],
            "differences": layout_differences(query_layout, view.layout, self.settings.max_diff_lines),
            "warnings": applicability_warnings(query_layout, view.layout, best),
        }

    # ---------------- prompt ----------------

    @staticmethod
    def to_prompt(result):
        if not result["capsules"]:
            return ""
        blocks = [
            "#### Retrieved risk memory (verified by simulation)\n"
            "Each capsule below is a past situation similar to the current one. At its key frame, every "
            "action was replayed in a simulator under several variations of the surrounding traffic:\n"
            "SAFE = recoverable in every variation; DANGEROUS = a collision could not be avoided in at "
            "least half of the variations; UNCERTAIN = in between.\n"
            "Risk levels: 0 low, 1 medium, 2 high; at a road boundary the left/right cell is 2."
        ]
        if any(entry["mirrored"] for entry in result["capsules"]):
            blocks.append(
                "A capsule marked MIRRORED is a past situation on the other side of the road: its directions, "
                "lateral motion and LANE_LEFT / LANE_RIGHT were swapped to match the current situation, so its "
                "Situation, SAFE / DANGEROUS / UNCERTAIN, differences and warnings already refer to the current "
                "side. Only its Explanation keeps the original wording, which describes the situation before "
                "mirroring (left and right reversed)."
            )
        blocks.extend(CapsuleRetriever._capsule_block(entry) for entry in result["capsules"])
        blocks.append(
            "Use these memories as references. The current observation always has priority: check the "
            "differences and warnings before following a recommendation."
        )
        return "\n\n".join(blocks)

    @staticmethod
    def _capsule_block(entry):
        table = entry["table"]
        actions = table["actions"]
        n = table["n_perturbations"]
        key = entry["key_frame"]
        direction, value = frame_risk(key)
        best = ACTION_NAMES[entry["best_action"]]

        def names(label):
            return [name for name, row in actions.items() if row["label"] == label]

        def unrecoverable(row):
            return sum(not p["recoverable"] for p in row["per_perturbation"])

        safe = [
            f"{name} ({'RECOMMENDED; ' if name == best else ''}peak risk {actions[name]['r_hat']:.2f}, "
            f"progress {actions[name]['p']:.2f})"
            for name in names("safe")
        ]
        danger = []
        for name in names("danger"):
            row = actions[name]
            when = f", collision after ~{row['crash_time']:.1f} s" if row["crash_time"] is not None else ""
            danger.append(f"{name} (unavoidable collision in {unrecoverable(row)}/{n} variations{when})")
        uncertain = [
            f"{name} (unavoidable collision in {unrecoverable(actions[name])}/{n} variations)"
            for name in names("uncertain")
        ]

        source = f"main risk from the {direction} vehicle (risk {value:.2f})" if direction else "no nearby vehicle"
        if entry["mirrored"]:
            original = ACTION_NAMES[entry["original_best_action"]]
            header = f"[Capsule {entry['capsule_id']} | MIRRORED: left and right swapped | similarity {entry['score']:.2f} | "
            situation = f"Situation at its key frame (after mirroring): {source}; ego speed {key['Ego']['v_x']:.1f} m/s."
            explanation = (
                "Explanation (original text, written for the situation BEFORE mirroring: read its left as right "
                f"and its right as left; its recommended {original} corresponds to {best} here): "
                + entry["explanation"]
            )
        else:
            header = f"[Capsule {entry['capsule_id']} | similarity {entry['score']:.2f} | "
            situation = (
                f"Situation at its key frame: {source}; ego speed {key['Ego']['v_x']:.1f} m/s "
                f"in lane {key['Ego']['lane_id']}."
            )
            explanation = "Explanation: " + entry["explanation"]

        lines = [
            header + f"{'the episode crashed' if entry['collision'] else 'no collision'}]",
            situation,
            "SAFE: " + ("; ".join(safe) if safe else "none"),
            "DANGEROUS: " + ("; ".join(danger) if danger else "none"),
            "UNCERTAIN: " + ("; ".join(uncertain) if uncertain else "none"),
            explanation,
        ]
        if entry["differences"]:
            lines.append("Differences from now: " + "; ".join(
                f"{d}: level {now} now vs {then} in memory" for d, now, then in entry["differences"]))
        if entry["warnings"]:
            lines.append(f"WARNING: the target lane/area of {best} is riskier now: " + "; ".join(
                f"{d} level {now} now vs {then} in memory" for d, now, then in entry["warnings"]))
        return "\n".join(lines)
