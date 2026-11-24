"""
Counterfactual risk capsules (Sec. III-C): after an episode, find the decision-critical state tau*,
verify its actions with rollouts, write a validated reflection and append the capsule to the bank.

    recorder = builder.start_episode(episode, seed, n_frames)   # n_frames = working-memory size N
    recorder.record(frame_json, lane_boundary, action, env)   # every step, before env.step
    summary  = builder.end_episode(recorder, crashed, llm)     # after the episode

Decision-critical state
    collision: search backwards over the decision steps kept in working memory (the last N steps
               before the collision) and take the latest one with a recoverable action; none -> no capsule.
    no collision: the earliest step with the episode's maximum directional risk, verified only if
               its risk level reaches min_peak_level.
A capsule is written only if tau* has both recoverable and dangerous actions.
"""

import json
import re
from collections import deque

from XMind.episodic_memory.rollout import ACTIONS, evaluate_frame, take_snapshot
from XMind.scenario.directions import (
    ACTION_NAMES,
    DIRECTIONS,
    FRAME_KEY,
    frame_risk,
    parse_frame,
    risk_level,
    risk_vector,
)


# ============================================================
# Episode recording (frames + snapshots)
# ============================================================

class EpisodeRecorder:
    """Decision frames of one episode plus snapshots of the candidate key states: the last
    `backtrack_frames` steps (collision episodes) and the peak-risk step (collision-free episodes)."""

    def __init__(self, episode, seed, backtrack_frames):
        self.episode = int(episode)
        self.seed = int(seed)
        self.frames = []                                       # Frame / Ego / 11 Directions / lane_boundary / Action
        self.recent = deque(maxlen=max(1, backtrack_frames))   # (Frame, snapshot)
        self.peak = None                                       # (Frame, risk, snapshot)

    def record(self, frame_json, lane_boundary, action, env):
        """Call before env.step, so that the environment is in the state of this frame."""
        frame = parse_frame(frame_json)
        if lane_boundary["lane_id"] != frame["Ego"]["lane_id"]:
            raise ValueError("Lane boundary and frame snapshot disagree")
        if self.frames and frame["Frame"] <= self.frames[-1]["Frame"]:
            raise ValueError("Frames must be recorded in increasing order")

        self.frames.append({
            "Frame": int(frame["Frame"]),
            "Ego": frame["Ego"],
            FRAME_KEY: frame[FRAME_KEY],
            "lane_boundary": {
                "lane_id": int(lane_boundary["lane_id"]),
                "left_blocked": bool(lane_boundary["left_blocked"]),
                "right_blocked": bool(lane_boundary["right_blocked"]),
            },
            "Action": int(action),
        })
        snapshot = take_snapshot(env)
        self.recent.append((frame["Frame"], snapshot))
        _, risk = frame_risk(frame)
        if self.peak is None or risk > self.peak[1]:
            self.peak = (frame["Frame"], risk, snapshot)

    def frame(self, frame_id):
        for frame in self.frames:
            if frame["Frame"] == frame_id:
                return frame
        raise KeyError(frame_id)


# ============================================================
# Verified reflection
# ============================================================

REFLECT_SYSTEM = (
    "You explain a verified counterfactual analysis of a highway driving event. "
    "Every action at the key frame was replayed in a simulator under several variations of the "
    "surrounding traffic, so the safe / dangerous labels are facts. Do not propose any other action."
)


def top_risk_directions(frame, n=2):
    occupied = [
        (direction, frame[FRAME_KEY][direction]["risk"]["value"])
        for direction in DIRECTIONS
        if frame[FRAME_KEY].get(direction) is not None
    ]
    occupied.sort(key=lambda item: -item[1])
    return [direction for direction, _ in occupied[:n]]


def _table_lines(table):
    lines = []
    for action in ACTIONS:
        o = table.outcomes[action]
        if not o.available:
            continue
        line = (
            f"- {o.name}: {o.label}; unrecoverable in {o.n_unrecoverable}/{table.n_perturbations} "
            f"traffic variations (c={o.c:.2f}); peak risk {o.r_hat:.2f}; progress {o.p:.2f}"
        )
        if o.crash_time is not None:
            line += f"; earliest collision after {o.crash_time:.1f} s"
        lines.append(line)
    return lines


def _compact_slots(frame):
    ego = frame["Ego"]
    return [
        f"{direction}: risk {vehicle['risk']['value']:.2f} (level {vehicle['risk']['level']}), "
        f"dx {vehicle['x'] - ego['x']:.1f} m, dvx {vehicle['v_x'] - ego['v_x']:.1f} m/s, "
        f"vy {vehicle['v_y']:.1f} m/s"
        for direction, vehicle in ((d, frame[FRAME_KEY].get(d)) for d in DIRECTIONS)
        if vehicle is not None
    ]


def reflection_prompt(window, key_frame, table, crashed):
    taken = key_frame["Action"]
    window_lines = [
        f"frame {row['Frame']}: action {row['action_name']}, max risk {row['max_risk']:.2f} ({row['max_direction']})"
        for row in window
    ]
    boundary = key_frame["lane_boundary"]
    return "\n".join([
        f"Episode outcome: {'collision' if crashed else 'no collision'}.",
        "Event window (decision frames around the key frame):",
        *window_lines,
        f"Key frame {key_frame['Frame']}: ego speed {key_frame['Ego']['v_x']:.1f} m/s, lane {key_frame['Ego']['lane_id']}, "
        f"left lane {'missing' if boundary['left_blocked'] else 'present'}, "
        f"right lane {'missing' if boundary['right_blocked'] else 'present'}.",
        "Vehicles at the key frame:",
        *_compact_slots(key_frame),
        "Verified action-outcome table at the key frame:",
        *_table_lines(table),
        f"Action actually taken: {ACTION_NAMES[taken]}. Verified best action: {ACTION_NAMES[table.best_action]}.",
        "",
        "Reply with a JSON object only:",
        '{"risk_source": "<direction of the main risk vehicle>", '
        '"recommended_action": "<one safe action name>", '
        '"explanation": "<at most 3 sentences: (1) which vehicle / relative motion causes the risk, '
        '(2) why the taken action was dangerous or not, (3) why the recommended action is safe>"}',
    ])


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)


def check_reflection(reply, key_frame, table):
    """risk_source must be one of the two highest-risk directions; recommended_action must be in A+."""
    match = re.search(r"\{.*\}", _strip_think(reply), flags=re.S)
    if match is None:
        raise ValueError("no JSON object in the reply")
    data = json.loads(match.group(0))
    allowed = top_risk_directions(key_frame, 2)
    if data.get("risk_source") not in allowed:
        raise ValueError(f"risk_source must be one of {allowed}")
    safe_names = [ACTION_NAMES[a] for a in table.safe]
    if data.get("recommended_action") not in safe_names:
        raise ValueError(f"recommended_action must be one of {safe_names}")
    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("explanation is empty")
    return explanation


def template_explanation(key_frame, table):
    direction, value = frame_risk(key_frame)
    taken = table.outcomes[key_frame["Action"]]
    best = table.outcomes[table.best_action]
    source = f"the {direction} vehicle (risk {value:.2f})" if direction else "no nearby vehicle"
    if not taken.available:
        taken_text = f"{taken.name} was not available"
    elif taken.label == "safe":
        taken_text = f"{taken.name} was also safe"
    else:
        taken_text = f"{taken.name} was unrecoverable in {taken.n_unrecoverable}/{table.n_perturbations} traffic variations"
    return (
        f"Main risk source: {source}. Taken action: {taken_text}. "
        f"Recommended {best.name}: recoverable in all {table.n_perturbations} traffic variations, "
        f"peak risk {best.r_hat:.2f}, progress {best.p:.2f}."
    )


def ask_llm(llm, system, user):
    from langchain.schema import HumanMessage, SystemMessage

    reply = llm([SystemMessage(content=system), HumanMessage(content=user)]).content
    return _strip_think(reply).strip()


# ============================================================
# Capsule building
# ============================================================

class CapsuleBuilder:
    def __init__(self, bank, settings, log=print):
        self.bank = bank
        self.settings = settings
        self.log = log

    def start_episode(self, episode, seed, n_frames):
        """n_frames: working-memory size N; snapshots of the last N decision steps are kept."""
        return EpisodeRecorder(episode, seed, n_frames)

    def _decision_critical_frame(self, recorder, crashed):
        """(tau*, action-outcome table, status); tau* is None when no capsule can be built."""
        rollout = self.settings.rollout
        if crashed:
            for frame_id, snapshot in reversed(recorder.recent):   # latest first
                table = evaluate_frame(snapshot, frame_id, recorder.seed, rollout)
                self.log(f"[A1] {table.summary()}")
                if table.safe:
                    return frame_id, table, "found"
            return None, None, "unavoidable_in_window"

        frame_id, risk, snapshot = recorder.peak
        if risk_level(risk) < self.settings.capsule.min_peak_level:
            return None, None, "low_risk_peak"
        table = evaluate_frame(snapshot, frame_id, recorder.seed, rollout)
        self.log(f"[A1] {table.summary()}")
        return frame_id, table, "found"

    def end_episode(self, recorder, crashed, llm=None):
        self.bank.note_episode()
        if not recorder.frames:
            return {"status": "no_frames"}

        tau, table, status = self._decision_critical_frame(recorder, crashed)
        if tau is None:
            return {"status": status}

        summary = {
            "status": status,
            "key_frame": tau,
            "safe": [ACTION_NAMES[a] for a in table.safe],
            "danger": [ACTION_NAMES[a] for a in table.danger],
            "best_action": ACTION_NAMES[table.best_action] if table.best_action is not None else None,
            "branches": table.n_branches,
        }
        if not table.safe or not table.danger:
            summary["status"] = "not_discriminative"
            return summary

        capsule = self._assemble(recorder, tau, table, crashed, llm)
        self.bank.add(capsule)
        summary.update({"status": "written", "capsule_id": capsule["capsule_id"],
                        "explanation_source": capsule["explanation_source"]})
        return summary

    def _window(self, recorder, tau):
        ids = [frame["Frame"] for frame in recorder.frames]
        center = ids.index(tau)
        low = max(0, center - self.settings.capsule.window_before)
        high = min(len(ids), center + self.settings.capsule.window_after + 1)
        window = []
        for frame in recorder.frames[low:high]:
            direction, value = frame_risk(frame)
            window.append({
                "Frame": frame["Frame"],
                "Action": frame["Action"],
                "action_name": ACTION_NAMES[frame["Action"]],
                "max_risk": round(value, 4),
                "max_direction": direction,
            })
        return window

    def _explain(self, window, key_frame, table, crashed, llm):
        """LLM reflection checked by check_reflection; rewritten with the reason on failure, template last."""
        if not self.settings.capsule.reflect or llm is None:
            return template_explanation(key_frame, table), "template"
        user = reflection_prompt(window, key_frame, table, crashed)
        for attempt in range(self.settings.capsule.reflect_retries + 1):
            try:
                return check_reflection(ask_llm(llm, REFLECT_SYSTEM, user), key_frame, table), "llm"
            except Exception as error:   # never stop the experiment because of a bad reflection
                self.log(f"[A1] reflection rejected (attempt {attempt + 1}): {error}")
                user += f"\n\nYour previous reply was rejected ({error}). Reply with the JSON object only."
        return template_explanation(key_frame, table), "template"

    def _assemble(self, recorder, tau, table, crashed, llm):
        key_frame = recorder.frame(tau)
        window = self._window(recorder, tau)
        layout = risk_vector(key_frame, key_frame["lane_boundary"])
        explanation, source = self._explain(window, key_frame, table, crashed, llm)
        rollout = self.settings.rollout
        return {
            "capsule_id": self.bank.new_id(),
            "Context": "highway-v0",
            "source": {
                "episode": recorder.episode,
                "seed": recorder.seed,
                "collision": bool(crashed),
                "lucky": bool(not crashed and key_frame["Action"] not in table.safe),
            },
            "key_frame": key_frame,
            "risk_layout": layout,
            "cluster": "".join(str(v) for v in layout),
            "severity": 1.0 if crashed else round(table.severity, 4),
            "window": window,
            "table": table.to_dict(),
            "safe_actions": table.safe,
            "danger_actions": table.danger,
            "uncertain_actions": table.uncertain,
            "best_action": table.best_action,
            "taken_action": key_frame["Action"],
            "discriminativeness": round(table.discriminativeness, 4),
            "explanation": explanation,
            "explanation_source": source,
            "rollout": {
                "horizon_s": table.horizon_s,
                "n_perturbations": rollout.n_perturbations,
                "continuation_policies": list(rollout.continuation_policies),
                "danger_threshold": rollout.danger_threshold,
                "efficiency_weight": rollout.efficiency_weight,
                "v_ref": rollout.v_ref,
                "prune_radius_m": rollout.prune_radius_m,
            },
        }
