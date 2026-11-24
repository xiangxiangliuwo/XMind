"""
Capsule bank on disk (append-only).

    bank_dir/bank.json       counters: next_id, episodes
    bank_dir/CAP000001.json  one file per capsule

Similarities are computed from each capsule's risk layout and key frame at retrieval time; no
embedding vectors are stored.
"""

import json
import os
from pathlib import Path

from XMind.scenario.directions import FRAME_KEY


CAPSULE_KEYS = (
    "capsule_id", "Context", "source", "key_frame", "risk_layout", "cluster",
    "severity", "window", "table", "safe_actions", "danger_actions",
    "uncertain_actions", "best_action", "taken_action", "discriminativeness",
    "explanation", "explanation_source", "rollout",
)
KEY_FRAME_KEYS = ("Frame", "Ego", FRAME_KEY, "lane_boundary", "Action")


def validate_capsule(capsule):
    where = capsule.get("capsule_id", "?") if isinstance(capsule, dict) else "?"
    if not isinstance(capsule, dict) or set(capsule) != set(CAPSULE_KEYS):
        raise ValueError(f"{where}: capsule must contain exactly {CAPSULE_KEYS}")
    if set(capsule["key_frame"]) != set(KEY_FRAME_KEYS):
        raise ValueError(f"{where}: key_frame must contain exactly {KEY_FRAME_KEYS}")
    if not capsule["safe_actions"] or capsule["best_action"] not in capsule["safe_actions"]:
        raise ValueError(f"{where}: best_action must be one of the safe actions")
    if not capsule["danger_actions"]:
        raise ValueError(f"{where}: a capsule needs at least one dangerous action")
    if len(capsule["risk_layout"]) != 12:
        raise ValueError(f"{where}: risk_layout must have 12 cells")
    return capsule


def _atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class CapsuleBank:
    def __init__(self, bank_dir):
        self.bank_dir = Path(bank_dir).resolve()
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.bank_dir / "bank.json"
        self.state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else {"next_id": 1, "episodes": 0}
        )
        self.capsules = {}
        for path in sorted(self.bank_dir.glob("CAP*.json")):
            capsule = validate_capsule(json.loads(path.read_text(encoding="utf-8-sig")))
            self.capsules[capsule["capsule_id"]] = capsule

    def __len__(self):
        return len(self.capsules)

    def describe(self):
        return "bank: {} | capsules {} | episodes {}".format(
            self.bank_dir.name, len(self.capsules), self.state["episodes"])

    def new_id(self):
        capsule_id = "CAP%06d" % self.state["next_id"]
        self.state["next_id"] += 1
        return capsule_id

    def add(self, capsule):
        validate_capsule(capsule)
        self.capsules[capsule["capsule_id"]] = capsule
        _atomic_write(self.bank_dir / (capsule["capsule_id"] + ".json"),
                      json.dumps(capsule, ensure_ascii=False, indent=2) + "\n")
        self.save_state()

    def note_episode(self):
        self.state["episodes"] += 1
        self.save_state()

    def save_state(self):
        _atomic_write(self.bank_dir / "bank.json", json.dumps(self.state, indent=2) + "\n")
