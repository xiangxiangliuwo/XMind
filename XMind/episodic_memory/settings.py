"""
Episodic-memory settings. The method's parameters are fixed here; config.yaml only switches
retrieval / capsule building on or off and names the capsule bank and the text-embedding model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class RolloutSettings:
    """Counterfactual rollouts (Sec. III-B)."""

    horizon_s: float = 4.0                         # T_cf, including the first action interval
    n_perturbations: int = 4                       # J traffic conditions; the first is unperturbed
    continuation_policies: Tuple[str, ...] = ("idm", "risk_greedy")   # IDM car following, greedy min-risk
    prune_radius_m: float = 150.0                  # keep vehicles within this range of the ego (0 = all)
    idm_scale: Tuple[float, float] = (0.85, 1.15)  # factors for target speed, time headway, minimum gap
    comfort_decel_scale: Tuple[float, float] = (0.8, 1.2)   # factor for comfortable deceleration
    danger_threshold: float = 0.5                  # eta: q(a) >= eta -> dangerous
    efficiency_weight: float = 0.5                 # lambda in a* = argmax_{A+} [p(a) - lambda * r(a)]
    v_ref: float = 30.0                            # m/s, progress p(a) = dx / (v_ref * T_cf)


@dataclass(frozen=True)
class CapsuleSettings:
    """Decision-critical states and capsules (Sec. III-C). Collision episodes search the decision steps
    kept in working memory (the last N = working_memory_size steps before the collision)."""

    min_peak_level: int = 1       # collision-free episodes: verify the peak state only from this level
    window_before: int = 4        # event window stored in a capsule: frames before / after the key frame
    window_after: int = 3
    reflect: bool = True          # LLM reflection (validated, template fallback); False: template only
    reflect_retries: int = 2


@dataclass(frozen=True)
class RetrievalSettings:
    """Risk-gated retrieval (Sec. III-D)."""

    match_risk_source: bool = True   # candidates must share the direction of the highest-risk vehicle
    mirror_capsules: bool = True     # also match left-right mirrored capsules
    model_path: Path = Path("../bge-m3")
    risk_weight: float = 0.7         # w in s = w * s_layout + (1 - w) * s_text (Eq. 17)
    max_diff_lines: int = 3          # layout differences listed in the prompt


@dataclass(frozen=True)
class EpisodicSettings:
    retrieve: bool = False
    evolve: bool = False
    bank_dir: Path = Path("capsule_bank")
    rollout: RolloutSettings = field(default_factory=RolloutSettings)
    capsule: CapsuleSettings = field(default_factory=CapsuleSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)

    def describe(self):
        r, c, q = self.rollout, self.capsule, self.retrieval
        return (
            "rollout: T_cf {}s, J {}, policies {}, prune {} m, eta {}, lambda {}, v_ref {} | "
            "capsule: reflect {} | "
            "retrieval: source match {}, mirror {}, text model {}, w {}"
        ).format(
            r.horizon_s, r.n_perturbations, "/".join(r.continuation_policies),
            r.prune_radius_m, r.danger_threshold, r.efficiency_weight, r.v_ref,
            "llm" if c.reflect else "template",
            "on" if q.match_risk_source else "off",
            "on" if q.mirror_capsules else "off",
            q.model_path, q.risk_weight,
        )


CONFIG_KEYS = {"retrieve", "evolve", "bank_dir"}


def _non_empty_path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    return value


def load_settings(config, project_root):
    """config.yaml (the whole dict) -> EpisodicSettings. Paths are relative to the project root."""
    em_config = dict(config.get("episodic_memory") or {})
    unknown = sorted(set(em_config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"episodic_memory: unknown keys {unknown} (allowed: {sorted(CONFIG_KEYS)})")
    for key in ("retrieve", "evolve"):
        if not isinstance(em_config.get(key, False), bool):
            raise ValueError(f"episodic_memory.{key} must be true or false")

    root = Path(project_root)
    model_path = _non_empty_path(config.get("model_path", str(RetrievalSettings.model_path)), "model_path")
    bank_dir = _non_empty_path(em_config.get("bank_dir", "capsule_bank"), "episodic_memory.bank_dir")
    return EpisodicSettings(
        retrieve=em_config.get("retrieve", False),
        evolve=em_config.get("evolve", False),
        bank_dir=(root / bank_dir).resolve(),
        retrieval=RetrievalSettings(model_path=(root / model_path).resolve()),
    )
