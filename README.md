# XMind: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving

XMind is a training-free framework that lets a frozen LLM driving agent "think twice". The agent
first decides from **working memory** (the most recent traffic frames). It then cross-checks that
decision against **episodic memory**: risk capsules recorded at critical states, in which every
candidate action was verified by counterfactual rollouts under perturbed traffic. If the retrieved
capsule prefers a different action, the agent decides a second time with that evidence.

This repository contains the closed-loop highway-env implementation used for the baseline
comparison (Q1) and the ablation study (Q3) of the paper.

## Method overview

| Paper | What happens | Code |
|---|---|---|
| Sec. III-A, Eqs. 1-5 | Each decision frame stores the ego state and one vehicle per direction slot (11 slots), with its interaction risk (DRF, or TTC for critical following) and risk level (0/1/2). The last N frames form the working memory. | `XMind/scenario/envScenario.py`, `XMind/scenario/risk_calculator.py`, `XMind/scenario/risk_field.py`, `XMind/scenario/directions.py`, `XMind/driver_agent/workingMemory.py` |
| Sec. III-B, Eqs. 6-11 | At a candidate state, every available action is executed for one decision interval under J = 4 traffic conditions, then continued by IDM car following and by a greedy minimum-risk policy. Actions are labelled recoverable / dangerous / uncertain, and the preferred action a* is selected. | `XMind/episodic_memory/rollout.py` |
| Sec. III-C, Eqs. 12-15 | After each episode, the decision-critical state is selected, verified, explained by the LLM and appended to the capsule bank. | `XMind/episodic_memory/capsule.py`, `XMind/episodic_memory/bank.py` |
| Sec. III-D, Eqs. 16-18 | Round 1 uses working memory only. When a vehicle reaches risk level 1, the best capsule is retrieved (same scenario type and same highest-risk direction, score = w * layout similarity + (1 - w) * text similarity, mirrored forms included). If its a* differs from round 1, the LLM decides again with the capsule. | `XMind/driver_agent/driverAgent.py`, `XMind/episodic_memory/retrieval.py` |

## Repository layout

```
run_XMind.py                     closed-loop simulation: episodes, memory, logs, videos
config.yaml                      LLM endpoint, seeds, traffic, memory switches
save_frames.py                   export frames of the recorded videos as PNG
requirements.txt
LICENSE                          Apache License 2.0
XMind/
  driver_agent/
    driverAgent.py               prompts and the two-round decision
    workingMemory.py             last N decision frames
  episodic_memory/
    settings.py                  all method parameters (fixed values used in the paper)
    rollout.py                   counterfactual verification and the action-outcome table
    capsule.py                   critical-state selection, verified reflection, capsule building
    bank.py                      capsule storage (one JSON file per capsule)
    retrieval.py                 risk-gated retrieval and the memory prompt
    memory.py                    EpisodicMemory entry point
  scenario/
    envScenario.py               scenario encoder (direction slots, risks, frame JSON)
    directions.py                direction slots, 4x3 risk layout, left-right mirroring
    risk_calculator.py           pairwise DRF / TTC risk
    risk_field.py                driver risk field of one vehicle
```

## Installation

Python 3.8 (tested with 3.8.20). A GPU is not needed.

```bash
pip install -r requirements.txt
```

Capsule retrieval needs a local copy of the [bge-m3](https://huggingface.co/BAAI/bge-m3) text-embedding
model. By default it is expected next to the repository (`../bge-m3`):

```bash
huggingface-cli download BAAI/bge-m3 --local-dir ../bge-m3
```

The model is only loaded when `retrieve: true`.

## Configuration

All settings are in `config.yaml`. The method parameters (rollouts, capsules, retrieval weights)
are fixed in `XMind/episodic_memory/settings.py` and are not part of the configuration.

| Key | Meaning |
|---|---|
| `OPENAI_API_TYPE` | `openai` (OpenAI or any OpenAI-compatible endpoint) or `azure` |
| `OPENAI_KEY` | API key; leave empty to use the `OPENAI_API_KEY` environment variable |
| `OPENAI_CHAT_MODEL`, `OPENAI_API_BASE` | model name and endpoint (the paper uses GPT-6-Astra) |
| `AZURE_*` | Azure OpenAI settings, used when `OPENAI_API_TYPE: azure` |
| `scene_seeds`, `episodes_num` | one episode per seed, repeated `episodes_num` times |
| `result_folder` | output goes to `<result_folder>_<timestamp>/` |
| `lanes_count`, `vehicles_density`, `vehicle_count`, `other_vehicle_type` | highway-env traffic |
| `simulation_duration` | decision steps per episode (30; one decision per second) |
| `working_memory_size` | N, frames kept in working memory (3) |
| `episodic_memory.retrieve` | retrieve capsules and decide twice when they disagree |
| `episodic_memory.evolve` | build capsules after every episode (counterfactual rollouts) |
| `episodic_memory.bank_dir` | capsule bank folder, relative to the repository |
| `model_path` | local bge-m3 folder, relative to the repository |

| `retrieve` | `evolve` | Mode |
|---|---|---|
| false | false | working memory only (one LLM decision per step) |
| false | true | build episodic memory without using it |
| true | false | XMind with a frozen memory (evaluation) |
| true | true | XMind that keeps adding capsules |

## Running

From the repository root:

```bash
python run_XMind.py
```

Each run writes `<result_folder>_<timestamp>/`:

- `log.txt`: one block per episode
  - `Simulation <i> | Seed <s> | Repeat <r> | Steps: <n> | Status: success|collision|error`
  - `Memory <i> | decision paths: {...} | A1: {...}`: how each step was decided
    (`retrieval_off`, `no_vehicle_risk`, `empty_bank`, `no_context_match`, `no_risk_source_match`,
    `memory_agrees`, `revised_with_memory`) and the capsule-building result
    (`written`, `not_discriminative`, `unavoidable_in_window`, `low_risk_peak`)
- `output.txt`: the complete console output (prompts, LLM replies, retrieval results)
- `videos/highway_<i>_seed<s>-episode-0.mp4`: the highway-env rendering of the episode
  (set `RECORD_VIDEO = False` at the top of `run_XMind.py` to disable)

Capsules are stored in `bank_dir` as `CAP000001.json`, ... together with `bank.json` (counters).

## Reproducing the experiments

1. **Build the episodic memory** on the memory-construction seeds with `evolve: true`. Each
   episode adds at most one capsule. Afterwards keep the bank fixed (`evolve: false`).
2. **Q1 (Table I)**: `retrieve: true`, `evolve: false`, `episodes_num: 3`, for each traffic
   configuration with its test seeds.
3. **Q3 (Tables IV and V)**, 5 lanes, density 3.0:

| Variant | Setting |
|---|---|
| Without memory | `working_memory_size: 1`, `retrieve: false` |
| Working memory only | `working_memory_size: 3`, `retrieve: false` |
| Text retrieval | `risk_weight = 0.0` in `RetrievalSettings` (`settings.py`) |
| Risk retrieval | `risk_weight = 1.0` in `RetrievalSettings` |
| XMind (different / matched scenarios) | bank built in another / the same traffic configuration (`bank_dir`) |
| Working memory window (Table IV) | `working_memory_size: 1 / 3 / 5`, window seeds below |

Seeds used in the paper:

```yaml
# Q1 test seeds, 5 lanes, density 3.0
[6932, 2779, 386, 1543, 2187, 72107, 9355, 7800821, 693, 914, 588, 3654, 8216, 4687, 637, 3412104, 694354, 5625983, 8091806, 8579079, 8774675, 5166103, 2147231, 2897107, 2582682, 5096435, 456178, 9059526, 8620879, 3641679]
# Q1 test seeds, 5 lanes, density 2.5
[6932, 2709, 376, 1543, 2089, 721007, 9355, 7808210, 693, 984, 589, 3647, 8206, 4697, 637, 3412104, 694354, 5625983, 8091806, 8579079, 8774675, 5166103, 214723, 2897107, 2582682, 5096435, 456178, 9059526, 8620879, 3641679]
# Q1 test seeds, 4 lanes, density 2.5
[6932, 2698, 376, 11453, 7302, 712070, 9355, 771228, 693, 984, 577, 3275, 8193, 4399, 627, 341210, 64354, 525983, 801806, 857079, 877475, 516103, 14723, 897107, 258682, 596435, 46178, 909526, 860879, 361679]
# memory construction
[6866371, 1211463, 462719, 6267180, 6992283, 8222120, 8696622, 4892351, 3421479, 6600245, 686371, 121463, 46719, 627180, 692283, 822120, 869622, 489351, 342479, 660245, 68466371, 12114623, 4627129, 62672180, 69922283, 82222120, 86926622, 48923521, 34214729, 66002452]
# Q3 component ablation (Table V)
[6932, 27018, 1646580, 1543, 2330150, 72007, 9355, 7800821, 6937, 9874, 5879, 36477, 82706, 46797, 3726688, 69172, 270119, 16416580, 15143, 23310150, 720107, 93155, 78010821, 69137, 98174, 58179, 364177, 827107, 467971, 37266188]
# Q3 working memory window (Table IV)
[6932, 27018, 1646580, 1543, 2330150, 72007, 9355, 7800821, 6937, 9874, 5879, 36477, 82706, 46797, 3726688]
```

LLM outputs are not fully deterministic, even at temperature 0, so repeated runs can differ slightly.

## Parameters

Fixed in the code (paper values first):

| Parameter | Value | Location |
|---|---|---|
| Risk levels (Eq. 4) | level 1 from 0.5, level 2 from 0.7 | `directions.LEVEL_THRESHOLDS` |
| TTC instead of DRF (Eq. 3) | leader slower than 5 m/s and closing speed above 5 m/s | `risk_calculator.TTC_*` |
| Retrieval trigger | a vehicle at risk level >= 1 | `directions.TRIGGER_LEVEL` |
| J, T_cf (Sec. III-B) | 4 conditions, 4 s | `RolloutSettings` |
| eta, lambda (Eqs. 10-11) | 0.5, 0.5 | `RolloutSettings` |
| v_ref (Eq. 9) | 30 m/s | `RolloutSettings` |
| Peak-state verification (Sec. III-C) | risk level >= 1 | `CapsuleSettings.min_peak_level` |
| w (Eq. 17) | 0.7 | `RetrievalSettings.risk_weight` |
| LLM | temperature 0, at most 2000 output tokens | `driverAgent.py` |

## Implementation details

Details that the paper leaves open, as implemented here:

- **Direction slots**: the eight nearby slots hold the nearest vehicle within 50 m in the ego lane
  or an adjacent lane (|dx| < 2.5 m counts as beside); the three extra forward slots hold the next
  vehicle ahead of front / front-left / front-right within 80 m.
- **DRF**: field regions are the grid cells (1 m) with intensity >= 0.01; the trailing vehicle's
  field is the reference. For a vehicle in an adjacent lane the ego field is evaluated as if the
  ego vehicle were shifted laterally onto that lane's centre line. TTC uses the bumper-to-bumper gap
  and the closing speed along the lane; it applies to the front and front-front slots.
- **Risk layout**: a missing adjacent lane sets the beside cell (left or right) to level 2.
  Layout similarity is 1 - ||rho_t - rho_m||^2 / 44 (11 vehicle cells x 2^2), clipped at 0.
- **Traffic conditions**: condition 1 is unchanged; conditions 2-4 scale the IDM parameters of every
  surrounding vehicle (target speed, desired time headway, minimum gap by U[0.85, 1.15]; comfortable
  deceleration by U[0.8, 1.2]), drawn reproducibly from (scene seed, frame). Rollouts keep vehicles
  within 150 m of the ego vehicle.
- **Continuation policies**: IDM car following (the ego vehicle becomes an IDM vehicle without lane
  changes) and a one-step greedy policy choosing the meta-action with the lowest maximum risk.
- **Decision-critical state**: collision episodes search the N decision steps kept in working memory
  before the collision. The capsule's event window holds 4 decision frames before and 3 after it.
- **Reflection**: the LLM explanation is accepted only if it names one of the two highest-risk
  directions and recommends a recoverable action; otherwise the LLM is asked again (2 retries) and a
  template explanation built from the verified table is used as the last resort.
- **Text similarity**: cosine similarity (clipped at 0) of bge-m3 embeddings of one sentence that
  describes the highest-risk vehicle (direction, relative position, relative velocity, acceleration,
  ego speed).
- **Memory prompt**: besides the action classes, preferred action, peak risk and reflection, the
  capsule block lists the largest layout differences from the current frame and warns when the
  recommended target area is riskier now than in the capsule.
## Saving video frames

```bash
python save_frames.py results_xxx/videos/highway_0_seed6932-episode-0.mp4 --sheet   # one thumbnail per decision step
python save_frames.py VIDEO --steps 0 5 12                                          # selected decision steps
python save_frames.py VIDEO --interactive                                           # browse and save
```

Run `python save_frames.py -h` for all options.

## Not included

- Evaluation metrics (SR, SS, mean speed, minimum TTC, token usage). Each episode's outcome and
  number of steps are in its `Simulation` line of `log.txt`.
- The K-Risk real-data evaluation (Q2).
- The baselines (DiLu, KoMA, RESPOND).
- The "w/o counterfactual verification" ablation, in which the LLM proposes the preferred action
  by reflection.

## License

This project is released under the Apache License 2.0; see [LICENSE](LICENSE).

## Acknowledgements

The simulation uses [highway-env](https://github.com/Farama-Foundation/HighwayEnv). Parts of
`run_XMind.py`, `XMind/driver_agent/driverAgent.py` and `XMind/scenario/envScenario.py` are adapted
from [DiLu](https://github.com/PJLab-ADG/DiLu) (Apache License 2.0) and modified for XMind. The
driver risk field follows RESPOND.

## Citation

Citation information will be added when the paper is published.
