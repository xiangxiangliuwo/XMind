# XMind🧠: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving

XMind is a training-free framework that lets an LLM driving agent "think twice". The agent first
decides from working memory of recent traffic states, then cross-checks the decision against
episodic memory of critical risk events, in which alternative actions were verified by
counterfactual rollouts. If the two disagree, it decides again with the retrieved evidence.

## 🚀 Getting Started

### 1. Requirements 📦

We recommend a conda environment with Python 3.8:

```bash
conda create -n xmind python=3.8
conda activate xmind
pip install -r requirements.txt
```

XMind depends on specific library versions (e.g. `langchain==0.0.335`, `openai==0.28.1`,
`gymnasium==0.29.1`, `highway-env==1.8.2`); please install exactly the versions in `requirements.txt`.

Capsule retrieval uses the [bge-m3](https://huggingface.co/BAAI/bge-m3) text-embedding model. Download it
next to the repository (the default `model_path` is `../bge-m3`):

```bash
huggingface-cli download BAAI/bge-m3 --local-dir ../bge-m3
```

### 2. Configuration ⚙️

All settings are in `config.yaml`.

- `OPENAI_API_TYPE`: `openai` or `azure`
- `OPENAI_KEY`: your API key (or leave it empty and set the `OPENAI_API_KEY` environment variable)
- `OPENAI_CHAT_MODEL`: the chat model (the paper uses `gpt-6-astra`)
- `OPENAI_API_BASE`: an OpenAI-compatible endpoint (leave empty for the default OpenAI endpoint)
- `AZURE_API_BASE`, `AZURE_API_VERSION`, `AZURE_API_KEY`, `AZURE_CHAT_DEPLOY_NAME`: Azure OpenAI settings
- `scene_seeds`, `episodes_num`: one episode per seed, repeated `episodes_num` times
- `lanes_count`, `vehicles_density`, `simulation_duration`: highway-env traffic and episode length
- `working_memory_size`: number of recent frames in working memory (3 in the paper)
- `episodic_memory.retrieve`: retrieve risk capsules and decide twice when they disagree
- `episodic_memory.evolve`: build new risk capsules after every episode
- `episodic_memory.bank_dir`: folder of the capsule bank
- `model_path`: local bge-m3 folder (only loaded when `retrieve` is `true`)

The method parameters (counterfactual rollouts, capsule selection, retrieval weight) are fixed in
`XMind/episodic_memory/settings.py`.

### 3. Running XMind 🧠

```bash
python run_XMind.py
```

The script runs one episode for each seed in `scene_seeds`. Results are saved in
`results_<timestamp>/`, which contains `log.txt` (outcome, steps and decision paths of each episode),
`output.txt` (the full console output, including prompts and LLM replies) and the simulation videos.

**Note:** while videos are recorded, highway-env opens a pygame window that may appear unresponsive.
This is normal as long as the terminal keeps printing output.

#### Build episodic memory

Set `evolve: true` in `config.yaml`. After every episode XMind verifies the decision-critical state
with counterfactual rollouts and saves a risk capsule (`CAP000001.json`, ...) in `bank_dir`.
To evaluate with a fixed memory, set `evolve: false` and `retrieve: true`.

### 4. Visualizing Results 📊

Export frames of a recorded video as PNG images:

```bash
python save_frames.py results_xxx/videos/highway_0_seed6932-episode-0.mp4 --sheet   # one thumbnail per decision step
python save_frames.py results_xxx/videos/highway_0_seed6932-episode-0.mp4 --steps 0 5 12
python save_frames.py results_xxx/videos/highway_0_seed6932-episode-0.mp4 --interactive
```

Run `python save_frames.py -h` for all options.

## 🔖 Citation

If you find XMind useful in your research, please cite our paper:

```bibtex
@article{xmind2026,
  title   = {XMind: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```

## 📝 License

XMind is released under the Apache 2.0 license. Parts of `run_XMind.py`,
`XMind/driver_agent/driverAgent.py` and `XMind/scenario/envScenario.py` are adapted from
[DiLu](https://github.com/PJLab-ADG/DiLu), which is also released under the Apache 2.0 license.
