"""
XMind: closed-loop highway-env simulation with a frozen LLM, working memory and episodic memory.

Each decision step
    1. encode the scene (11 direction slots with risks) and add it to working memory
    2. DriverAgent.decision: round 1 from working memory; with retrieval on, a second round only
       when the retrieved capsule's preferred action differs from round 1
    3. with evolve on, record the frame and an environment snapshot (before env.step)
    4. env.step
After each episode (evolve on): find the decision-critical state, run counterfactual rollouts
and write a capsule. Results go to <result_folder>_<timestamp>/ (log.txt, output.txt, videos/).
"""

import atexit
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo
from rich import print
from rich.markup import escape

from XMind.driver_agent.driverAgent import DriverAgent
from XMind.driver_agent.workingMemory import WorkingMemory
from XMind.episodic_memory import EpisodicMemory
from XMind.scenario.envScenario import ENV_TYPE, EnvScenario


DRIVING_INTENSIONS = "Drive safely and avoid collisions"
RECORD_VIDEO = True          # one video per episode: <result folder>/videos/*.mp4 (requires moviepy)
ERROR_COOLDOWN_S = 1         # pause after an episode that raised an error


# ============================================================
# Output: terminal as is, plus a copy without colour codes in output.txt
# ============================================================

ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1B\[[0-?]*[ -/]*[@-~]"
    r"|\x1B\][^\x07]*(?:\x07|\x1B\\)"
    r"|\x1B[@-_]"
    r"|\x9B[0-?]*[ -/]*[@-~]"
    r"|\[(?:\d{1,3}(?:;\d{1,3})*)m)"
)


class OutputTee:
    """Write terminal output unchanged and a clean copy to a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        if not data:
            return 0
        self.stream.write(data)
        self.stream.flush()
        clean = ANSI_ESCAPE_RE.sub("", str(data)).replace("\r\n", "\n").replace("\r", "\n")
        self.log_file.write(clean)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ============================================================
# Configuration
# ============================================================

def setup_env(config):
    """Export the LLM settings for langchain and return the highway-env configuration."""
    api_type = config["OPENAI_API_TYPE"]
    if api_type == "azure":
        os.environ["OPENAI_API_TYPE"] = api_type
        os.environ["OPENAI_API_VERSION"] = config["AZURE_API_VERSION"]
        os.environ["OPENAI_API_BASE"] = config["AZURE_API_BASE"]
        os.environ["OPENAI_API_KEY"] = config["AZURE_API_KEY"]
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config["AZURE_CHAT_DEPLOY_NAME"]
    elif api_type == "openai":
        os.environ["OPENAI_API_TYPE"] = api_type
        if config.get("OPENAI_KEY"):
            os.environ["OPENAI_API_KEY"] = config["OPENAI_KEY"]
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_KEY in config.yaml or the OPENAI_API_KEY environment variable")
        os.environ["OPENAI_CHAT_MODEL"] = config["OPENAI_CHAT_MODEL"]
        if config.get("OPENAI_API_BASE"):   # optional OpenAI-compatible endpoint
            os.environ["OPENAI_API_BASE"] = config["OPENAI_API_BASE"]
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

    lanes_count = config["lanes_count"]
    if type(lanes_count) is not int or lanes_count <= 0:
        raise ValueError("lanes_count must be a positive integer")

    return {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "vehicles_count": config["vehicle_count"],
            "see_behind": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(5, 32, 9),
        },
        "lanes_count": lanes_count,
        "other_vehicles_type": config["other_vehicle_type"],
        "duration": config["simulation_duration"],
        "vehicles_density": config["vehicles_density"],
        "show_trajectories": True,
        "render_agent": True,
        "scaling": 5,
        "initial_lane_id": None,
        "ego_spacing": 4,
    }


def episode_seed_list(config):
    """Seeds in rounds: every seed once, then again ... (len(scene_seeds) x episodes_num episodes)."""
    scene_seeds = config.get("scene_seeds")
    if not isinstance(scene_seeds, list) or not scene_seeds or any(type(s) is not int or s < 0 for s in scene_seeds):
        raise ValueError("scene_seeds must be a non-empty list of non-negative integers")
    repeats = config["episodes_num"]
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("episodes_num must be a positive integer")
    return scene_seeds, scene_seeds * repeats


def open_result_folder(config):
    folder = "{}_{}".format(config["result_folder"].rstrip("/\\"), datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)

    run_log_path = os.path.join(folder, "output.txt")
    run_log_file = open(run_log_path, "w", encoding="utf-8-sig")   # BOM: readable by Windows Notepad too
    stdout_tee, stderr_tee = OutputTee(sys.stdout, run_log_file), OutputTee(sys.stderr, run_log_file)
    sys.stdout, sys.stderr = stdout_tee, stderr_tee

    def finalize_run_log():
        stdout_tee.flush()
        stderr_tee.flush()
        # restore the terminal first: output at interpreter exit must not hit a closed file
        sys.stdout, sys.stderr = stdout_tee.stream, stderr_tee.stream
        run_log_file.close()

    atexit.register(finalize_run_log)
    print(f"Result folder: {os.path.abspath(folder)}")
    print(f"Output log: {os.path.abspath(run_log_path)}")
    return folder


def append_log(result_folder, *lines):
    with open(os.path.join(result_folder, "log.txt"), "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + " \n")


# ============================================================
# Video
# ============================================================

def start_video(env, video_folder, name_prefix):
    """Record <video_folder>/<name_prefix>-episode-0.mp4 from the frame after reset. Intermediate
    simulation frames are included (video fps = 2 x simulation frequency); the file is written when
    the episode ends or the env is closed."""
    video_env = RecordVideo(
        env,
        video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda _: True,
        disable_logger=True,
    )
    base = video_env.unwrapped
    base.set_record_video_wrapper(video_env)
    try:
        video_env.start_video_recorder()
    except Exception:
        base.set_record_video_wrapper(None)
        raise
    return video_env


# ============================================================
# One episode
# ============================================================

def run_episode(episode, seed, config, env_config, memory, video_folder=None):
    """Run one episode and return its record; an error only skips this episode."""
    env = gym.make(ENV_TYPE, render_mode="rgb_array")
    env.unwrapped.configure(env_config)
    env.reset(seed=seed)
    result_prefix = f"highway_{episode}"
    sce = EnvScenario(env)

    video_path = None
    if video_folder is not None:
        try:
            env = start_video(env, video_folder, f"{result_prefix}_seed{seed}")
            video_path = env.video_recorder.path
        except Exception as error:
            print(escape(f"WARNING | Episode {episode} | Seed {seed}: video disabled: {error!r}"))

    memory_size = config.get("working_memory_size", 3)
    working_memory = WorkingMemory(max_frames=memory_size)
    agent = DriverAgent(retriever=memory.retriever if memory is not None else None)

    builder = memory.builder if memory is not None else None
    recorder = builder.start_episode(episode, seed, memory_size) if builder is not None else None

    decision_paths = Counter()
    steps = 0
    status = "error"
    a1_summary = None

    try:
        for i in range(config["simulation_duration"]):
            print(f"Episode {episode} | Step {i}")
            frame_json = sce.describe(i)   # before env.step: boundary and JSON describe the same frame
            working_memory.add_frame(frame_json)
            lane_boundary = sce.lane_boundary()
            print("[cyan]Current driving context (working memory): [/cyan]\n", working_memory.to_json())

            decision = agent.decision(
                working_memory,
                sce.available_actions_description(),
                DRIVING_INTENSIONS,
                current_frame=frame_json,
                lane_boundary=lane_boundary,
            )
            decision_paths[decision.path] += 1

            retrieval = decision.retrieval
            if retrieval is not None:
                print(escape(
                    f"Episodic retrieval: enabled={retrieval['enabled']}, reason={retrieval['reason']}, "
                    f"risk_source={retrieval['risk_source']}, triggers={retrieval['trigger_directions']}, matches="
                    + json.dumps([
                        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                        for row in retrieval["matches"]
                    ])
                ))
            print(escape(f"Decision: {decision.summary()}"))

            if recorder is not None:
                recorder.record(frame_json, lane_boundary, decision.action, env)

            _, _, terminated, truncated, info = env.step(decision.action)
            steps += 1
            print("--------------------")
            if terminated or truncated:
                print(f"Simulation finished after step {i}: terminated={terminated}, truncated={truncated}")
                break

        crashed = bool(info.get("crashed", sce.ego.crashed))
        status = "collision" if crashed else "success"

        if builder is not None:
            started = time.time()
            a1_summary = builder.end_episode(recorder, crashed, llm=agent.llm)
            a1_summary["seconds"] = round(time.time() - started, 1)
            print(escape(f"[A1] {json.dumps(a1_summary, ensure_ascii=False)}"))

    except Exception as error:
        # the driving result is kept if the error happened while building the capsule
        print(escape(f"ERROR | Episode {episode} | Seed {seed}: {error!r}"))
        traceback.print_exc()
        time.sleep(ERROR_COOLDOWN_S)

    finally:
        try:
            env.close()   # also writes the video of an episode that ended with an error
        except Exception as error:
            print(escape(f"ERROR | Episode {episode} | Seed {seed}: env.close failed: {error!r}"))
            traceback.print_exc()

    if video_path is not None and not os.path.isfile(video_path):
        print(f"[red]WARNING: video for episode {episode} was not written: {video_path}[/red]")
        video_path = None

    return {
        "status": status,
        "steps": steps,
        "decision_paths": dict(decision_paths),
        "a1": a1_summary,
        "video": video_path,
    }


# ============================================================
# Main
# ============================================================

def main(config_path="config.yaml"):
    import warnings
    warnings.filterwarnings("ignore")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = setup_env(config)
    scene_seeds, episode_seeds = episode_seed_list(config)
    if RECORD_VIDEO:
        try:
            import moviepy  # noqa: F401  (used by gymnasium's RecordVideo)
        except ImportError:
            raise SystemExit("RECORD_VIDEO needs moviepy (pip install moviepy), or set RECORD_VIDEO = False in run_XMind.py")
    result_folder = open_result_folder(config)
    project_root = Path(__file__).resolve().parent

    append_log(
        result_folder,
        "result_folder {} | lanes_count: {}".format(result_folder, env_config["lanes_count"]),
        "episode_seeds: {}".format(episode_seeds),
    )
    print(f"Lanes: {env_config['lanes_count']} | Episode seeds: {episode_seeds}")

    video_folder = os.path.join(result_folder, "videos") if RECORD_VIDEO else None
    video_status = "enabled | " + os.path.abspath(video_folder) if RECORD_VIDEO else "disabled"
    print(f"Video: {video_status}")
    append_log(result_folder, "video: {}".format(video_status))

    memory = EpisodicMemory.from_config(config, project_root)
    memory_status = "enabled | " + memory.describe() if memory is not None else "disabled"
    print(f"Episodic memory: {memory_status}")
    append_log(result_folder, "episodic_memory: {}".format(memory_status))

    for episode, seed in enumerate(episode_seeds):
        repeat = episode // len(scene_seeds) + 1
        record = run_episode(episode, seed, config, env_config, memory, video_folder)

        result_line = "Simulation {} | Seed {} | Repeat {} | Steps: {} | Status: {} | File prefix: highway_{}".format(
            episode, seed, repeat, record["steps"], record["status"], episode)
        memory_line = "Memory {} | decision paths: {} | A1: {}".format(
            episode, json.dumps(record["decision_paths"]), json.dumps(record["a1"], ensure_ascii=False))
        log_lines = [result_line, memory_line]
        if RECORD_VIDEO:
            log_lines.append("Video {} | Seed {} | {}".format(
                episode, seed, os.path.relpath(record["video"], result_folder) if record["video"] else "n/a"))
        append_log(result_folder, *log_lines)
        print(result_line)
        print(escape(memory_line))
        if RECORD_VIDEO:
            print(escape(log_lines[-1]))
        print("==========Simulation {} Done==========".format(episode))


if __name__ == "__main__":
    main()
