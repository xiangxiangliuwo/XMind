import json
from collections import deque

from XMind.scenario.directions import FRAME_KEY


class WorkingMemory:
    """The latest N decision frames (JSON), oldest first."""

    def __init__(self, max_frames: int = 3) -> None:
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self._frames = deque(maxlen=max_frames)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def add_frame(self, frame_json: str) -> None:
        frame = json.loads(frame_json)
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be a JSON object")
        if any(key not in frame for key in ("Frame", "Context", "Ego", FRAME_KEY)):
            raise ValueError("Missing required driving-frame fields")
        frame_id = frame["Frame"]
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("Frame must be a non-negative integer")
        json.dumps(frame, ensure_ascii=False, allow_nan=False)   # reject NaN / Infinity before storing

        if self._frames:
            latest_id = self._frames[-1]["Frame"]
            if frame_id < latest_id:
                raise ValueError("Frame moved backwards; clear memory for a new episode")
            if frame_id == latest_id:   # a retried frame replaces the latest one
                self._frames[-1] = frame
                return
        self._frames.append(frame)

    def to_json(self) -> str:
        context = {
            "current_frame": self._frames[-1]["Frame"] if self._frames else None,
            "frames": list(self._frames),
        }
        return json.dumps(context, ensure_ascii=False, allow_nan=False, indent=2)
