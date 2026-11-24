"""
Save frames of the videos written by run_XMind.py (<result folder>/videos/*.mp4) as PNG.

Frame <-> decision step (highway-env defaults: simulation 15 Hz, policy 1 Hz):
    frame 0 is the scene after reset, i.e. what the LLM sees at decision step 0;
    each decision step spans 15 frames, so decision step i is frame 15*i (step05 = frame 75);
    frames in between are simulation frames while the action executes (step05+07 = 7 frames after step 5).
    Videos are 30 fps (2x speed): player time T s = frame round(30*T) = simulation time 2T s.
    The frames per step default to fps / 2; use --frames-per-step if the policy frequency is not 1 Hz.

Usage (from the project root):
    # contact sheet with one thumbnail per decision step (plus the last frame), to pick frames
    python save_frames.py results_xxx/videos/highway_0_seed6932-episode-0.mp4 --sheet
    # save by decision step / frame index / player time (s); options can be combined
    python save_frames.py VIDEO --steps 0 5 12
    python save_frames.py VIDEO --frames 37 120
    python save_frames.py VIDEO --times 2.5 6
    # every N-th frame; --last also saves the last frame (e.g. the collision)
    python save_frames.py VIDEO --every 15 --last
    # browse interactively: left/right frame, up/down decision step, Home/End, s or Enter to save, q to quit
    python save_frames.py VIDEO --interactive
    # several videos at once get the same selection; output goes to frames/ next to the video unless --out

File names: <video name>_f<frame>_step<step>[+<offset>].png
    e.g. highway_0_seed6932_f0075_step05.png, highway_0_seed6932_f0082_step05+07.png
"""

import argparse
import os
import re
import sys

import numpy as np


# ============================================================
# Reading, frame numbers, saving
# ============================================================

def load_video(path):
    """All frames (about 450 at most per episode) and the fps."""
    import imageio.v2 as imageio   # installed with moviepy

    reader = imageio.get_reader(path)
    try:
        fps = float(reader.get_meta_data()["fps"])
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path}: no frames")
    return frames, fps


def video_stem(path):
    """highway_0_seed6932-episode-0.mp4 -> highway_0_seed6932"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"-episode-\d+$", "", stem)


def step_label(index, frames_per_step):
    step, offset = divmod(index, frames_per_step)
    return f"step{step:02d}" + (f"+{offset:02d}" if offset else "")


def frame_path(out_dir, stem, index, frames_per_step):
    return os.path.join(out_dir, f"{stem}_f{index:04d}_{step_label(index, frames_per_step)}.png")


def save_frame(frames, index, out_dir, stem, frames_per_step):
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    path = frame_path(out_dir, stem, index, frames_per_step)
    Image.fromarray(frames[index]).save(path)
    return path


def select_indices(n_frames, fps, frames_per_step, args):
    """Frame indices selected by --frames / --steps / --times / --every / --last (sorted, unique)."""
    wanted = []
    wanted += [(f, f"frame {f}") for f in args.frames or []]
    wanted += [(s * frames_per_step, f"step {s}") for s in args.steps or []]
    wanted += [(int(round(t * fps)), f"time {t:g}s") for t in args.times or []]
    if args.every:
        wanted += [(f, None) for f in range(0, n_frames, args.every)]
    if args.last:
        wanted.append((n_frames - 1, None))

    indices = set()
    for index, label in wanted:
        if 0 <= index < n_frames:
            indices.add(index)
        else:
            print(f"  skip {label}: frame {index} is outside 0..{n_frames - 1} "
                  f"(the episode ended at step {(n_frames - 1) // frames_per_step})")
    return sorted(indices)


# ============================================================
# Contact sheet
# ============================================================

def contact_sheet(frames, frames_per_step, path, columns=4, width=300):
    """One thumbnail per decision step (frames 0, 15, 30, ...) plus the last frame, labelled with step and frame."""
    from PIL import Image, ImageDraw

    indices = list(range(0, len(frames), frames_per_step))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    thumbs = []
    for index in indices:
        image = Image.fromarray(frames[index])
        height = int(round(image.height * width / image.width))
        thumbs.append((index, image.resize((width, height), Image.LANCZOS)))

    label_h, gap = 16, 4
    cell_w = width + gap
    cell_h = thumbs[0][1].height + label_h + gap
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w + gap, rows * cell_h + gap), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for k, (index, thumb) in enumerate(thumbs):
        x = gap + (k % columns) * cell_w
        y = gap + (k // columns) * cell_h
        text = f"{step_label(index, frames_per_step)}  |  frame {index}"
        if index == len(frames) - 1:
            text += "  (last)"
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0))
        sheet.paste(thumb, (x, y + label_h))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    return path


# ============================================================
# Interactive picker (matplotlib window)
# ============================================================

class FramePicker:
    """left/right: frame, up/down: decision step, Home/End, s or Enter: save, q / Esc: quit."""

    def __init__(self, frames, fps, frames_per_step, save):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # disable matplotlib's own shortcuts (e.g. s = save figure, arrows = view history)
        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[key] = []

        self.plt = plt
        self.frames = frames
        self.fps = fps
        self.frames_per_step = frames_per_step
        self.save = save
        self.index = 0
        self.saved = []
        self.message = ""

        height, width = frames[0].shape[:2]
        self.fig = plt.figure(figsize=(max(8.0, width / 70), max(5.5, height / 70 + 1.2)))
        self.ax = self.fig.add_axes([0.02, 0.14, 0.96, 0.78])
        self.ax.axis("off")
        self.image = self.ax.imshow(frames[0])
        slider_ax = self.fig.add_axes([0.12, 0.05, 0.76, 0.03])
        self.slider = Slider(slider_ax, "frame", 0, len(frames) - 1, valinit=0, valstep=1, valfmt="%d")
        self.slider.on_changed(lambda value: self.show(int(round(value))))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show(0)

    def show(self, index):
        self.index = index
        self.image.set_data(self.frames[index])
        mark = "  [saved]" if index in self.saved else ""
        self.ax.set_title(
            f"frame {index}/{len(self.frames) - 1} | {step_label(index, self.frames_per_step)} | "
            f"video {index / self.fps:.2f}s{mark}\n"
            f"<-/-> frame   up/down step   Home/End   s/Enter save   q quit   {self.message}",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def go(self, index):
        index = min(max(index, 0), len(self.frames) - 1)
        if index != int(round(self.slider.val)):
            self.slider.set_val(index)     # triggers show()
        else:
            self.show(index)

    def on_key(self, event):
        key = event.key
        step = self.frames_per_step
        if key == "right":
            self.go(self.index + 1)
        elif key == "left":
            self.go(self.index - 1)
        elif key == "up":
            self.go((self.index // step + 1) * step)          # next decision step
        elif key == "down":
            self.go((self.index - 1) // step * step if self.index else 0)   # previous decision step
        elif key == "home":
            self.go(0)
        elif key == "end":
            self.go(len(self.frames) - 1)
        elif key in ("s", "enter"):
            path = self.save(self.index)
            if self.index not in self.saved:
                self.saved.append(self.index)
            self.message = "saved " + os.path.basename(path)
            print("  saved", path)
            self.show(self.index)
        elif key in ("q", "escape"):
            self.plt.close(self.fig)

    def run(self):
        self.plt.show()
        return self.saved


# ============================================================
# Main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pick frames from XMind videos and save them as PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("videos", nargs="+", help="mp4 file(s) written by run_XMind.py")
    parser.add_argument("--steps", type=int, nargs="+", help="decision steps (frame = step x frames-per-step)")
    parser.add_argument("--frames", type=int, nargs="+", help="video frame indices (0-based)")
    parser.add_argument("--times", type=float, nargs="+", help="times in seconds as shown by a video player")
    parser.add_argument("--every", type=int, help="save every N-th frame")
    parser.add_argument("--last", action="store_true", help="also save the last frame (e.g. the collision)")
    parser.add_argument("--sheet", action="store_true", help="save an overview with one thumbnail per decision step")
    parser.add_argument("--interactive", action="store_true", help="browse frames in a window and press s to save")
    parser.add_argument("--out", help="output folder (default: frames/ next to the video)")
    parser.add_argument("--frames-per-step", type=int,
                        help="video frames per decision step (default: fps / 2, i.e. 15)")
    args = parser.parse_args(argv)

    if not (args.steps or args.frames or args.times or args.every or args.last
            or args.sheet or args.interactive):
        parser.error("choose what to save: --steps / --frames / --times / --every / --last, "
                     "or use --sheet / --interactive")
    if args.every is not None and args.every <= 0:
        parser.error("--every must be positive")
    if args.frames_per_step is not None and args.frames_per_step <= 0:
        parser.error("--frames-per-step must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)

    for video in args.videos:
        frames, fps = load_video(video)
        frames_per_step = args.frames_per_step or max(1, int(round(fps / 2)))
        stem = video_stem(video)
        out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(video)), "frames")
        print(f"{video}: {len(frames)} frames, {fps:g} fps, {frames_per_step} frames per decision step, "
              f"last frame = {step_label(len(frames) - 1, frames_per_step)}")

        if args.sheet:
            print("  sheet", contact_sheet(frames, frames_per_step, os.path.join(out_dir, f"{stem}_sheet.png")))

        for index in select_indices(len(frames), fps, frames_per_step, args):
            print("  saved", save_frame(frames, index, out_dir, stem, frames_per_step))

        if args.interactive:
            picker = FramePicker(
                frames, fps, frames_per_step,
                lambda index: save_frame(frames, index, out_dir, stem, frames_per_step),
            )
            picker.run()


if __name__ == "__main__":
    sys.exit(main())
