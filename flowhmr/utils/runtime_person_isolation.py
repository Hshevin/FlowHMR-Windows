"""Optional instance segmentation and per-person conditioning, inspired by MIRAI.

Masks are recorded together with tracker IDs during analysis, never reassociated
by box overlap during rendering. Original video coordinates and frame numbers are
preserved. This suppresses distractors; it does not reconstruct hidden pixels.
"""

from collections import defaultdict
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

import cv2
import numpy as np


BACKENDS = ("yolox", "yolo26_isolated")
MASK_CACHE = "person_masks.sqlite"
ISOLATION_VERSION = 1


def check_dependencies():
    from importlib.util import find_spec

    if find_spec("ultralytics") is None or find_spec("lap") is None:
        raise RuntimeError(
            "人物分割隔离需要可选依赖。请在当前推理环境执行 "
            "uv pip install --python <推理环境的python> -r requirements-isolation.txt，然后重启。"
        )


def analyze_people(video, cache_path, device, *, check=lambda: None, notify=lambda _: None):
    """Run one fresh tracker per video and publish its masks atomically."""
    check_dependencies()
    from ultralytics import YOLO
    from flowhmr.model_files import MODEL_ROOT
    from .runtime_video_processing import TrackInfo, TrackingResult

    model_dir = Path(MODEL_ROOT) / "person_isolation"
    model_dir.mkdir(parents=True, exist_ok=True)
    check()
    # Ultralytics downloads these official weights on first use if absent.
    model = YOLO(str(model_dir / "yolo26l-seg.pt"))
    reid = YOLO(str(model_dir / "yolo26n-cls.pt"))
    del reid  # tracker owns the encoder; only ensure its managed path exists here
    tracker_config = dict(
        tracker_type="botsort", track_high_thresh=.25, track_low_thresh=.1,
        new_track_thresh=.25, track_buffer=60, match_thresh=.8, fuse_score=True,
        gmc_method="sparseOptFlow", proximity_thresh=.5, appearance_thresh=.8,
        with_reid=True, model=str(model_dir / "yolo26n-cls.pt"),
    )
    destination = Path(cache_path)
    temporary = destination.with_suffix(".part.sqlite")
    temporary.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(video))
    raw = defaultdict(lambda: ([], []))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tempfile.TemporaryDirectory(prefix="flowhmr-tracker-") as config_dir, closing(sqlite3.connect(temporary)) as db:
            config = Path(config_dir) / "botsort.yaml"
            # JSON is a YAML subset; no runtime edits to packaged tracker defaults.
            config.write_text(json.dumps(tracker_config), encoding="utf-8")
            db.execute("CREATE TABLE masks (frame INTEGER, track INTEGER, polygon TEXT, PRIMARY KEY(frame, track))")
            while True:
                check()
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.track(frame, persist=True, tracker=str(config), classes=[0],
                                     conf=.1, iou=.7, imgsz=960, device=str(device), verbose=False)[0]
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    polygons = result.masks.xy if result.masks is not None else []
                    for i, (tid, box) in enumerate(zip(ids, xyxy)):
                        raw[tid][0].append(count)
                        raw[tid][1].append(box)
                        polygon = np.asarray(polygons[i]) if i < len(polygons) else np.empty((0, 2))
                        if polygon.ndim == 2 and polygon.shape[1] == 2 and np.isfinite(polygon).all() and len(polygon) >= 3:
                            db.execute("INSERT INTO masks VALUES (?, ?, ?)",
                                       (count, tid, json.dumps(polygon.round(2).tolist())))
                count += 1
                if count % 30 == 0:
                    db.commit()
                    notify(f"人物分割与跟踪 · {count}/{total or '?'} 帧")
            if not count or (total > 0 and count != total):
                raise ValueError(f"人物分割未完整读取视频：{count}/{total} 帧。")
            if not raw:
                raise ValueError("分割模式未检测到人物，请尝试 YOLOX 或其他素材。")
            db.execute("CREATE TABLE metadata (value TEXT)")
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(dict(
                version=ISOLATION_VERSION, frames=count, width=width, height=height)),))
            db.commit()
        check()
        os.replace(temporary, destination)
    finally:
        cap.release()
        temporary.unlink(missing_ok=True)
        del model
    tracks = []
    for tid, (frames, boxes) in raw.items():
        boxes = np.asarray(boxes, dtype=np.float32)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        tracks.append(TrackInfo(int(tid), np.asarray(frames, dtype=np.int64), boxes, len(frames), float(area.mean())))
    return TrackingResult(tracks, count, width, height)


class PersonMasks:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            self.meta = json.loads(self.db.execute("SELECT value FROM metadata").fetchone()[0])
            if self.meta["version"] != ISOLATION_VERSION:
                raise ValueError("人物分割缓存版本已变化，请重新检测人物。")
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def mask(self, frame, track):
        shape = (self.meta["height"], self.meta["width"])
        target = np.zeros(shape, dtype=np.uint8)
        others = np.zeros_like(target)
        for tid, encoded in self.db.execute("SELECT track, polygon FROM masks WHERE frame=?", (int(frame),)):
            polygon = np.asarray(json.loads(encoded), dtype=np.float32)
            cv2.fillPoly(target if tid == track else others, [np.rint(polygon).astype(np.int32)], 255)
        # Ambiguous pixels must not expose the other actor. No box fallback when
        # the target is missing, and no propagation of stale masks across occlusion.
        target[others != 0] = 0
        return target


def isolate_frame(image, mask):
    """Keep original target pixels, blur everything else; feather only inward."""
    if mask.shape != image.shape[:2]:
        raise ValueError("人物掩码与原视频尺寸不一致，请重新检测。")
    h, w = mask.shape
    small = cv2.resize(image, (max(1, w // 32), max(1, h // 32)), interpolation=cv2.INTER_AREA)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    background = cv2.GaussianBlur(background, (0, 0), 12)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / 2, 0, 1)[..., None]
    return np.rint(image * alpha + background * (1 - alpha)).clip(0, 255).astype(np.uint8)


def render_person(video, cache_path, track, destination, frame_range, *, check=lambda: None, notify=lambda _: None):
    """Make a full-size, 30 FPS input with source-frame indices preserved."""
    from .runtime_video_processing import find_ffmpeg

    destination = Path(destination)
    stamp_path = destination.with_suffix(".json")
    signature = dict(version=ISOLATION_VERSION, track=int(track), range=list(map(int, frame_range)),
                     source=[Path(video).stat().st_size, Path(video).stat().st_mtime_ns],
                     masks=[Path(cache_path).stat().st_size, Path(cache_path).stat().st_mtime_ns])
    try:
        report = json.loads(stamp_path.read_text(encoding="utf-8"))
        if destination.is_file() and report["signature"] == signature:
            report["reused"] = True
            return str(destination), report
    except (OSError, ValueError, KeyError):
        pass
    temporary = destination.with_suffix(".part.mp4")
    cap = cv2.VideoCapture(str(video))
    process = None
    missing = 0
    first, last = map(int, frame_range)
    try:
        with closing(PersonMasks(cache_path)) as masks, tempfile.TemporaryFile() as errors:
            w, h = masks.meta["width"], masks.meta["height"]
            if not 0 <= first <= last < masks.meta["frames"]:
                raise ValueError("人物隔离区间超出素材范围。")
            command = [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv444p", str(temporary)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, stdout=subprocess.DEVNULL)
            # Include the prefix so SAM and HaMeR can keep using original frame offsets.
            for frame_number in range(last + 1):
                check()
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"人物隔离未完整读取视频：第 {frame_number + 1} 帧。")
                mask = masks.mask(frame_number, track) if frame_number >= first else np.zeros((h, w), np.uint8)
                if frame_number >= first and not mask.any():
                    missing += 1
                process.stdin.write(isolate_frame(image, mask).tobytes())
                if frame_number % 30 == 0:
                    notify(f"逐人隔离 · {frame_number + 1}/{last + 1} 帧")
            process.stdin.close()
            if process.wait(timeout=60):
                errors.seek(0)
                raise RuntimeError("人物隔离视频编码失败：" + errors.read().decode(errors="replace")[-2000:])
            if missing == last - first + 1:
                raise ValueError("选中人物在处理区间内没有有效掩码，请重新选择人物或改用 YOLOX。")
        check()
        report = dict(signature=signature, missing_mask_frames=missing, frames=last - first + 1)
        os.replace(temporary, destination)
        stamp_path.write_text(json.dumps(report), encoding="utf-8")
        report["reused"] = False
        return str(destination), report
    finally:
        cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # preserve cancellation/encoding error after terminating FFmpeg
        temporary.unlink(missing_ok=True)
