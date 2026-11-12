"""Isolation behavior and job integration without downloading inference weights."""
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from flowhmr.utils import runtime_person_isolation as isolation


def make_masks(path, frames=4, width=64, height=48):
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE masks (frame INTEGER, track INTEGER, polygon TEXT, PRIMARY KEY(frame, track))")
        db.execute("CREATE TABLE metadata (value TEXT)")
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(dict(
            version=isolation.ISOLATION_VERSION, frames=frames, width=width, height=height)),))
        for frame in range(frames):
            # Both people overlap, with their exclusive pixels on opposite sides.
            for tid, polygon in [(11, [[4, 4], [40, 4], [40, 40], [4, 40]]),
                                 (22, [[25, 4], [59, 4], [59, 40], [25, 40]])]:
                if frame == 2 and tid == 11:
                    continue
                db.execute("INSERT INTO masks VALUES (?, ?, ?)", (frame, tid, json.dumps(polygon)))
        db.commit()


class IsolationTest(unittest.TestCase):
    def test_overlap_is_not_restored_and_missing_target_has_no_box_fallback(self):
        with TemporaryDirectory() as root:
            path = Path(root, "masks.sqlite")
            make_masks(path)
            with closing(isolation.PersonMasks(path)) as masks:
                left, right = masks.mask(0, 11), masks.mask(0, 22)
                self.assertEqual(int(left[20, 15]), 255)
                self.assertEqual(int(right[20, 50]), 255)
                self.assertFalse(left[20, 30] or right[20, 30])
                self.assertFalse(masks.mask(2, 11).any())
                self.assertFalse(masks.mask(0, 99).any())
                image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
                rendered = isolation.isolate_frame(image, left)
                np.testing.assert_array_equal(rendered[20, 15], image[20, 15])
                self.assertFalse(np.array_equal(rendered[20, 50], image[20, 50]))
                with self.assertRaises(ValueError):
                    isolation.isolate_frame(image, left[:20])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for real video encoding")
    def test_video_keeps_source_offsets_reports_missing_masks_and_cancels_atomically(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            source, masks, out = root / "input.avi", root / "masks.sqlite", root / "isolated.mp4"
            make_masks(masks)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for n in range(4):
                frame = np.full((48, 64, 3), 20 + n * 45, np.uint8)
                writer.write(frame)
            writer.release()
            fake_video = SimpleNamespace(find_ffmpeg=lambda: shutil.which("ffmpeg"))
            with patch.dict(sys.modules, {"flowhmr.utils.runtime_video_processing": fake_video}):
                path, report = isolation.render_person(source, masks, 11, out, (1, 3))
                self.assertEqual((report["missing_mask_frames"], report["frames"]), (1, 3))
                cap = cv2.VideoCapture(path)
                decoded = []
                try:
                    self.assertEqual(round(cap.get(cv2.CAP_PROP_FPS)), 30)
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        decoded.append(frame)
                finally:
                    cap.release()
                self.assertEqual(len(decoded), 4)  # includes prefix, no timeline shift
                self.assertEqual(decoded[0].shape, (48, 64, 3))
                self.assertLess(abs(float(decoded[3].mean()) - 155), 4)
                _, cached = isolation.render_person(source, masks, 11, out, (1, 3))
                self.assertTrue(cached["reused"])
                before = out.read_bytes()
                def cancel():
                    raise RuntimeError("cancelled")
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    isolation.render_person(source, masks, 22, out, (0, 3), check=cancel)
                self.assertEqual(out.read_bytes(), before)
                self.assertFalse(out.with_suffix(".part.mp4").exists())
                _, changed = isolation.render_person(source, masks, 22, out, (0, 3))
                self.assertFalse(changed["reused"])
                self.assertEqual(changed["missing_mask_frames"], 0)

    def test_analysis_binds_polygons_to_tracker_ids_not_box_order_and_resets_tracker(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)
            def int(self): return self
            def cpu(self): return self
            def tolist(self): return self.value.tolist()
            def numpy(self): return self.value

        @dataclass
        class Track:
            track_id: int
            frames: object
            bboxes: object
            length: int
            avg_area: float

        @dataclass
        class Result:
            tracks: object
            total_frames: int
            width: int
            height: int

        models = []
        class Model:
            def __init__(self, path):
                self.calls = 0
                models.append(self)
            def track(self, frame, **kwargs):
                config = json.loads(Path(kwargs["tracker"]).read_text())
                self_config.append(config)
                ids = [22, 11] if self.calls else [11, 22]
                self.calls += 1
                polygons = {11: np.array([[1, 1], [15, 1], [15, 25], [1, 25]]),
                            22: np.array([[40, 1], [60, 1], [60, 25], [40, 25]])}
                return [SimpleNamespace(boxes=SimpleNamespace(id=Array(ids), xyxy=Array([[0, 0, 30, 40]] * 2)),
                                        masks=SimpleNamespace(xy=[polygons[i] for i in ids]))]

        self_config = []
        with TemporaryDirectory() as root:
            root = Path(root)
            source = root / "input.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            for _ in range(2): writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            with (patch.object(isolation, "check_dependencies"),
                  patch("flowhmr.model_files.MODEL_ROOT", root),
                  patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Model),
                    "flowhmr.utils.runtime_video_processing": SimpleNamespace(TrackInfo=Track, TrackingResult=Result)})):
                for index in range(2):
                    cache = root / f"masks{index}.sqlite"
                    result = isolation.analyze_people(source, cache, "cpu")
                    self.assertEqual([t.track_id for t in result.tracks], [11, 22])
                    self.assertEqual(result.total_frames, 2)
                    with closing(isolation.PersonMasks(cache)) as masks:
                        self.assertEqual(int(masks.mask(1, 11)[10, 10]), 255)
                        self.assertFalse(masks.mask(1, 11)[10, 50])
                self.assertEqual([m.calls for m in models], [2, 0, 2, 0])
                self.assertTrue(all(c["with_reid"] for c in self_config))


if __name__ == "__main__":
    unittest.main()
