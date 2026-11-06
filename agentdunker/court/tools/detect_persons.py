"""Stand-in for M1: per-frame person boxes (no identities) written in tracks.json shape.

python -m agentdunker.court.tools.detect_persons <video> -o persons.json [--model yolo11m.pt] [--stride 1]
Used to mask players during line fitting and to put box bottom-centres on the minimap until M1 tracks exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(8)
    from ultralytics import YOLO
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    frames = []
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f % a.stride == 0:
            r = model.predict(fr, imgsz=a.imgsz, classes=[0], conf=0.35, verbose=False)[0]
            for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                frames.append({"f": f, "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                               "foot": [round((x0 + x1) / 2, 1), round(y1, 1)], "conf": round(c, 3), "observed": True})
        f += 1
    Path(a.out).write_text(json.dumps({"video": {"path": a.video, "n_frames": f},
                                       "tracks": [{"track_id": 0, "role": "other", "team": None,
                                                   "team_color": [255, 60, 60], "frames": frames}]}))
    print(a.out, len(frames), "boxes over", f, "frames")


if __name__ == "__main__":
    main()
