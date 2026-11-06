"""Smoke tests: camera round trip, synthetic-frame line refinement, contract mapping."""
import cv2
import numpy as np

from agentdunker.court import camera as cam
from agentdunker.court.api import CourtCalibration
from agentdunker.court.draw import draw_court
from agentdunker.court.frame_solver import refine
from agentdunker.court.geometry import EdgeMap, project

SIZE = (1920, 1080)


def _synthetic():
    # broadcast-like camera: centre court, 80 ft back, 30 ft up, looking at the left half
    p = cam.look_at([0.0, 80.0, -30.0], [-25.0, 0.0, 0.0], 2400.0)
    img = np.full((SIZE[1], SIZE[0], 3), (90, 140, 190), np.uint8)
    draw_court(img, cam.to_H(p, SIZE), (255, 255, 255), 3)
    return p, img


def test_camera_round_trip():
    p, _ = _synthetic()
    assert cam.plausible(p, SIZE)
    H = cam.to_H(p, SIZE)
    q = cam.from_H(H, SIZE, np.array([[-20.0, 0.0]]))
    assert np.abs(cam.to_H(q, SIZE) - H).max() < 1e-6


def test_line_refinement_recovers_perturbed_camera():
    p, img = _synthetic()
    em = EdgeMap.from_frame(img)
    p0 = p.copy()
    p0[:3] += [0.004, -0.004, 0.0]
    pr, _ = refine(p0, em, SIZE)
    g = np.array([[-47, -25], [-28, 8], [0, 0], [-47, 8]], float)
    err = np.linalg.norm(project(cam.to_H(pr, SIZE), g) - project(cam.to_H(p, SIZE), g), axis=1)
    assert np.median(err) < 6.0   # starts ~13 px off; edge sampling is integer at half resolution


def test_img_to_court_inverts_contract_H():
    p, _ = _synthetic()
    Hc = cam.to_H(p, SIZE)
    Hi = np.linalg.inv(Hc)
    cal = CourtCalibration({"frames": [{"f": 0, "H_img2court": (Hi / Hi[2, 2]).tolist(), "ok": True}]})
    court = np.array([[-28.0, 8.0], [-47.0, -25.0]])
    img = project(Hc, court)
    assert np.allclose(cal.img_to_court(0, img), court, atol=1e-6)
    assert np.allclose(cal.court_to_img(0, court), img, atol=1e-6)
    assert np.isnan(cal.img_to_court(5, img)).all()
