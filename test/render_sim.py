"""Render crude synthetic street images (true poses) for the simulated project, for UI testing."""
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "..")
from mly_ba import project_norm  # noqa: E402
from sim_street import street, make_project  # noqa: E402


def render(proj, out_dir, true_R, true_C, W=2048, H=1536):
    f, k1, k2 = 0.6, -0.08, 0.02

    def P(X, R, C):
        xc = R @ (np.asarray(X) - C)
        if xc[2] < 0.3:
            return None
        uv = project_norm(xc[None], np.array([0]), np.array([f]), np.array([k1]), np.array([k2]))[0]
        x, y = uv[0] * W + W / 2, uv[1] * W + H / 2
        if abs(x) > 5 * W or abs(y) > 5 * H:
            return None
        return (x, y)

    def poly(d, pts3, R, C, fill, outline=None):
        q = [P(p, R, C) for p in pts3]
        if any(v is None for v in q):
            return
        d.polygon(q, fill=fill, outline=outline)

    def line(d, pts3, R, C, fill, width):
        q = [P(p, R, C) for p in pts3]
        for a, b in zip(q, q[1:]):
            if a and b:
                d.line([a, b], fill=fill, width=width)

    truth = proj["_truth"]["points"]
    for i, im in enumerate(proj["images"]):
        R, C = true_R[i], true_C[i]
        img = Image.new("RGB", (W, H), (170, 196, 214))
        d = ImageDraw.Draw(img)
        # ground: road and sidewalks as strips split into short pieces (crude clipping)
        for a in np.arange(-5, 90, 1.0):
            poly(d, [street(a, -4, 0), street(a + 1, -4, 0), street(a + 1, 4, 0), street(a, 4, 0)], R, C, (92, 96, 100))
            poly(d, [street(a, 4, 0.15), street(a + 1, 4, 0.15), street(a + 1, 6.6, 0.15), street(a, 6.6, 0.15)], R, C, (176, 170, 160))
            poly(d, [street(a, -4, 0.15), street(a + 1, -4, 0.15), street(a + 1, -7.1, 0.15), street(a, -7.1, 0.15)], R, C, (176, 170, 160))
            poly(d, [street(a, 6.6, 0.15), street(a + 1, 6.6, 0.15), street(a + 1, 6.6, 5), street(a, 6.6, 5)], R, C, (196, 120, 90) if int(a) % 6 < 3 else (210, 200, 180))
            poly(d, [street(a, -7.1, 0.15), street(a + 1, -7.1, 0.15), street(a + 1, -7.1, 5), street(a, -7.1, 5)], R, C, (120, 150, 170) if int(a) % 8 < 4 else (220, 215, 200))
        for c in (4, -4):
            line(d, [street(a, c, 0) for a in np.arange(-5, 90, 1.0)], R, C, (235, 235, 235), 3)
        # parked cars: body side panel + wheels
        for car in proj["cars"]:
            Fh, Rh = np.array(truth[car["id"] + ".F"]), np.array(truth[car["id"] + ".R"])
            ax = (Fh - Rh) / np.linalg.norm(Fh - Rh)
            up = np.array([0, 0, 1.0])
            body = [Rh - 0.85 * ax + 0.0 * up, Fh + 0.85 * ax + 0.0 * up, Fh + 0.85 * ax + 1.15 * up, Rh - 0.85 * ax + 1.15 * up]
            poly(d, body, R, C, (40, 70, 140) if car["id"] in ("C1", "C4", "C6") else (160, 40, 40), (20, 20, 20))
            n = np.cross(ax, up)
            for hub in (Fh, Rh):
                for rad, col in ((0.31, (25, 25, 25)), (0.19, (190, 190, 195)), (0.05, (60, 60, 60))):
                    ring = [hub + rad * (math.cos(t) * ax + math.sin(t) * up) for t in np.linspace(0, 2 * math.pi, 40)]
                    poly(d, ring, R, C, col)
        # edge point markers (curb joints / wall corners)
        for pid, X in truth.items():
            if pid[0] == "C":
                continue
            q = P(X, R, C)
            if q:
                d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=(255, 230, 0), width=3)
        img.save(f"{out_dir}/{im['id']}.png")


if __name__ == "__main__":
    import os
    from scipy.spatial.transform import Rotation
    os.makedirs("imgs", exist_ok=True)
    proj = make_project(seed=7, cam="perspective", wheel_dist=(3.5, 12), n_per_point=6, spacing=2.0)
    tR = [Rotation.from_rotvec(c["R"]).as_matrix() for c in proj["_truth"]["cams"]]
    tC = [np.array(c["C"]) for c in proj["_truth"]["cams"]]
    render(proj, "imgs", tR, tC)
    json.dump(proj, open("sim_e2e.json", "w"))
    print(len(proj["images"]), "images,", len(proj["observations"]), "obs")
