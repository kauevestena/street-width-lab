#!/usr/bin/env python3
"""Simulated street + Mapillary-like poses, written in the tool's export format.

Street axis bearing 62°, one-way street, 8.00 m curb to curb, parking on both sides,
sidewalks 2.60 m (left) and 3.10 m (right). The 'Mapillary' trajectory is the true one
scaled by `gps_scale` about its centroid (GPS-derived scale error) plus noise.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from mly_ba import enu_to_geodetic, project_norm

REF = (-49.2711, -25.4410, 935.0)  # somewhere in Curitiba, only used as an ENU origin
BEARING = math.radians(62.0)
ALONG = np.array([math.sin(BEARING), math.cos(BEARING), 0.0])
LEFT = np.array([-math.cos(BEARING), math.sin(BEARING), 0.0])
UP = np.array([0.0, 0.0, 1.0])
TRUTH = {"road (curb to curb)": 8.00, "sidewalk left": 2.60, "sidewalk right": 3.10}


def street(a, c, z):
    return a * ALONG + c * LEFT + z * UP


def camera_R(heading, pitch=0.0, roll=0.0):
    f = np.array([math.sin(heading), math.cos(heading), 0.0])
    r = np.array([math.cos(heading), -math.sin(heading), 0.0])
    d = np.array([0.0, 0.0, -1.0])
    R = np.stack([r, d, f])
    return Rotation.from_euler("zxy", [roll, pitch, 0.0]).as_matrix() @ R


def make_project(seed=0, cam="perspective", gps_scale=1.03, click_sigma_px=1.5, n_per_point=4,
                 lookup_error=None, rot_noise_deg=0.3, pos_noise_m=0.05, spacing=2.5,
                 wheel_dist=(3.5, 22.0), edge_dist=(3.5, 22.0), display_long=2048):
    rng = np.random.default_rng(seed)
    # ---- object points
    pts = {}
    cars = []
    wb_true = [2.465, 2.590, 2.523, 2.700, 2.551, 2.636, 2.420]
    starts = [2.0, 7.2, 12.6, 18.1, 4.5, 10.3, 16.0]
    sides = ["R", "R", "R", "R", "L", "L", "L"]
    for k, (L, a0, sd) in enumerate(zip(wb_true, starts, sides)):
        cid = f"C{k + 1}"
        c = -2.2 if sd == "R" else 2.2
        z = 0.31
        # right-side cars point along +a (front further along), left-side cars point the same way
        pts[f"{cid}.R"] = street(a0 + 15, c, z)
        pts[f"{cid}.F"] = street(a0 + 15 + L, c, z)
        Llook = L if lookup_error is None else L + lookup_error[k]
        cars.append(dict(id=cid, model=f"sim model {k + 1}", wheelbase_m=round(Llook, 3), sigma_m=0.01,
                         side="right" if sd == "R" else "left"))
    edge_pts = []
    for g, c, z, a_list in [("curb_L", 4.0, 0.0, np.arange(14, 40, 5.0)),
                            ("curb_R", -4.0, 0.0, np.arange(16, 42, 5.0)),
                            ("facade_L", 6.6, 0.15, np.arange(15, 41, 6.0)),
                            ("facade_R", -7.1, 0.15, np.arange(17, 43, 6.0))]:
        for n, a in enumerate(a_list):
            pid = f"{g}.{n + 1}"
            pts[pid] = street(a + rng.uniform(-0.8, 0.8), c, z)
            edge_pts.append(dict(id=pid, kind="edge", group=g))
    # ---- true cameras
    a_cam = np.arange(0.0, 50.0, spacing)
    H = 1.6 if cam == "perspective" else 2.1
    true_R, true_C = [], []
    for a in a_cam:
        heading = BEARING + math.radians(rng.normal(0, 1.0))
        true_R.append(camera_R(heading, math.radians(rng.normal(-3, 0.5)), math.radians(rng.normal(0, 0.5))))
        true_C.append(street(a, rng.normal(0, 0.15), H))
    true_R, true_C = np.array(true_R), np.array(true_C)
    # ---- 'Mapillary' poses: similarity-distorted + noise
    cbar = true_C.mean(0)
    mC = cbar + gps_scale * (true_C - cbar) + np.array([1.3, -0.7, 2.0]) + rng.normal(0, pos_noise_m, true_C.shape)
    mR = np.array([Rotation.from_rotvec(rng.normal(0, math.radians(rot_noise_deg), 3)).as_matrix() @ R for R in true_R])
    W, Hh = (2048, 1536) if cam == "perspective" else (2048, 1024)
    params = [0.6, -0.08, 0.02] if cam == "perspective" else []
    code = 0 if cam == "perspective" else 2
    images = []
    for i in range(len(a_cam)):
        lon, lat, h = enu_to_geodetic(mC[i], REF)
        ax = mR[i].T @ np.array([0, 0, 1.0])
        images.append(dict(id=f"img{i:03d}", sequence="simseq", captured_at=1700000000000 + 1000 * i,
                           camera_type=cam, camera_parameters=params, width=W, height=Hh,
                           computed_geometry={"type": "Point", "coordinates": [float(lon[0]), float(lat[0])]},
                           computed_altitude=float(h[0]), computed_rotation=Rotation.from_matrix(mR[i]).as_rotvec().tolist(),
                           computed_compass_angle=math.degrees(math.atan2(ax[0], ax[1])) % 360, exif_orientation=1))
    # ---- clicks: project TRUE points with TRUE poses
    obs = []
    f, k1, k2 = (params + [0, 0, 0])[:3] if params else (1.0, 0.0, 0.0)
    for pid, X in pts.items():
        cand = []
        for i in range(len(a_cam)):
            xc = true_R[i] @ (X - true_C[i])
            dist = np.linalg.norm(X - true_C[i])
            lo, hi = wheel_dist if '.' in pid and pid[0] == 'C' else edge_dist
            if dist < lo or dist > hi:
                continue
            if code == 0 and xc[2] < 0.5:
                continue
            uv = project_norm(xc[None], np.array([code]), np.array([f]), np.array([k1]), np.array([k2]))[0]
            xf, yf = uv[0] * max(W, Hh) / W + 0.5, uv[1] * max(W, Hh) / Hh + 0.5
            if 0.01 < xf < 0.99 and 0.01 < yf < 0.99:
                cand.append((i, xf, yf))
        if len(cand) < 2:
            continue
        sel = np.linspace(0, len(cand) - 1, min(n_per_point, len(cand))).round().astype(int)
        for q in sorted(set(sel)):
            i, xf, yf = cand[q]
            nx, ny = rng.normal(0, click_sigma_px, 2) / display_long
            obs.append(dict(image=f"img{i:03d}", point=pid, x=xf + nx * max(W, Hh) / W, y=yf + ny * max(W, Hh) / Hh, display_long=display_long))
    points = []
    for c in cars:
        points += [dict(id=f"{c['id']}.F", kind="wheel", car=c["id"], pos="front"),
                   dict(id=f"{c['id']}.R", kind="wheel", car=c["id"], pos="rear")]
    points += edge_pts
    s0 = street(10, 0, 0); s1 = street(45, 0, 0)
    g0 = enu_to_geodetic(s0, REF); g1 = enu_to_geodetic(s1, REF)
    return dict(format="mly-scalebar-project/1",
                reference=dict(lon=REF[0], lat=REF[1], alt=REF[2]),
                segment=dict(start=[float(g0[0][0]), float(g0[1][0])], end=[float(g1[0][0]), float(g1[1][0])], buffer_m=15),
                settings=dict(sigma_click_px=click_sigma_px, display_long_side=2048),
                images=images, cars=cars, points=points, observations=obs,
                width_pairs=[{"name": "road (curb to curb)", "a": "curb_L", "b": "curb_R"},
                             {"name": "sidewalk left", "a": "curb_L", "b": "facade_L"},
                             {"name": "sidewalk right", "a": "curb_R", "b": "facade_R"}],
                ground_truth=[dict(pair=k, width_m=v) for k, v in TRUTH.items()],
                _truth=dict(points={k: v.tolist() for k, v in pts.items()}, scale=1 / gps_scale,
                            cams=[dict(R=Rotation.from_matrix(true_R[i]).as_rotvec().tolist(), C=true_C[i].tolist()) for i in range(len(a_cam))]))


if __name__ == "__main__":
    import sys
    cam = sys.argv[1] if len(sys.argv) > 1 else "perspective"
    json.dump(make_project(cam=cam), open(f"sim_{cam}.json", "w"), indent=1)
    print("written", f"sim_{cam}.json")
