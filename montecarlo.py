#!/usr/bin/env python3
"""Monte Carlo over simulated streets: is the predicted sigma honest, and which capture/click strategy works?"""
import json
import sys
from multiprocessing import Pool

import numpy as np

from mly_ba import load_project, run
from sim_street import make_project, TRUTH

SCEN = {
    "S1 forward cam, hubs clicked 3.5-22 m, 4 frames": dict(cam="perspective"),
    "S2 forward cam, hubs clicked 3.5-10 m, 4 frames": dict(cam="perspective", wheel_dist=(3.5, 10)),
    "S3 forward cam, hubs clicked 3.5-12 m, every frame": dict(cam="perspective", wheel_dist=(3.5, 12), n_per_point=12, spacing=1.5),
    "S4 = S3 with Mapillary rotation noise 0.1°": dict(cam="perspective", wheel_dist=(3.5, 12), n_per_point=12, spacing=1.5, rot_noise_deg=0.1),
    "S5 360° pano, 2048 px thumbnails, every frame": dict(cam="spherical", wheel_dist=(2.5, 12), n_per_point=12, spacing=1.5),
    "S6 360° pano, full res 5760 px, every frame": dict(cam="spherical", wheel_dist=(2.5, 12), n_per_point=12, spacing=1.5, display_long=5760),
}


def one(args):
    name, seed = args
    d = make_project(seed=seed, **SCEN[name])
    p = load_project(d)
    true_s = d["_truth"]["scale"]
    out = {}
    for mode in ("mapillary", "wheelbase"):
        r = run(p, mode)
        w = next(x for x in r["widths"] if x["name"].startswith("road"))
        out[mode] = dict(err=w["width"] - TRUTH["road (curb to curb)"], sig=w["sigma"],
                         s_err=r["scale"]["value"] - true_s, s_sig=r["scale"]["sigma"], s0=r["sigma0"])
    return name, out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    jobs = [(nm, 1000 + k) for nm in SCEN for k in range(n)]
    with Pool(2) as pool:
        res = pool.map(one, jobs)
    summary = {}
    for nm in SCEN:
        rows = [o for name, o in res if name == nm]
        s = {}
        for mode in ("mapillary", "wheelbase"):
            e = np.array([r[mode]["err"] for r in rows])
            sg = np.array([r[mode]["sig"] for r in rows])
            se = np.array([r[mode]["s_err"] for r in rows])
            ss = np.array([r[mode]["s_sig"] for r in rows])
            s[mode] = dict(bias_m=round(e.mean(), 4), rmse_m=round(np.sqrt((e ** 2).mean()), 4),
                           pred_sigma_m=round(sg.mean(), 4), scale_rmse_pct=round(100 * np.sqrt((se ** 2).mean()), 3),
                           scale_pred_pct=round(100 * ss.mean(), 3), sigma0=round(np.mean([r[mode]["s0"] for r in rows]), 2))
        summary[nm] = s
        print(nm)
        for mode in s:
            print("   ", mode, s[mode])
    json.dump(summary, open("montecarlo.json", "w"), indent=1)
