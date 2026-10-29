import json, numpy as np, sys
sys.path.insert(0, '..')
from mly_ba import geodetic_to_enu, enu_to_geodetic, project_norm, bearing_norm, CAM_CODES
from scipy.spatial.transform import Rotation
exec(open('test_e2e.py').read().split('with sync_playwright() as pw:')[0])
from sim_street import make_project
with sync_playwright() as pw:
    br = pw.chromium.launch(); page = br.new_page(); page.route("**/*", route); page.goto(PAGE)
    page.wait_for_function("window.__swl")
    for cam, params in (("spherical", []), ("fisheye", [0.45, 0.04, -0.01]), ("perspective", [0.6, -0.08, 0.02])):
        d = make_project(seed=3, cam="spherical" if cam == "spherical" else "perspective")
        for im in d["images"]:
            im["camera_type"] = cam; im["camera_parameters"] = params
        page.evaluate("d => window.__swl.loadProjectObj(d)", d)
        ref = page.evaluate("window.__swl.S.ref"); refp = (ref["lon"], ref["lat"], ref["alt"])
        worst = 0; rt = 0
        for im in d["images"][::3]:
            R = Rotation.from_rotvec(im["computed_rotation"]).as_matrix()
            g = im["computed_geometry"]["coordinates"]; C = geodetic_to_enu(g[0], g[1], im["computed_altitude"], refp)
            W, H = im["width"], im["height"]; s = max(W, H); code = CAM_CODES[cam]
            f, k1, k2 = (params + [1, 0, 0])[:3] if params else (1, 0, 0)
            for pid, X in list(d["_truth"]["points"].items())[::2]:
                lon, lat, h = enu_to_geodetic(np.array(X), (d["reference"]["lon"], d["reference"]["lat"], d["reference"]["alt"]))
                Xe = geodetic_to_enu(lon[0], lat[0], h[0], refp)
                xc = R @ (Xe - C)
                if code == 0 and xc[2] < 0.1: continue
                uv = project_norm(xc[None], np.array([code]), np.array([f]), np.array([k1]), np.array([k2]))[0]
                py = [(uv[0] * s + W / 2) / W, (uv[1] * s + H / 2) / H]
                js = page.evaluate(f"(() => {{ const w = window.__swl; const X = w.toENU({lon[0]}, {lat[0]}, {h[0]}, w.S.ref); return w.projWorld(w.cams.get('{im['id']}'), X); }})()")
                if js is None: continue
                worst = max(worst, abs(js[0] - py[0]) * W, abs(js[1] - py[1]) * H)
                # round trip: JS ray through the Python pixel, point at the true distance, reproject
                dist = float(np.linalg.norm(Xe - C))
                back = page.evaluate(f"(() => {{ const w = window.__swl, c = w.cams.get('{im['id']}'); const r = w.rayOf(c, {py[0]}, {py[1]}); return w.projWorld(c, [r.C[0]+{dist}*r.d[0], r.C[1]+{dist}*r.d[1], r.C[2]+{dist}*r.d[2]]); }})()")
                rt = max(rt, abs(back[0] - py[0]) * W, abs(back[1] - py[1]) * H)
                # Python bearing inverse too
                b = bearing_norm(uv[0], uv[1], code, f, k1, k2)
                assert np.allclose(b, xc / np.linalg.norm(xc), atol=1e-6), (cam, b, xc / np.linalg.norm(xc))
        print(f"{cam:12s} JS vs Python projection max {worst:.4f} px · JS ray/projection round trip max {rt:.2e} px")
    br.close()
