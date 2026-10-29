"""End-to-end UI test with mocked Mapillary/OpenFreeMap: draw segment, search, click every observation, export."""
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, "..")
from mly_ba import geodetic_to_enu, project_norm  # noqa: E402

HERE = Path(__file__).parent
PAGE = (HERE.parent / "street-width-lab.html").resolve().as_uri()
proj = json.load(open(HERE / "sim_e2e.json"))
images = {im["id"]: im for im in proj["images"]}
STYLE = {"version": 8, "sources": {}, "layers": [{"id": "bg", "type": "background", "paint": {"background-color": "#dfe3dc"}}]}
log = []


def route(r):
    url = r.request.url
    u = urlparse(url)
    if url.startswith("file://"):
        return r.continue_()
    if "unpkg.com/maplibre-gl" in url:
        fn = "maplibre-gl.css" if url.endswith(".css") else "maplibre-gl.js"
        return r.fulfill(path=str(HERE / "node_modules/maplibre-gl/dist" / fn))
    if "fonts.g" in url:
        return r.fulfill(status=200, body="", content_type="text/css")
    if "openfreemap" in url:
        return r.fulfill(status=200, body=json.dumps(STYLE), content_type="application/json")
    if "tiles.mapillary.com" in url:
        return r.fulfill(status=404, body="")
    if u.netloc == "graph.mapillary.com":
        q = parse_qs(u.query)
        log.append(u.path)
        if u.path == "/images":
            data = []
            for im in proj["images"]:
                d = {k: v for k, v in im.items()}
                d["captured_at"] = im["captured_at"]
                d["thumb_256_url"] = f"https://img.test/{im['id']}.png"
                d["geometry"] = d["computed_geometry"]
                data.append(d)
            return r.fulfill(status=200, body=json.dumps({"data": data}), content_type="application/json",
                             headers={"access-control-allow-origin": "*"})
        iid = u.path.strip("/")
        return r.fulfill(status=200, body=json.dumps({"id": iid, "thumb_256_url": f"https://img.test/{iid}.png",
                                                      "thumb_2048_url": f"https://img.test/{iid}.png",
                                                      "thumb_original_url": f"https://img.test/{iid}.png"}),
                         content_type="application/json", headers={"access-control-allow-origin": "*"})
    if u.netloc == "img.test":
        return r.fulfill(path=str(HERE / "imgs" / Path(u.path).name), content_type="image/png")
    return r.fulfill(status=404, body="")


with sync_playwright() as pw:
    br = pw.chromium.launch()
    ctx = br.new_context(viewport={"width": 1500, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append("console: " + m.text) if m.type == "error" else None)
    page.route("**/*", route)
    page.goto(PAGE)
    page.wait_for_function("window.__swl && window.__swl.map && window.__swl.map.loaded()", timeout=20000)

    page.fill("#token", "MLY|test")
    page.dispatch_event("#token", "change")
    seg = proj["segment"]
    mid = [(seg["start"][0] + seg["end"][0]) / 2, (seg["start"][1] + seg["end"][1]) / 2]
    page.evaluate(f"window.__swl.map.jumpTo({{center: {mid}, zoom: 18.3}})")
    page.wait_for_timeout(500)
    box = page.locator("#map").bounding_box()
    page.click("#btn-draw")
    for ll in (seg["start"], seg["end"]):
        p = page.evaluate(f"(() => {{ const p = window.__swl.map.project({ll}); return [p.x, p.y]; }})()")
        page.mouse.click(box["x"] + p[0], box["y"] + p[1])
        page.wait_for_timeout(150)
    page.fill("#buffer", "25")
    page.dispatch_event("#buffer", "input")
    page.click("#btn-search")
    page.wait_for_function("document.querySelector('#search-status').classList.contains('ok')", timeout=10000)
    print("search:", page.inner_text("#search-status"))
    page.screenshot(path="shot_select.png")
    page.click("#btn-use")
    page.wait_for_function("window.__swl.V.img !== null", timeout=10000)

    # cars + edge points via the UI buttons
    for car in proj["cars"]:
        page.click("#add-car")
        row = page.locator(f'.car[data-car="{car["id"]}"]')
        row.locator(".model").fill(car["model"])
        row.locator(".model").dispatch_event("change")
        row.locator(".wb").fill(str(car["wheelbase_m"]))
        row.locator(".wb").dispatch_event("change")
    for p in proj["points"]:
        if p["kind"] != "edge":
            continue
        page.select_option("#grp-sel", p["group"])
        page.click("#add-pt")
    ids = page.evaluate("window.__swl.S.points.map(p => p.id)")
    assert set(ids) == {p["id"] for p in proj["points"]}, "point ids differ"

    # click every simulated observation through the canvas
    order = {im["id"]: i for i, im in enumerate(proj["images"])}
    obs = sorted(proj["observations"], key=lambda o: (order[o["image"]], o["point"]))
    cur = None
    cbox = page.locator("#cv").bounding_box()
    for o in obs:
        if o["image"] != cur:
            page.evaluate(f"window.__swl.showImage({order[o['image']]})")
            page.wait_for_function(f"window.__swl.V.img && window.__swl.S.images[window.__swl.S.cur].id === '{o['image']}'")
            cur = o["image"]
        page.evaluate(f"window.__swl.setActive('{o['point']}')")
        sx, sy = page.evaluate(f"""(() => {{ const V = window.__swl.V; V.z = 3; V.cx = {o['x']}; V.cy = {o['y']};
              return window.__swl.i2s({o['x']} * V.nw, {o['y']} * V.nh); }})()""")
        cbox = page.locator("#cv").bounding_box()
        page.mouse.click(cbox["x"] + sx, cbox["y"] + sy)
    n_obs = page.evaluate("window.__swl.S.observations.length")
    print("observations clicked:", n_obs, "of", len(obs))
    got = page.evaluate("window.__swl.S.observations")
    gk = {(o["image"], o["point"]): o for o in got}
    miss = [(o["image"], o["point"], round(o["x"], 3), round(o["y"], 3)) for o in obs if (o["image"], o["point"]) not in gk]
    dev = max(max(abs(gk[k]["x"] - o["x"]) * 2048, abs(gk[k]["y"] - o["y"]) * 1536) for o in obs if (k := (o["image"], o["point"])) in gk)
    print("missing:", miss[:10], " max click deviation (px):", round(dev, 3))

    # tape measurements
    page.click("summary:has-text('Widths and tape')")
    for g in proj["ground_truth"]:
        inp = page.locator(f'input[data-gt="{g["pair"]}"]')
        inp.fill(str(g["width_m"]))
        inp.dispatch_event("change")

    # JS vs Python projection parity on the Mapillary poses
    ref = page.evaluate("window.__swl.S.ref")
    worst = 0
    for im in proj["images"][::4]:
        for pid, X in list(proj["_truth"]["points"].items())[::3]:
            Xg = np.array(X)  # ENU in the simulator's REF frame -> geodetic -> JS ENU
            from mly_ba import enu_to_geodetic
            lon, lat, h = enu_to_geodetic(Xg, (proj["reference"]["lon"], proj["reference"]["lat"], proj["reference"]["alt"]))
            js = page.evaluate(f"(() => {{ const S = window.__swl.S; const X = window.__swl.toENU({lon[0]}, {lat[0]}, {h[0]}, S.ref); return window.__swl.projWorld(window.__swl.cams.get('{im['id']}'), X); }})()")
            # Python: same Mapillary pose
            from scipy.spatial.transform import Rotation
            R = Rotation.from_rotvec(im["computed_rotation"]).as_matrix()
            Cg = im["computed_geometry"]["coordinates"]
            refp = (proj["reference"]["lon"], proj["reference"]["lat"], proj["reference"]["alt"])
            C = geodetic_to_enu(Cg[0], Cg[1], im["computed_altitude"], refp)
            xc = R @ (Xg - C)
            if xc[2] < 0.1:
                continue
            uv = project_norm(xc[None], np.array([0]), np.array([0.6]), np.array([-0.08]), np.array([0.02]))[0]
            py = [(uv[0] * 2048 + 1024) / 2048, (uv[1] * 2048 + 768) / 1536]
            if js is None:
                continue
            worst = max(worst, abs(js[0] - py[0]) * 2048, abs(js[1] - py[1]) * 1536)
    print("JS vs Python projection, worst difference (px):", round(worst, 6))

    # screenshot of the measuring workspace on a frame with several clicks
    page.evaluate("window.__swl.showImage(6)")
    page.wait_for_timeout(400)
    page.evaluate("window.__swl.setActive('C3.F')")
    page.evaluate("(() => { const V = window.__swl.V; V.z = 1; V.cx = .5; V.cy = .5; })()")
    page.click("#v-fit")
    page.wait_for_timeout(300)
    page.screenshot(path="shot_measure.png")

    with page.expect_download() as dl:
        page.click("#btn-export")
    dl.value.save_as("exported.json")
    print("errors:", errors[:10])
    br.close()
