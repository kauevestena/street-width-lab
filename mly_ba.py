#!/usr/bin/env python3
"""
mly_ba.py - bundle adjustment of Mapillary images with car wheelbases as scale bars.

Input: the JSON project exported by the Street-width lab tool (format "mly-scalebar-project/1").

Model
-----
* Camera models as in OpenSfM / MapillaryJS: perspective (f, k1, k2), fisheye (f, k1, k2)
  and spherical (equirectangular). Image coordinates are OpenSfM normalized coordinates,
  i.e. pixels centred on the principal point and divided by max(width, height).
* computed_rotation is an angle-axis vector R mapping local ENU (east, north, up) to the camera
  frame (x right, y down, z forward); camera centre C = ENU(computed_geometry, computed_altitude).
  X_cam = R (X_enu - C).
* Unknowns: per image a small rotation dw (R = exp(dw) R0) and a position correction dC,
  one global scale s that rescales Mapillary's camera trajectory about its centroid
  (C = c + s (C0 - c) + dC), and the object points (wheel hubs, edge points).
* Observations: image clicks; pseudo-observations dw = 0 and dC = 0 (Mapillary poses as priors);
  optionally s = 1 +- sigma_s; wheelbases |X_front - X_rear| = L +- sigma_L.

Modes
-----
  mapillary : s fixed to 1, no scale bars       -> widths at Mapillary's GPS-derived scale
  wheelbase : s free, wheelbases as scale bars  -> scale comes from the cars only
  combined  : s with prior 1 +- sigma_s, plus wheelbases
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------------- geodesy
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(lon, lat, h):
    lon, lat = np.radians(lon), np.radians(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def _enu_matrix(lon0, lat0):
    lo, la = math.radians(lon0), math.radians(lat0)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def geodetic_to_enu(lon, lat, h, ref):
    x = geodetic_to_ecef(np.asarray(lon, float), np.asarray(lat, float), np.asarray(h, float))
    x0 = geodetic_to_ecef(ref[0], ref[1], ref[2])
    return (x - x0) @ _enu_matrix(ref[0], ref[1]).T


def ecef_to_geodetic(x):
    x = np.atleast_2d(x)
    lon = np.arctan2(x[:, 1], x[:, 0])
    p = np.hypot(x[:, 0], x[:, 1])
    lat = np.arctan2(x[:, 2], p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(x[:, 2], p * (1 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lon), np.degrees(lat), h


def enu_to_geodetic(enu, ref):
    enu = np.atleast_2d(enu)
    x0 = geodetic_to_ecef(ref[0], ref[1], ref[2])
    return ecef_to_geodetic(enu @ _enu_matrix(ref[0], ref[1]) + x0)


# ----------------------------------------------------------------------------- cameras
CAM_CODES = {"perspective": 0, "fisheye": 1, "spherical": 2, "equirectangular": 2}


def project_norm(xc, code, f, k1, k2):
    """Camera-frame points (N,3) -> OpenSfM normalized coords (N,2). Vectorized over mixed types."""
    out = np.zeros((len(xc), 2))
    X, Y, Z = xc[:, 0], xc[:, 1], xc[:, 2]
    m = code == 0
    if m.any():
        z = np.maximum(Z[m], 1e-3)
        x, y = X[m] / z, Y[m] / z
        r2 = x * x + y * y
        d = 1 + k1[m] * r2 + k2[m] * r2 * r2
        out[m, 0] = f[m] * d * x
        out[m, 1] = f[m] * d * y
    m = code == 1
    if m.any():
        l = np.hypot(X[m], Y[m])
        th = np.arctan2(l, Z[m])
        d = 1 + k1[m] * th ** 2 + k2[m] * th ** 4
        l = np.maximum(l, 1e-12)
        out[m, 0] = f[m] * th * d * X[m] / l
        out[m, 1] = f[m] * th * d * Y[m] / l
    m = code == 2
    if m.any():
        lon = np.arctan2(X[m], Z[m])
        lat = np.arctan2(-Y[m], np.hypot(X[m], Z[m]))
        out[m, 0] = lon / (2 * np.pi)
        out[m, 1] = -lat / (2 * np.pi)
    return out


def bearing_norm(u, v, code, f, k1, k2):
    """Inverse of project_norm for one point: normalized coords -> unit bearing (camera frame)."""
    if code == 2:
        lon, lat = 2 * np.pi * u, -2 * np.pi * v
        return np.array([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)])
    rho = math.hypot(u, v) / f
    if rho < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    t = rho
    for _ in range(30):  # Newton on t*d(t) = rho
        if code == 0:
            g = t * (1 + k1 * t * t + k2 * t ** 4) - rho
            dg = 1 + 3 * k1 * t * t + 5 * k2 * t ** 4
        else:
            g = t * (1 + k1 * t * t + k2 * t ** 4) - rho
            dg = 1 + 3 * k1 * t * t + 5 * k2 * t ** 4
        t -= g / dg
    du, dv = u / (rho * f), v / (rho * f)  # unit direction in the image plane
    if code == 0:
        b = np.array([du * t, dv * t, 1.0])
    else:
        b = np.array([du * math.sin(t), dv * math.sin(t), math.cos(t)])
    return b / np.linalg.norm(b)


# ----------------------------------------------------------------------------- project
@dataclass
class Img:
    id: str
    code: int
    f: float
    k1: float
    k2: float
    w: int
    h: int
    R0: np.ndarray
    C0: np.ndarray
    compass: float | None
    sequence: str = ""

    @property
    def s(self):
        return max(self.w, self.h)


@dataclass
class Project:
    ref: tuple
    images: list
    points: list            # ids
    point_info: list        # dicts
    obs: np.ndarray         # (n, 2) normalized coords
    obs_img: np.ndarray
    obs_pt: np.ndarray
    obs_sigma: np.ndarray   # normalized units
    obs_display_long: np.ndarray
    cars: list              # dicts with keys id, front, rear (point idx), L, sigma
    groups: dict            # group -> list of point idx
    pairs: list
    ground_truth: dict
    axis_dir: np.ndarray | None
    warnings: list = field(default_factory=list)


def load_project(src) -> Project:
    d = json.load(open(src)) if isinstance(src, str) else src
    warn = []
    settings = d.get("settings", {})
    sig_px = float(settings.get("sigma_click_px", 1.5))
    disp_long_default = float(settings.get("display_long_side", 2048))

    raw_imgs = d["images"]
    if d.get("reference"):
        r = d["reference"]
        ref = (float(r["lon"]), float(r["lat"]), float(r.get("alt", 0.0)))
    else:
        lons = [im["computed_geometry"]["coordinates"][0] for im in raw_imgs]
        lats = [im["computed_geometry"]["coordinates"][1] for im in raw_imgs]
        alts = [im.get("computed_altitude", 0.0) or 0.0 for im in raw_imgs]
        ref = (float(np.mean(lons)), float(np.mean(lats)), float(np.mean(alts)))

    images, idmap = [], {}
    for im in raw_imgs:
        if im.get("computed_rotation") is None or im.get("computed_geometry") is None:
            warn.append(f"image {im['id']}: no computed pose, skipped")
            continue
        if int(im.get("exif_orientation") or 1) not in (0, 1):
            warn.append(f"image {im['id']}: exif_orientation {im.get('exif_orientation')} not supported, skipped")
            continue
        ctype = (im.get("camera_type") or "perspective").lower()
        code = CAM_CODES.get(ctype)
        if code is None:
            warn.append(f"image {im['id']}: camera type {ctype} unsupported, skipped")
            continue
        p = list(im.get("camera_parameters") or []) + [0.0, 0.0, 0.0]
        f, k1, k2 = (float(p[0]) or 0.85, float(p[1]), float(p[2])) if code != 2 else (1.0, 0.0, 0.0)
        lon, lat = im["computed_geometry"]["coordinates"][:2]
        alt = im.get("computed_altitude")
        if alt is None:
            alt = im.get("altitude") or ref[2]
            warn.append(f"image {im['id']}: no computed_altitude, used {alt}")
        C0 = geodetic_to_enu(lon, lat, alt, ref)
        R0 = Rotation.from_rotvec(np.asarray(im["computed_rotation"], float)).as_matrix()
        idmap[str(im["id"])] = len(images)
        images.append(Img(str(im["id"]), code, f, k1, k2, int(im["width"]), int(im["height"]), R0, C0,
                          im.get("computed_compass_angle", im.get("compass_angle")), str(im.get("sequence", ""))))

    pts_by_id = {str(p["id"]): p for p in d.get("points", [])}
    obs_by_pt = {}
    for o in d.get("observations", []):
        if str(o["image"]) not in idmap or str(o["point"]) not in pts_by_id:
            continue
        obs_by_pt.setdefault(str(o["point"]), []).append(o)

    points, info = [], []
    for pid, plist in obs_by_pt.items():
        if len({o["image"] for o in plist}) < 2:
            warn.append(f"point {pid}: observed in fewer than 2 images, not used")
            continue
        points.append(pid)
        info.append(pts_by_id[pid])
    pidx = {p: i for i, p in enumerate(points)}

    ob, oi, op, osg, odl = [], [], [], [], []
    for pid in points:
        for o in obs_by_pt[pid]:
            im = images[idmap[str(o["image"])]]
            xpx, ypx = float(o["x"]) * im.w, float(o["y"]) * im.h
            ob.append(((xpx - im.w / 2) / im.s, (ypx - im.h / 2) / im.s))
            oi.append(idmap[str(o["image"])])
            op.append(pidx[pid])
            dl = float(o.get("display_long", disp_long_default))
            odl.append(dl)
            osg.append(float(o.get("sigma_px", sig_px)) / dl)

    cars = []
    for c in d.get("cars", []):
        fid, rid = f"{c['id']}.F", f"{c['id']}.R"
        if fid in pidx and rid in pidx and c.get("wheelbase_m"):
            cars.append(dict(id=c["id"], model=c.get("model", ""), front=pidx[fid], rear=pidx[rid],
                             L=float(c["wheelbase_m"]), sigma=float(c.get("sigma_m") or 0.01)))
        else:
            warn.append(f"car {c['id']}: needs both hubs in >=2 images and a wheelbase; not used as scale bar")

    groups = {}
    for i, pinfo in enumerate(info):
        g = pinfo.get("group")
        if pinfo.get("kind") == "edge" and g:
            groups.setdefault(g, []).append(i)

    pairs = d.get("width_pairs") or [
        {"name": "road (curb to curb)", "a": "curb_L", "b": "curb_R"},
        {"name": "sidewalk left", "a": "curb_L", "b": "facade_L"},
        {"name": "sidewalk right", "a": "curb_R", "b": "facade_R"},
    ]
    gt = {g["pair"]: float(g["width_m"]) for g in d.get("ground_truth", []) if g.get("width_m") not in (None, "")}

    axis = None
    seg = d.get("segment")
    if seg and seg.get("start") and seg.get("end"):
        a = geodetic_to_enu(seg["start"][0], seg["start"][1], ref[2], ref)
        b = geodetic_to_enu(seg["end"][0], seg["end"][1], ref[2], ref)
        v = (b - a)[:2]
        if np.linalg.norm(v) > 0:
            axis = v / np.linalg.norm(v)

    return Project(ref, images, points, info, np.array(ob).reshape(-1, 2), np.array(oi, int), np.array(op, int),
                   np.array(osg), np.array(odl), cars, groups, pairs, gt, axis, warn)


# ----------------------------------------------------------------------------- helpers
def triangulate(proj: Project, R, C):
    """Linear (midpoint) triangulation of all points from bearings."""
    X = np.zeros((len(proj.points), 3))
    A = np.zeros((len(proj.points), 3, 3))
    b = np.zeros((len(proj.points), 3))
    for k in range(len(proj.obs)):
        im = proj.images[proj.obs_img[k]]
        i, j = proj.obs_img[k], proj.obs_pt[k]
        bc = bearing_norm(proj.obs[k, 0], proj.obs[k, 1], im.code, im.f, im.k1, im.k2)
        bw = R[i].T @ bc
        P = np.eye(3) - np.outer(bw, bw)
        A[j] += P
        b[j] += P @ C[i]
    for j in range(len(X)):
        X[j] = np.linalg.lstsq(A[j], b[j], rcond=None)[0]
    return X


def heading_check(proj: Project):
    out = []
    for im in proj.images:
        if im.compass is None:
            continue
        ax = im.R0.T @ np.array([0, 0, 1.0])
        hd = math.degrees(math.atan2(ax[0], ax[1])) % 360
        out.append(((hd - float(im.compass) + 180) % 360) - 180)
    return np.array(out)


def width_between(XA, XB, axis_dir=None):
    """Horizontal distance between two (roughly parallel) edge lines given their points (ENU)."""
    A2, B2 = XA[:, :2], XB[:, :2]
    if (len(A2) >= 2 or len(B2) >= 2) and len(A2) + len(B2) >= 3:
        cen = np.vstack([A2 - A2.mean(0), B2 - B2.mean(0)])
        w, v = np.linalg.eigh(cen.T @ cen)
        d = v[:, -1]
        if axis_dir is not None and np.dot(d, axis_dir) < 0:
            d = -d
    elif axis_dir is not None:
        d = axis_dir
    else:
        return float("nan"), None
    n = np.array([-d[1], d[0]])
    return float(abs(n @ (B2.mean(0) - A2.mean(0)))), d


# ----------------------------------------------------------------------------- adjustment
@dataclass
class Options:
    mode: str = "wheelbase"         # mapillary | wheelbase | combined
    sigma_rot_deg: float = 1.0
    sigma_pos_m: float = 0.25
    sigma_scale: float = 0.05
    loss: str = "linear"            # or 'huber'
    f_scale: float = 3.0


class BundleAdjustment:
    def __init__(self, proj: Project, opt: Options):
        self.p, self.o = proj, opt
        self.nI, self.nP = len(proj.images), len(proj.points)
        self.R0 = np.stack([im.R0 for im in proj.images])
        self.C0 = np.stack([im.C0 for im in proj.images])
        self.cbar = self.C0.mean(0)
        self.use_s = opt.mode in ("wheelbase", "combined")
        self.use_bars = opt.mode in ("wheelbase", "combined")
        self.code = np.array([proj.images[i].code for i in proj.obs_img])
        self.f = np.array([proj.images[i].f for i in proj.obs_img])
        self.k1 = np.array([proj.images[i].k1 for i in proj.obs_img])
        self.k2 = np.array([proj.images[i].k2 for i in proj.obs_img])
        self.sr = math.radians(opt.sigma_rot_deg)
        # layout
        self.i_w = 0
        self.i_c = 3 * self.nI
        self.i_s = 6 * self.nI
        self.i_x = self.i_s + (1 if self.use_s else 0)
        self.n = self.i_x + 3 * self.nP
        self.bars = proj.cars if self.use_bars else []

    def unpack(self, x):
        dw = x[self.i_w:self.i_c].reshape(-1, 3)
        dC = x[self.i_c:self.i_s].reshape(-1, 3)
        s = x[self.i_s] if self.use_s else 1.0
        X = x[self.i_x:].reshape(-1, 3)
        R = Rotation.from_rotvec(dw).as_matrix() @ self.R0
        C = self.cbar + s * (self.C0 - self.cbar) + dC
        return R, C, s, X

    def residuals(self, x):
        p, o = self.p, self.o
        R, C, s, X = self.unpack(x)
        xc = np.einsum("nij,nj->ni", R[p.obs_img], X[p.obs_pt] - C[p.obs_img])
        pr = project_norm(xc, self.code, self.f, self.k1, self.k2)
        r_img = ((pr - p.obs) / p.obs_sigma[:, None]).ravel()
        # perspective points behind the camera: heavy penalty
        bad = (self.code == 0) & (xc[:, 2] <= 0.05)
        if bad.any():
            r_img = r_img.reshape(-1, 2)
            r_img[bad] = 1e3
            r_img = r_img.ravel()
        r_rot = x[self.i_w:self.i_c] / self.sr
        r_pos = x[self.i_c:self.i_s] / o.sigma_pos_m
        parts = [r_img, r_rot, r_pos]
        if self.use_s and o.mode == "combined":
            parts.append(np.array([(s - 1.0) / o.sigma_scale]))
        if self.bars:
            parts.append(np.array([(np.linalg.norm(X[c["front"]] - X[c["rear"]]) - c["L"]) / c["sigma"]
                                   for c in self.bars]))
        return np.concatenate(parts)

    def sparsity(self):
        p = self.p
        m = 2 * len(p.obs) + 6 * self.nI + (1 if (self.use_s and self.o.mode == "combined") else 0) + len(self.bars)
        S = lil_matrix((m, self.n), dtype=int)
        for k in range(len(p.obs)):
            i, j = p.obs_img[k], p.obs_pt[k]
            for r in (2 * k, 2 * k + 1):
                S[r, self.i_w + 3 * i:self.i_w + 3 * i + 3] = 1
                S[r, self.i_c + 3 * i:self.i_c + 3 * i + 3] = 1
                S[r, self.i_x + 3 * j:self.i_x + 3 * j + 3] = 1
                if self.use_s:
                    S[r, self.i_s] = 1
        r0 = 2 * len(p.obs)
        for q in range(6 * self.nI):
            S[r0 + q, q] = 1
        r0 += 6 * self.nI
        if self.use_s and self.o.mode == "combined":
            S[r0, self.i_s] = 1
            r0 += 1
        for c in self.bars:
            for j in (c["front"], c["rear"]):
                S[r0, self.i_x + 3 * j:self.i_x + 3 * j + 3] = 1
            r0 += 1
        return S

    def solve(self):
        p = self.p
        X0 = triangulate(p, self.R0, self.C0)
        x0 = np.zeros(self.n)
        if self.use_s:
            x0[self.i_s] = 1.0
        x0[self.i_x:] = X0.ravel()
        res = least_squares(self.residuals, x0, jac_sparsity=self.sparsity(), method="trf",
                            x_scale="jac", loss=self.o.loss, f_scale=self.o.f_scale,
                            xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=200)
        self.res, self.X_init = res, X0
        J = res.jac.toarray() if hasattr(res.jac, "toarray") else np.asarray(res.jac)
        r = res.fun
        m, n = J.shape
        N = J.T @ J
        try:
            Q = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            Q = np.linalg.pinv(N)
        dof = m - n
        s0 = math.sqrt(float(r @ r) / dof) if dof > 0 else float("nan")
        red = 1.0 - np.einsum("ij,jk,ik->i", J, Q, J)
        self.J, self.Q, self.s0, self.dof, self.red = J, Q, s0, dof, red
        return res

    # ---- results
    def results(self):
        p = self.p
        R, C, s, X = self.unpack(self.res.x)
        s0 = self.s0
        k = max(s0, 1.0)  # conservative: do not shrink a-priori precision when s0 < 1
        cov = self.Q * k * k
        out = dict(mode=self.o.mode, sigma0=s0, dof=self.dof, n_images=self.nI, n_points=self.nP,
                   n_obs=len(p.obs), converged=bool(self.res.success), nfev=int(self.res.nfev))
        if self.use_s:
            out["scale"] = dict(value=float(s), sigma=float(math.sqrt(cov[self.i_s, self.i_s])))
        else:
            out["scale"] = dict(value=1.0, sigma=0.0)
        # pose corrections
        dw = self.res.x[self.i_w:self.i_c].reshape(-1, 3)
        dC = self.res.x[self.i_c:self.i_s].reshape(-1, 3)
        out["pose_corrections"] = dict(rot_rms_deg=float(np.degrees(np.sqrt((dw ** 2).sum(1).mean()))),
                                       rot_max_deg=float(np.degrees(np.sqrt((dw ** 2).sum(1)).max())),
                                       pos_rms_m=float(np.sqrt((dC ** 2).sum(1).mean())),
                                       pos_max_m=float(np.sqrt((dC ** 2).sum(1)).max()))
        # reprojection residuals in display pixels
        xc = np.einsum("nij,nj->ni", R[p.obs_img], X[p.obs_pt] - C[p.obs_img])
        pr = project_norm(xc, self.code, self.f, self.k1, self.k2)
        v_px = (pr - p.obs) * p.obs_display_long[:, None]
        nob = len(p.obs)
        w_std = (self.res.fun[:2 * nob] / np.sqrt(np.maximum(self.red[:2 * nob], 1e-9))).reshape(-1, 2)
        out["reprojection_rms_px"] = float(np.sqrt((v_px ** 2).sum(1).mean()))
        worst = np.argsort(-np.abs(w_std).max(1))[:8]
        out["largest_residuals"] = [dict(point=p.points[p.obs_pt[q]], image=p.images[p.obs_img[q]].id,
                                         v_px=[round(float(v_px[q, 0]), 2), round(float(v_px[q, 1]), 2)],
                                         w=round(float(np.abs(w_std[q]).max()), 2)) for q in worst]
        # points
        pts = []
        for j, pid in enumerate(p.points):
            sl = slice(self.i_x + 3 * j, self.i_x + 3 * j + 3)
            sj = np.sqrt(np.diag(cov[sl, sl]))
            m = p.obs_pt == j
            pts.append(dict(id=pid, enu=[round(float(v), 4) for v in X[j]], sigma=[round(float(v), 4) for v in sj],
                            n_obs=int(m.sum()), rms_px=round(float(np.sqrt((v_px[m] ** 2).sum(1).mean())), 2)))
        out["points"] = pts
        # cars
        cars = []
        for c in p.cars:
            d0 = float(np.linalg.norm(self.X_init[c["front"]] - self.X_init[c["rear"]]))
            d1 = float(np.linalg.norm(X[c["front"]] - X[c["rear"]]))
            cars.append(dict(id=c["id"], model=c["model"], L=c["L"], d_mapillary=round(d0, 4),
                             ratio=round(c["L"] / d0, 4), d_adjusted=round(d1, 4), v=round(d1 - c["L"], 4)))
        out["cars"] = cars
        # widths with error propagation
        widths = []
        for pr_ in p.pairs:
            A, B = p.groups.get(pr_["a"], []), p.groups.get(pr_["b"], [])
            if not A or not B:
                continue
            idx = A + B

            def wfun(flat):
                P = flat.reshape(-1, 3)
                return width_between(P[:len(A)], P[len(A):], p.axis_dir)[0]
            flat = X[idx].ravel()
            w0 = wfun(flat)
            g = np.zeros_like(flat)
            for t in range(len(flat)):
                e = np.zeros_like(flat); e[t] = 1e-5
                g[t] = (wfun(flat + e) - wfun(flat - e)) / 2e-5
            cols = np.concatenate([np.arange(self.i_x + 3 * j, self.i_x + 3 * j + 3) for j in idx])
            sw = float(math.sqrt(max(g @ cov[np.ix_(cols, cols)] @ g, 0)))
            _, d = width_between(X[A], X[B], p.axis_dir)
            n = np.array([-d[1], d[0]]) if d is not None else None
            offs = []
            if n is not None:
                a0 = X[A][:, :2].mean(0)
                offs = [round(float(abs(n @ (X[j][:2] - a0))), 3) for j in B]
            item = dict(name=pr_["name"], a=pr_["a"], b=pr_["b"], width=round(w0, 4), sigma=round(sw, 4),
                        nA=len(A), nB=len(B), per_point_offsets=offs)
            if pr_["name"] in p.ground_truth:
                item["truth"] = p.ground_truth[pr_["name"]]
                item["error"] = round(w0 - p.ground_truth[pr_["name"]], 4)
            widths.append(item)
        out["widths"] = widths
        return out


def run(proj: Project, mode: str, **kw):
    ba = BundleAdjustment(proj, Options(mode=mode, **kw))
    ba.solve()
    return ba.results()


# ----------------------------------------------------------------------------- report
def report_md(proj: Project, results: list[dict]) -> str:
    L = []
    hd = heading_check(proj)
    L.append("# Bundle adjustment report\n")
    L.append(f"Images: {len(proj.images)} · points: {len(proj.points)} · observations: {len(proj.obs)} · "
             f"cars usable as scale bars: {len(proj.cars)}")
    if len(hd):
        L.append(f"\nConvention check (camera axis from computed_rotation vs computed_compass_angle): "
                 f"mean {hd.mean():+.2f}°, max |Δ| {np.abs(hd).max():.2f}°")
    if proj.warnings:
        L.append("\nWarnings:\n" + "\n".join(f"- {w}" for w in proj.warnings))
    L.append("\n## Widths by mode\n")
    names = [w["name"] for w in results[0]["widths"]]
    hdr = "| width | " + " | ".join(r["mode"] for r in results) + " | truth |"
    L.append(hdr)
    L.append("|" + "---|" * (len(results) + 2))
    for nm in names:
        row = [nm]
        for r in results:
            w = next((x for x in r["widths"] if x["name"] == nm), None)
            row.append(f"{w['width']:.3f} ± {w['sigma']:.3f}" if w else "–")
        row.append(f"{proj.ground_truth[nm]:.3f}" if nm in proj.ground_truth else "–")
        L.append("| " + " | ".join(row) + " |")
    L.append("\n## Adjustment summary\n")
    L.append("| mode | σ0 | dof | scale s | reproj. RMS (px) | rot. corr. RMS (°) | pos. corr. RMS (m) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        sc = r["scale"]
        L.append(f"| {r['mode']} | {r['sigma0']:.2f} | {r['dof']} | {sc['value']:.4f} ± {sc['sigma']:.4f} | "
                 f"{r['reprojection_rms_px']:.2f} | {r['pose_corrections']['rot_rms_deg']:.3f} | "
                 f"{r['pose_corrections']['pos_rms_m']:.3f} |")
    wb = next((r for r in results if r["mode"] == "wheelbase"), results[-1])
    if wb["cars"]:
        L.append("\n## Cars (scale bars)\n")
        L.append("| car | model | L (m) | Mapillary-scale d (m) | L/d | adjusted d (m) | residual (m) |")
        L.append("|---|---|---|---|---|---|---|")
        for c in wb["cars"]:
            L.append(f"| {c['id']} | {c['model']} | {c['L']:.3f} | {c['d_mapillary']:.3f} | {c['ratio']:.4f} | "
                     f"{c['d_adjusted']:.3f} | {c['v']:+.3f} |")
        rat = np.array([c["ratio"] for c in wb["cars"]])
        L.append(f"\nPer-car scale ratios: median {np.median(rat):.4f}, spread (std) {rat.std(ddof=1) if len(rat) > 1 else 0:.4f}")
    L.append("\n## Largest standardized residuals (wheelbase mode)\n")
    L.append("| point | image | v (px) | w |")
    L.append("|---|---|---|---|")
    for q in wb["largest_residuals"]:
        L.append(f"| {q['point']} | {q['image']} | {q['v_px']} | {q['w']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--modes", default="mapillary,wheelbase,combined")
    ap.add_argument("--sigma-rot-deg", type=float, default=1.0)
    ap.add_argument("--sigma-pos-m", type=float, default=0.25)
    ap.add_argument("--sigma-scale", type=float, default=0.05)
    ap.add_argument("--huber", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    proj = load_project(a.project)
    kw = dict(sigma_rot_deg=a.sigma_rot_deg, sigma_pos_m=a.sigma_pos_m, sigma_scale=a.sigma_scale,
              loss="huber" if a.huber else "linear")
    results = [run(proj, m, **kw) for m in a.modes.split(",")]
    md = report_md(proj, results)
    print(md)
    if a.report:
        open(a.report, "w").write(md)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
