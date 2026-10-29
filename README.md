# street-width-lab

Experiment: measure road and sidewalk widths from Mapillary imagery, using the **wheelbase of parked cars as scale bars**.
Mapillary poses come from GPS-derived structure from motion, so their scale is only good to a few percent.
A car of known model has a known distance between its front and rear hub centres. Clicking both hubs in several frames turns each car into a scale bar for a bundle adjustment.

## Parts

| File | Purpose |
|---|---|
| `street-width-lab.html` | Browser tool. Draw a street segment on an OpenFreeMap/MapLibre map, fetch Mapillary images and poses, click wheel hubs and edge points, export a JSON project. |
| `mly_ba.py` | Bundle adjustment on the exported JSON. Solves image orientations, positions, points and one global scale. Wheelbases enter as scale bars. Reports widths with propagated σ in three modes: `mapillary` (GPS scale only), `wheelbase`, `combined`. |
| `sim_street.py`, `montecarlo.py` | Simulated street with a planted GPS scale error. Used to check the adjustment and its predicted σ. |
| `test/` | Headless end-to-end test (mocked Mapillary and OpenFreeMap) and camera-model parity check between the JS and Python implementations. |

## Use

1. Open `street-width-lab.html` in Chrome or Firefox (if the browser blocks requests from `file://`, run `python -m http.server` here and open `http://localhost:8000/street-width-lab.html`).
2. Paste a Mapillary client token (Mapillary dashboard → Developers). It stays in your browser.
3. Draw the segment, search, pick one sequence (360° if available).
4. Add cars with their wheelbase, click front and rear hub centres in every frame where they are visible within about 12 m. Click edge points (curb joints, wall corners) in two or more frames.
5. Enter tape measurements, export the JSON, then:

```bash
pip install -r requirements.txt
python mly_ba.py exported.json --report report.md
```

## Conventions

OpenSfM / MapillaryJS: `computed_rotation` is an angle-axis vector mapping local ENU to the camera frame (x right, y down, z forward); camera centre from `computed_geometry` and `computed_altitude`. Image coordinates are normalized by `max(width, height)`.
The tool's quick-look panel compares each camera axis with `computed_compass_angle`; more than about 3° means the convention does not match the data.

## Simulation results (idealized: exact wheelbases, perfect matching)

Road width, 8.00 m between curbs, 3% planted scale error, 20 runs per scenario.

| Capture and clicking | RMSE, Mapillary scale | RMSE, wheelbase scale |
|---|---|---|
| Forward camera, hubs within 22 m, 4 frames each | 24 cm | 8 cm |
| Forward camera, hubs within 12 m, every frame | 23 cm | 5 cm |
| 360°, 2048 px | 24 cm | 1.4 cm |
| 360°, full resolution | 24 cm | 0.5 cm |

Raw numbers in `montecarlo_results.json`. Reproduce with `python montecarlo.py 20`.

## Tests

```bash
cd test && npm install && python render_sim.py && python test_e2e.py && python parity.py
```

## Status

Validated on simulated data only. First real-data run pending.
