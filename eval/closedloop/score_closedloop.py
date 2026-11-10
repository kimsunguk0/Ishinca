#!/usr/bin/env python3
"""Re-aggregate AlpaSim closed-loop results for the 150-scene Pisco protocol.

  python score_closedloop.py --arm pisco4b=runs/pisco4b/*/aggregate/results-summary.json \
                             --arm teacher=runs/teacher/*/aggregate/results-summary.json

Scoring (AlpaSim ``score_criteria``): a rollout fails on at-fault collision or off-road and then scores 0; otherwise
its score is ``min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1)`` (1.0 if the recorded GT path is shorter than 5 m). The script
uses the per-rollout ``score`` written by AlpaSim and cross-checks it against this formula.

* Scenes are taken from scenes/scenes150_20261101.txt; the 3 invalid-map scenes (recorded GT path itself off-road)
  are flagged and reported separately: all150 includes them, all147 excludes them for every arm.
* Groups: holdout58 / ext60 (scenes/closedloop_ext60_20261030.json) / new32 (the rest).
* Paired differences use a scene bootstrap (scene = resampling unit, rollouts averaged within scene).
* Rear collisions are not at-fault (``collision_rear``); they are reported but do not zero the score.
"""

import argparse
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).parent / "scenes"
INVALID_PREFIXES = ("a9e8dfc7", "cc8cbc45", "29815599")  # clip-id prefixes: GT path is off-road in the map


def load_sets():
    order = (HERE / "scenes150_20261101.txt").read_text().split()
    holdout = set((HERE / "holdout58.txt").read_text().split())
    ext = set(json.loads((HERE / "closedloop_ext60_20261030.json").read_text())["scene_ids"])
    invalid = {s for s in order if any(p in s for p in INVALID_PREFIXES)}
    group = lambda s: "holdout58" if s in holdout else "ext60" if s in ext else "new32"
    return order, group, invalid


def load_arm(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in json.loads(Path(f).read_text())["rollouts"]:
                m = r["metrics"]
                prog = max(0.0, min(1.0, m["progress_clipped_rel"]))
                formula = 1.0 if m["gt_dist_traveled_m"] < 5.0 else min(prog / 0.8, 1.0)
                if m["offroad"] > 0 or m["collision_at_fault"] > 0:
                    formula = 0.0
                rows.append(dict(scene=r["clipgt_id"], score=r["score"], formula=formula,
                                 fail=int(not r["passed"]), offroad=int(m["offroad"] > 0),
                                 at_fault=int(m["collision_at_fault"] > 0), rear=int(m["collision_rear"] > 0)))
    return rows


def per_scene(rows):
    out = {}
    for r in rows:
        out.setdefault(r["scene"], []).append(r["score"])
    return {s: sum(v) / len(v) for s, v in out.items()}


def summarize(name, rows, scenes):
    rr = [r for r in rows if r["scene"] in scenes]
    n = max(len(rr), 1)
    return (f"{name:12s} scenes {len({r['scene'] for r in rr}):3d} rollouts {len(rr):3d} score {sum(r['score'] for r in rr)/n:.3f} "
            f"failed {sum(r['fail'] for r in rr)} (offroad {sum(r['offroad'] for r in rr)}, at-fault {sum(r['at_fault'] for r in rr)}) "
            f"rear {sum(r['rear'] for r in rr)}")


def paired_bootstrap(a, b, scenes, n_boot=20000, seed=0):
    common = [s for s in scenes if s in a and s in b]
    d = [a[s] - b[s] for s in common]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)], len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=GLOB", help="repeatable; first arm is the baseline")
    ap.add_argument("--baseline", help="arm name for paired differences (default: first)")
    a = ap.parse_args()
    order, group, invalid = load_sets()
    arms = {}
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        arms[name] = load_arm([pat])
    base = a.baseline or next(iter(arms))
    sets = {g: {s for s in order if group(s) == g} for g in ("holdout58", "new32", "ext60")}
    sets["all150"] = set(order)
    sets["all147"] = set(order) - invalid
    for g, scenes in sets.items():
        print(f"== {g} ({len(scenes)} scenes)")
        for name, rows in arms.items():
            print("  " + summarize(name, rows, scenes))
    print("== paired difference vs", base, "(all150 / all147; 95% scene bootstrap)")
    for name, rows in arms.items():
        if name == base:
            continue
        for g in ("all150", "all147"):
            d, lo, hi, n = paired_bootstrap(per_scene(rows), per_scene(arms[base]), sets[g])
            print(f"  {name} - {base} {g}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n}")
    for name, rows in arms.items():
        bad = sum(abs(r["score"] - r["formula"]) > 1e-6 for r in rows)
        print(f"[check] {name}: {bad}/{len(rows)} rollouts differ from the documented score formula")


if __name__ == "__main__":
    main()
