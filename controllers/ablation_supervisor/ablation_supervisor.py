"""One-button lecture demos: runs a suite of classroom configurations
back-to-back, resets the world between runs, prints a results table.

Suites: drift | noise | damage | explore | ekf | backends
Usage: controllerArgs [ "--suite", "noise", "--limit", "180" ]
Delete ablation_state.json to rerun. Suite "none" (default) idles.
"""

import argparse
import glob
import json
import os
import shutil

from controller import Supervisor

SUITES = {
    "drift":   [("odom", {"lesson": 1}), ("sm", {"lesson": 3})],
    "noise":   [("clean", {"lesson": 5}),
                ("gauss15", {"lesson": 5, "noise": 2}),
                ("drop30", {"lesson": 5, "noise": 3}),
                ("ghosts", {"lesson": 5, "noise": 4})],
    "damage":  [("sm", {"lesson": 3}),
                ("sm_dmg", {"lesson": 3, "odom_damage": True}),
                ("ekf", {"lesson": 9}),
                ("ekf_dmg", {"lesson": 9, "odom_damage": True})],
    "explore": [("wander", {"lesson": 1}), ("frontier", {"lesson": 8})],
    "ekf":     [("ekf_slam", {"lesson": 9}),
                ("ekf_loc", {"lesson": 9, "ekf_known": True}),
                ("ekf_loc_dmg", {"lesson": 9, "ekf_known": True,
                                 "odom_damage": True})],
    "backends": [("linear", {"lesson": 6, "posegraph": False}),
                 ("gaussnewton", {"lesson": 6, "posegraph": True})],
}


def _summary(rows):
    print("\n=== ablation results ===")
    print("run              sm_ATE  odom_ATE  ekf_ATE  cov%")
    for r in rows:
        fmt = lambda v: ("%6.3f" % v) if isinstance(v, float) else "     -"
        print("%-16s %s  %s  %s  %5.1f" % (
            r["tag"], fmt(r["sm"]), fmt(r["odom"]), fmt(r["ekf"]),
            r["cov"] if r["cov"] is not None else -1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="none")
    ap.add_argument("--limit", type=float, default=180.0)
    ap.add_argument("--fresh", action="store_true")
    args, _ = ap.parse_known_args()

    here = os.path.dirname(os.path.abspath(__file__))
    teach = os.path.abspath(os.path.join(here, os.pardir, "slam_teacher"))
    ep_dir = os.path.join(here, "episodes")
    os.makedirs(ep_dir, exist_ok=True)
    state_path = os.path.join(here, "ablation_state.json")
    cfg_path = os.path.join(here, "teach_config.json")
    fin_path = os.path.join(teach, "ate_final.txt")
    csv_path = os.path.join(here, "ablation_results.csv")

    runs = SUITES.get(args.suite)
    if not runs:
        print("[ablation] suite '%s' unknown or 'none' -- idling. "
              "Options: %s" % (args.suite, ", ".join(SUITES)))
        return

    state = {"run": 1, "rows": []}
    if not args.fresh and os.path.exists(state_path):
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (OSError, ValueError):
            pass
    run_i = state["run"]
    if run_i > len(runs):
        _summary(state["rows"])
        return

    def write_cfg(i):
        tag, cfg = runs[i]
        cfg = dict(cfg)
        cfg["tag"] = "%s_%s" % (args.suite, tag)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

    write_cfg(run_i - 1)
    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    print("[ablation] suite '%s': %d runs x <=%g s" % (
        args.suite, len(runs), args.limit))

    last_t = 0.0
    started = False

    def read_final():
        try:
            with open(fin_path) as f:
                v = f.read().strip().split(",")
                return {"sm": float(v[0]), "odom": float(v[1]),
                        "ekf": float(v[2]), "cov": float(v[4])}
        except (OSError, ValueError, IndexError):
            return None

    def finish(cov_note):
        nonlocal run_i, started
        tag, _cfg = runs[run_i - 1]
        v = read_final() or {}
        row = {"tag": tag, "sm": v.get("sm"), "odom": v.get("odom"),
               "ekf": v.get("ekf"), "cov": v.get("cov"), "note": cov_note}
        state["rows"].append(row)
        for src, dstname in ((os.path.join(teach, "pose_log.csv"),
                              "abl_%s_pose_log.csv" % tag),
                             (os.path.join(teach, "cov_log.csv"),
                              "abl_%s_cov_log.csv" % tag),
                             (os.path.join(teach, "map_classroom.pgm"),
                              "abl_%s_map.pgm" % tag)):
            if os.path.exists(src):
                try:
                    shutil.copyfile(src, os.path.join(ep_dir, dstname))
                except OSError:
                    pass
        try:
            os.remove(fin_path)
        except OSError:
            pass
        state["run"] = run_i + 1
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)
        print("[ablation] '%s' done (%s): sm=%s odom=%s ekf=%s cov=%s" % (
            tag, cov_note, row["sm"], row["odom"], row["ekf"], row["cov"]))
        run_i += 1
        if run_i > len(runs):
            _summary(state["rows"])
            try:
                os.remove(cfg_path)
            except OSError:
                pass
            print("[ablation] complete -- delete ablation_state.json "
                  "to rerun.")
        else:
            write_cfg(run_i - 1)
            print("[ablation] resetting for '%s' ..." % runs[run_i - 1][0])
            sup.simulationReset()
        started = False

    while sup.step(ts) != -1:
        if run_i > len(runs):
            continue
        t = sup.getTime()
        if t < last_t - 0.5:
            last_t = 0.0
        last_t = t
        if not started:
            started = True
            continue
        v = read_final()
        cov = v["cov"] if v else 0.0
        if args.suite == "explore" and cov >= 97.0:
            finish("cov97")
        elif t >= args.limit:
            finish("time")


if __name__ == "__main__":
    main()
