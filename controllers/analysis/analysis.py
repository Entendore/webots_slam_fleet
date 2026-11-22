"""Offline analysis v3: ATE summaries, trajectory + ATE plots, coverage
curves, kidnap/closure event markers.

  python analysis.py pose_log.csv --plot --cov
  python analysis.py a.csv --compare b.csv
"""

import argparse
import csv
import math
import os


def load_pose(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def fl(k):
                v = r.get(k, "nan")
                return None if v in ("nan", "", None) else float(v)
            rows.append({"t": fl("t"), "e": (fl("ex"), fl("ey")),
                         "r": (fl("rx"), fl("ry")),
                         "t_": (fl("tx"), fl("ty")),
                         "k": (fl("kx"), fl("ky"))})
    return rows


def load_events(path):
    out = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                out.append((float(r["t"]), r["event"]))
    except (OSError, ValueError, KeyError):
        pass
    return out


def load_cov(path):
    out = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                out.append((float(r["t"]), float(r["coverage"]),
                            r["lesson"], r["mode"]))
    except (OSError, ValueError, KeyError):
        pass
    return out


def summarize(rows, label, key):
    errs = [math.hypot(r[key][0] - r["t_"][0], r[key][1] - r["t_"][1])
            for r in rows
            if r["t_"][0] is not None and r[key][0] is not None]
    if not errs:
        return "%-10s (no data)" % label
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    return ("%-10s RMSE %.3f  mean %.3f  max %.3f m  (%d samples)"
            % (label, rmse, sum(errs) / len(errs), max(errs), len(errs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--compare")
    ap.add_argument("--cov", default=None)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(args.log))
    cov_path = args.cov or os.path.join(base, "cov_log.csv")
    ev_path = args.events or os.path.join(base, "events.csv")

    rows = load_pose(args.log)
    print("file:", args.log, "| %d samples" % len(rows))
    print(summarize(rows, "estimate", "e"))
    print(summarize(rows, "raw-odom", "r"))
    print(summarize(rows, "ekf", "k"))
    if args.compare:
        rows2 = load_pose(args.compare)
        print("\ncompare:", args.compare)
        print(summarize(rows2, "estimate", "e"))
        print(summarize(rows2, "raw-odom", "r"))
        print(summarize(rows2, "ekf", "k"))

    if not args.plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return

    events = load_events(ev_path)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for key, col, lab in (("t_", "g", "truth"), ("e", "r", "estimate"),
                          ("r", "orange", "raw odom"), ("k", "m", "ekf")):
        pts = [(r[key], r["t_"]) for r in rows
               if r[key][0] is not None and r["t_"][0] is not None]
        if not pts:
            continue
        ax.plot([p[0][0] for p in pts], [p[0][1] for p in pts], col,
                label=lab, linewidth=1)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("trajectory")

    for key, col, lab in (("e", "r", "estimate"), ("k", "m", "ekf"),
                          ("r", "orange", "raw odom")):
        tt, ee = [], []
        for r in rows:
            if r[key][0] is None or r["t_"][0] is None:
                continue
            tt.append(r["t"])
            ee.append(math.hypot(r[key][0] - r["t_"][0],
                                 r[key][1] - r["t_"][1]))
        if tt:
            ax2.plot(tt, ee, col, label=lab, linewidth=1)
    for t, name in events:
        if name == "kidnap-detected":
            ax2.axvline(t, color="r", ls="--", lw=0.7, alpha=0.6)
        elif name.startswith("closure-err"):
            ax2.axvline(t, color="b", ls="--", lw=0.7, alpha=0.6)
    ax2.legend()
    ax2.set_title("ATE over time (m)  [red=kidnap, blue=closure]")
    ax2.set_xlabel("t (s)")
    fig.savefig("trajectory.png", dpi=130)
    print("wrote trajectory.png")

    cov = load_cov(cov_path)
    if cov:
        fig2, ax3 = plt.subplots(figsize=(7, 4))
        runs = []
        cur = [cov[0]]
        for prev, row in zip(cov, cov[1:]):
            if (row[2], row[3]) != (prev[2], prev[3]):
                runs.append(cur)
                cur = []
            cur.append(row)
        runs.append(cur)
        for run in runs:
            lbl = "L%s/%s" % (run[0][2], run[0][3])
            ax3.plot([r[0] for r in run], [r[1] for r in run],
                     label=lbl, linewidth=1.2)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("t (s)")
        ax3.set_ylabel("known cells (%)")
        ax3.set_title("coverage")
        ax3.legend(fontsize=8)
        fig2.savefig("coverage.png", dpi=130)
        print("wrote coverage.png")


if __name__ == "__main__":
    main()
