"""Auto-grader v2 for the SLAM classroom.

    python grader.py                          # grade ../slam_teacher
    python grader.py --dir path/to/folder     # grade one folder
    python grader.py --root students/         # grade every subfolder (class)
    python grader.py --min 8 --json out.json  # pass mark + machine-readable

Exit code (single mode) = number of failed exercises;
exit code (class mode)  = number of folders below the pass mark.
"""

import argparse
import csv
import json
import math
import os


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def load_rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


class Data:
    def __init__(self, base):
        self.events = [(fnum(r["t"]), r["event"])
                       for r in load_rows(os.path.join(base, "events.csv"))]
        self.events = [(t, n) for t, n in self.events if t is not None]
        self.poses = []
        for r in load_rows(os.path.join(base, "pose_log.csv")):
            self.poses.append({
                "t": fnum(r["t"]), "e": (fnum(r["ex"]), fnum(r["ey"])),
                "r": (fnum(r["rx"]), fnum(r["ry"])),
                "t_": (fnum(r["tx"]), fnum(r["ty"])),
                "k": (fnum(r["kx"]), fnum(r["ky"]))})
        self.cov = [(fnum(r["t"]), fnum(r["coverage"]), fnum(r["lesson"]))
                    for r in load_rows(os.path.join(base, "cov_log.csv"))]
        # spans from the coverage log's lesson column (robust to the initial
        # lesson never being key-pressed)
        self.spans = {}
        prev, start = None, None
        for t, _c, l in self.cov:
            if t is None or l is None:
                continue
            if l != prev:
                if prev is not None:
                    self.spans.setdefault(prev, [start, t])
                prev, start = l, t
        if prev is not None:
            self.spans.setdefault(prev, [start, 1e9])

    def between(self, t0, t1):
        return [p for p in self.poses if p["t"] is not None
                and t0 <= p["t"] <= t1]

    def pos(self, p, key):
        a = p[key]
        return a if a[0] is not None and a[1] is not None else None

    def pathlen(self, rows, key):
        tot, prev = 0.0, None
        for p in rows:
            q = self.pos(p, key)
            if q is None:
                continue
            if prev is not None:
                tot += math.hypot(q[0] - prev[0], q[1] - prev[1])
            prev = q
        return tot

    def ate(self, p, key="e"):
        t = self.pos(p, "t_")
        q = self.pos(p, key)
        if t is None or q is None:
            return None
        return math.hypot(q[0] - t[0], q[1] - t[1])


def check_e1(d):
    s = d.spans.get(0)
    if not s:
        return False, "no Lesson-0 time logged"
    rows = d.between(*s)
    if len(rows) < 20:
        return False, "L0 too short (%d samples)" % len(rows)
    key = "t_" if d.pos(rows[len(rows) // 2], "t_") else "r"
    path = d.pathlen(rows, key)
    a, b = d.pos(rows[0], key), d.pos(rows[-1], key)
    if a is None or b is None or path < 2.0:
        return False, "path %.1f m (need > 2 m of driving)" % path
    disp = math.hypot(b[0] - a[0], b[1] - a[1])
    return disp < 0.6, "path %.1f m, net return error %.2f m" % (path, disp)


def check_e2(d):
    s = d.spans.get(1)
    if not s or s[1] - s[0] < 100:
        return False, "Lesson 1 not run for >= 100 s"
    rows = d.between(*s)
    errs = [e for e in (d.ate(p, "r") for p in rows) if e is not None]
    if not errs:
        return False, "no ground truth in log"
    return errs[-1] >= 0.3, "raw-odometry ATE after L1: %.2f m" % errs[-1]


def check_e3(d):
    lo = any(n.startswith("l_occ=") for _, n in d.events)
    lf = any(n.startswith("l_free=") for _, n in d.events)
    rs = any(n == "map-reset" for _, n in d.events)
    return (lo and lf and rs), "l_occ:%s l_free:%s reset:%s" % (lo, lf, rs)


def check_e4(d):
    ons = [t for t, n in d.events if n == "sm-True"]
    if not ons:
        return False, "scan matching was never switched on"
    raw = [e for e in (d.ate(p, "r") for p in d.poses) if e is not None]
    est = [e for e in (d.ate(p) for p in d.poses if p["t"] >= ons[0])
           if e is not None]
    if not est or not raw:
        return False, "no ground truth"
    ok = max(raw) > 0.5 and min(est) < 0.35
    return ok, "max raw ATE %.2f m, best est ATE %.2f m" % (max(raw), min(est))


def check_e5(d):
    mcl_on = None
    for t, n in d.events:
        if n == "mcl-True":
            mcl_on = t
        elif n == "mcl-False":
            mcl_on = None
        elif n == "kidnap-detected" and mcl_on is not None and t >= mcl_on:
            pre = [e for e in (d.ate(p) for p in d.poses
                               if p["t"] is not None and t - 5 <= p["t"] < t)
                   if e is not None]
            post = [e for e in (d.ate(p) for p in d.poses
                                if p["t"] is not None and t <= p["t"] <= t + 25)
                    if e is not None]
            if pre and post and max(pre) > 0.5 and min(post) < 0.35:
                return True, ("kidnap at %.0f s: ATE %.2f -> %.2f m in 25 s"
                              % (t, max(pre), min(post)))
    return False, "no kidnap with MCL active that visibly recovered"


def check_e6(d):
    cl = [n for _, n in d.events
          if n.startswith("closure-err=") or n == "loop-detected"]
    if not cl:
        return False, "no accepted loop closure"
    return True, "%d closure event(s)" % len(cl)


def check_e7(d):
    profs = set(n[6:] for _, n in d.events if n.startswith("noise-"))
    return len(profs) >= 3, "%d profiles tried: %s" % (
        len(profs), ", ".join(sorted(profs)) or "none")


def check_e8(d):
    covs = [c for t, c, l in d.cov if l == 8 and c is not None]
    if not covs:
        return False, "Lesson 8 coverage never logged"
    return max(covs) >= 85, "max coverage in L8: %.0f%%" % max(covs)


def check_e9(d):
    ks = [e for e in (d.ate(p, "k") for p in d.poses) if e is not None]
    if not ks:
        return False, "EKF never ran (no kx/ky in log)"
    return ks[-1] < 0.35, "final EKF ATE: %.3f m" % ks[-1]


def check_e10(d):
    loads = [t for t, n in d.events if n == "amcl-load"]
    if not loads:
        return False, "AMCL mode never used (press L after mapping)"
    for kd in (t for t, n in d.events
               if n == "kidnap-detected" and t > loads[0]):
        post = [e for e in (d.ate(p) for p in d.poses
                            if p["t"] is not None and kd <= p["t"] <= kd + 30)
                if e is not None]
        if post and min(post) < 0.5:
            return True, "AMCL relocalized after kidnap to %.2f m" % min(post)
    return False, "AMCL loaded but no kidnap recovery observed"


def check_e11(d):
    dones = [n for _, n in d.events if n.startswith("pg-done")]
    if not dones:
        return False, "no Gauss-Newton solve (run L6 with the graph back-end)"
    best = None
    for n in dones:
        try:
            parts = dict(p.split("=") for p in n.split()[1:])
            ratio = float(parts["J"]) / max(0.01, float(parts["J0"]))
            best = ratio if best is None else min(best, ratio)
        except (ValueError, IndexError):
            continue
    if best is None:
        return False, "pg-done events unparsable"
    return best <= 0.7, "best cost ratio J/J0 = %.2f (need <= 0.7)" % best


def check_e12(d):
    seen = set(n for _, n in d.events if n.startswith("cam-seen-"))
    return len(seen) >= 3, "%d/4 camera landmarks observed (need 3)" % len(seen)


EXERCISES = [
    ("E1", "L0 loop + return to start", check_e1),
    ("E2", "L1 drift observed (>=0.3 m)", check_e2),
    ("E3", "L2 retuned L_OCC/L_FREE + reset", check_e3),
    ("E4", "L3 scan matching beats odometry", check_e4),
    ("E5", "L4 MCL kidnap recovery", check_e5),
    ("E6", "L6 loop closure fired", check_e6),
    ("E7", "L7 >=3 sensor profiles tried", check_e7),
    ("E8", "L8 frontier coverage >=85%", check_e8),
    ("E9", "L9 EKF converged", check_e9),
    ("E10", "AMCL kidnap recovery", check_e10),
    ("E11", "L6 pose graph converged (J/J0<=0.7)", check_e11),
    ("E12", "L10 >=3 camera landmarks seen", check_e12),
]

HINTS = {
    "E1": "Lesson 0: drive a full square with the arrow keys, end where you started.",
    "E2": "Let Lesson 1 run at least 100 s without toggling S.",
    "E3": "In Lesson 2 press [ ] then - = at least once, and R to reset.",
    "E4": "Lesson 3: run with SM off, then press S.",
    "E5": "Lesson 4: press M, wait, K to kidnap, wait 25 s.",
    "E6": "Lesson 6 needs a full lap (~2 min) with closure on (C).",
    "E7": "Lesson 7: press N at least three times.",
    "E8": "Lesson 8 (F) must reach 85% coverage.",
    "E9": "Lesson 9 (E): drive between beacons so all four get observed.",
    "E10": "Map first (any mapping lesson), then L, wait, K.",
    "E11": "Lesson 6 with the GN back-end on (Y); a full lap gives the solver work.",
    "E12": "Lesson 10 (X): drive so the camera sees 3 of the 4 colored pillars.",
}


def grade_dir(base):
    d = Data(base)
    if not d.poses and not d.events:
        return None, []
    runtime = d.poses[-1]["t"] if d.poses else 0
    results = []
    failed = 0
    for eid, title, fn in EXERCISES:
        try:
            ok, detail = fn(d)
        except Exception as e:
            ok, detail = False, "grader error: %s" % e
        failed += 0 if ok else 1
        results.append({"id": eid, "title": title, "ok": ok,
                        "detail": detail})
    return {"runtime": runtime, "failed": failed,
            "score": len(EXERCISES) - failed}, results


def print_results(results):
    for r in results:
        print("%-4s %-4s %-38s %s" % (r["id"], "PASS" if r["ok"] else "FAIL",
                                      r["title"], r["detail"]))
        if not r["ok"]:
            print("     hint: %s" % HINTS[r["id"]])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.abspath(
        os.path.join(here, os.pardir, "slam_teacher")))
    ap.add_argument("--root", help="grade every subdirectory with logs")
    ap.add_argument("--min", type=int, default=8,
                    help="pass mark out of %d (class mode)" % len(EXERCISES))
    ap.add_argument("--json", metavar="OUT")
    args = ap.parse_args()

    all_results = {}
    if args.root:
        print("=== class mode: %s (pass mark %d/%d) ==="
              % (args.root, args.min, len(EXERCISES)))
        below = 0
        for name in sorted(os.listdir(args.root)):
            folder = os.path.join(args.root, name)
            if not os.path.isdir(folder):
                continue
            summary, results = grade_dir(folder)
            if summary is None:
                continue
            all_results[name] = results
            flag = "OK " if summary["score"] >= args.min else "LOW"
            below += 0 if summary["score"] >= args.min else 1
            print("%-20s %s  %2d/%d  (%.0f s runtime)"
                  % (name, flag, summary["score"], len(EXERCISES),
                     summary["runtime"]))
        print("\nfolders below pass mark: %d" % below)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(all_results, f, indent=1)
        raise SystemExit(below)

    summary, results = grade_dir(args.dir)
    if summary is None:
        raise SystemExit("no logs found in %s -- run the classroom first"
                         % args.dir)
    print("=== SLAM classroom auto-grader (%s) ===" % args.dir)
    print("runtime: %.0f s\n" % summary["runtime"])
    print_results(results)
    print("\nscore: %d/%d" % (summary["score"], len(EXERCISES)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", args.json)
    raise SystemExit(summary["failed"])


if __name__ == "__main__":
    main()
