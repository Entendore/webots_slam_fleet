"""Episodic training benchmark with per-episode randomization (v4).

Each episode: randomized spawn slots + obstacle layout (unless
--no-randomize); ends at --target % fleet coverage or --time-limit.
Artifacts archived to episodes/, live maps/logs cleared, world reset.
Q-tables persist, so episodes train cumulatively on DIFFERENT layouts.
Handshake: episode_config.json is written BEFORE simulationReset; fleet
controllers read it at startup (2.5 s watch window). Delete
benchmark_state.json (or --fresh) to restart.
"""

import argparse
import glob
import json
import math
import os
import random
import shutil

from controller import Supervisor

SPAWN_SLOTS = [(-4, -4), (-4, 0), (-4, 4), (0, 4),
               (4, 4), (4, 0), (4, -4), (0, -4)]
OBST_DEFS = [("BOX_1", 0.2), ("BOX_2", 0.2), ("BOX_3", 0.2),
             ("BOX_4", 0.2), ("PILLAR_1", 0.25), ("PILLAR_2", 0.25)]
ROBOT_DEFS = {"scout_lidar": "SCOUT_LIDAR",
              "scout_sonar": "SCOUT_SONAR",
              "scout_half": "SCOUT_HALF"}


class Bench:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--episodes", type=int, default=3)
        ap.add_argument("--time-limit", type=float, default=420.0)
        ap.add_argument("--target", type=float, default=90.0)
        ap.add_argument("--seed", type=int, default=7)
        ap.add_argument("--no-randomize", action="store_true")
        ap.add_argument("--fresh", action="store_true")
        args, _ = ap.parse_known_args()
        self.eps_n, self.limit, self.target = (args.episodes,
                                               args.time_limit, args.target)
        self.seed, self.rand = args.seed, not args.no_randomize

        self.here = os.path.dirname(os.path.abspath(__file__))
        self.fleet = os.path.abspath(os.path.join(self.here, os.pardir,
                                                  "slam_fleet_controller"))
        self.ep_dir = os.path.join(self.here, "episodes")
        os.makedirs(self.ep_dir, exist_ok=True)
        self.state_path = os.path.join(self.here, "benchmark_state.json")
        self.cfg_path = os.path.join(self.here, "episode_config.json")
        self.csv_path = os.path.join(self.here, "fleet_log.csv")

        self.state = None
        if not args.fresh and os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    self.state = json.load(f)
            except (OSError, ValueError):
                self.state = None
        if self.state is None:
            if args.fresh:
                self._wipe_all()
            self.state = {"episode": 1, "history": []}
            self._save_state()

        self.ep = self.state["episode"]
        self.done = self.ep > self.eps_n

        cfg = self._read_cfg()
        if cfg is None or cfg.get("episode") != self.ep:
            cfg = self._gen_cfg(self.ep)
            self._write_cfg(cfg)
        self.cfg = cfg

        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w") as f:
                    f.write("episode,reason,end_s,coverage,t50,t80\n")
            except OSError:
                pass

        self.sup = Supervisor()
        self.ts = int(self.sup.getBasicTimeStep())
        self.need_apply = True
        self.last_t = 0.0
        self.next_cov = 0.0
        self.series = []
        self.miles = {50: None, 80: None, 95: None}

        if self.done:
            self._delete_cfg()
            self._summary()
            print("[bench] complete -- delete benchmark_state.json "
                  "or use --fresh to rerun.")
        else:
            print("=== benchmark v4: episode %d/%d | target %.0f%% | "
                  "limit %.0f s | seed %d%s ===" % (
                      self.ep, self.eps_n, self.target, self.limit,
                      self.seed, "" if self.rand else " | fixed layout"))

    def _wipe_all(self):
        for pat in ("map_*.pgm", "log_*.csv", "q_react_*.json",
                    "q_strat_*.json"):
            for p in glob.glob(os.path.join(self.fleet, pat)):
                try:
                    os.remove(p)
                except OSError:
                    pass
        for p in glob.glob(os.path.join(self.ep_dir, "*")):
            try:
                os.remove(p)
            except OSError:
                pass
        for p in (self.csv_path, self.cfg_path):
            try:
                os.remove(p)
            except OSError:
                pass

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=1)
        except OSError as e:
            print("[bench] could not save state:", e)

    def _read_cfg(self):
        try:
            with open(self.cfg_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_cfg(self, cfg):
        try:
            with open(self.cfg_path, "w") as f:
                json.dump(cfg, f)
        except OSError as e:
            print("[bench] could not write config:", e)

    def _delete_cfg(self):
        try:
            os.remove(self.cfg_path)
        except OSError:
            pass

    def _gen_cfg(self, ep):
        if not self.rand:
            return {"episode": ep, "seed": 0, "robots": {}, "obstacles": {}}
        rng = random.Random(self.seed * 1000 + ep)
        slots = rng.sample(SPAWN_SLOTS, 3)
        names = list(ROBOT_DEFS.keys())
        rng.shuffle(names)
        robots = {}
        for name, (sx, sy) in zip(names, slots):
            robots[name] = {"x": sx, "y": sy,
                            "t": round(math.atan2(-sy, -sx), 3)}
        obstacles = {}
        placed = list(slots)
        for d, _z in OBST_DEFS:
            for _try in range(300):
                x = round(rng.uniform(-3.4, 3.4), 2)
                y = round(rng.uniform(-3.4, 3.4), 2)
                if all(math.hypot(x - px, y - py) > 2.0 for px, py in placed):
                    obstacles[d] = [x, y]
                    placed.append((x, y))
                    break
        return {"episode": ep, "seed": self.seed * 1000 + ep,
                "robots": robots, "obstacles": obstacles}

    def apply_config(self):
        for d, z in OBST_DEFS:
            pos = self.cfg.get("obstacles", {}).get(d)
            if not pos:
                continue
            node = self.sup.getFromDef(d)
            if node:
                node.getField("translation").setSFVec3f([pos[0], pos[1], z])
        for name, s in self.cfg.get("robots", {}).items():
            node = self.sup.getFromDef(ROBOT_DEFS.get(name, ""))
            if node:
                node.getField("translation").setSFVec3f([s["x"], s["y"], 0.0])
                node.getField("rotation").setSFRotation([0, 0, 1, s["t"]])
                node.resetPhysics()
        print("[bench] episode %d layout applied" % self.cfg["episode"])

    def coverage(self):
        files = glob.glob(os.path.join(self.fleet, "map_*.pgm"))
        union = None
        for fp in files:
            try:
                with open(fp, "rb") as f:
                    raw = f.read()
                i = raw.index(b"255\n") + 4
                hdr = raw[:i].split()
                w, h = int(hdr[1]), int(hdr[2])
                data = raw[i:]
                if len(data) < w * h:
                    continue
            except (OSError, ValueError):
                continue
            if union is None:
                union = bytearray(w * h)
            for j in range(w * h):
                if data[j] != 205 and union[j] == 0:
                    union[j] = 1
        if union is None:
            return 0.0
        return 100.0 * sum(union) / len(union)

    def _end(self, reason, cov):
        k = self.ep
        sources = (glob.glob(os.path.join(self.fleet, "map_*.pgm"))
                   + glob.glob(os.path.join(self.fleet, "log_*.csv"))
                   + glob.glob(os.path.join(self.fleet, "q_react_*.json"))
                   + glob.glob(os.path.join(self.fleet, "q_strat_*.json"))
                   + [self.cfg_path])
        for src in sources:
            if os.path.exists(src):
                dst = os.path.join(self.ep_dir,
                                   "ep%d_%s" % (k, os.path.basename(src)))
                try:
                    shutil.copyfile(src, dst)
                except OSError:
                    pass
        for pat in ("map_*.pgm", "log_*.csv"):
            for p in glob.glob(os.path.join(self.fleet, pat)):
                try:
                    os.remove(p)
                except OSError:
                    pass
        row = {"episode": k, "reason": reason,
               "end_s": round(self.last_t, 1), "coverage": round(cov, 1),
               "t50": self.miles[50], "t80": self.miles[80]}
        self.state["history"].append(row)
        self.state["episode"] = k + 1
        self._save_state()
        try:
            with open(self.csv_path, "a") as f:
                f.write("%d,%s,%.1f,%.1f,%s,%s\n"
                        % (k, reason, row["end_s"], row["coverage"],
                           row["t50"], row["t80"]))
        except OSError:
            pass
        print("[bench] episode %d done (%s) after %.0f s: %.0f%% coverage"
              % (k, reason, row["end_s"], cov))
        if k >= self.eps_n:
            self.done = True
            self._delete_cfg()
            self._summary()
            print("[bench] complete -- delete benchmark_state.json "
                  "to run it again.")
        else:
            self.cfg = self._gen_cfg(k + 1)
            self._write_cfg(self.cfg)
            print("[bench] resetting world for episode %d "
                  "(Q-tables persist) ..." % (k + 1))
            self.sup.simulationReset()

    def _summary(self):
        print("\n=== fleet learning summary ===")
        print("episode  reason      end_s  coverage  t50    t80")
        for h in self.state["history"]:
            print("%5d    %-10s  %6.1f  %7.1f%%  %5s  %5s"
                  % (h["episode"], h["reason"], h["end_s"], h["coverage"],
                     h["t50"], h["t80"]))

    def run(self):
        while self.sup.step(self.ts) != -1:
            if self.done:
                continue
            t = self.sup.getTime()
            if t < self.last_t - 0.5:
                self.last_t = 0.0
                self.series = []
                self.miles = {50: None, 80: None, 95: None}
                self.next_cov = 0.0
                self.need_apply = True
            self.last_t = t
            if self.need_apply:
                self.need_apply = False
                self.apply_config()
            if t >= self.next_cov:
                self.next_cov = t + 5.0
                cov = self.coverage()
                self.series.append((t, cov))
                for m in self.miles:
                    if self.miles[m] is None and cov >= m:
                        self.miles[m] = t
                        print("[bench] episode %d: reached %d%% at %.0f s"
                              % (self.ep, m, t))
                if cov >= self.target:
                    self._end("target", cov)
                    continue
            if t >= self.limit:
                self._end("time-limit",
                          self.series[-1][1] if self.series else 0.0)


if __name__ == "__main__":
    Bench().run()
