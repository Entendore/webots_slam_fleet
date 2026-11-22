"""Collaborative multi-robot SLAM with hierarchical learning exploration (v4).

Layers: 1 reflex (+teammate avoidance) | 2 reactive Q (27x7) | 3 A* frontier
planner with mid-pursuit replanning | 4 strategy Q (12 states, learns WHICH
frontier heuristic). Fleet channel shares scans, goals and both Q-tables;
everything persists to disk. Spawn/layout randomization via the supervisor's
episode_config.json. controllerArgs: --x --y --t --r --b [--greedy].
"""

import argparse
import glob
import heapq
import json
import math
import os
import random
from collections import deque

from controller import Robot

LIDAR_NAME, DISPLAY_NAME = "lidar", "display"
EMITTER_NAME, RECEIVER_NAME = "emitter", "receiver"
LEFT_MOTOR, RIGHT_MOTOR = "left wheel motor", "right wheel motor"
LEFT_ENCODER, RIGHT_ENCODER = "left wheel sensor", "right wheel sensor"

WHEEL_RADIUS, AXLE_LENGTH, MAX_WHEEL_SPEED = 0.03, 0.30, 6.0
SCAN_DIRECTION = 1.0

GRID_N, MAP_SIZE = 100, 10.0
CELL = MAP_SIZE / GRID_N
L_OCC, L_FREE, L_MIN, L_MAX = 0.9, -0.4, -5.0, 5.0
FREE_TH, OCC_TH = -1.0, 1.5

SCAN_EVERY, DECIDE_EVERY, COMM_EVERY = 4, 12, 16
DRAW_FULL_EVERY, TELEMETRY_EVERY, MAP_SAVE_STEPS = 24, 150, 240
REPLAN_EVERY = 60

N_ACT = 7
ACTIONS = [(5.5, 5.5), (3.0, 6.0), (0.0, 6.0), (-6.0, 6.0),
           (6.0, 3.0), (6.0, 0.0), (6.0, -6.0)]
ALPHA, GAMMA = 0.35, 0.85
EPS0, EPS_MIN, EPS_TAU = 0.5, 0.05, 200.0
R_CELL, R_TIME = 0.06, 0.02
BUCKET_LO, BUCKET_HI = 0.5, 1.3

N_SACT = 4
S_ALPHA, S_GAMMA = 0.4, 0.9
S_EPS0, S_EPS_MIN, S_EPS_TAU = 0.6, 0.1, 120.0
NEAR_TEAM_R = 2.5
PENDING_TTL = 15.0

PLAN_MIN_CLUSTER, CLEARANCE, UNKNOWN_COST = 3, 2, 3.0
GOAL_TOL, PURSUIT_MAX_T, PURSUIT_MAX_M = 0.22, 30.0, 10.0
CLAIM_R, ABANDON_R, ABANDON_COOLDOWN = 1.2, 1.0, 20.0

V_LIN, TURN_K, SPIN_W = 0.105, 1.2, 5.0
FRONT_STOP, FRONT_CLEAR, SONAR_MAX = 0.33, 0.45, 1.9
TEAM_FILTER_R, TEAM_AVOID_R, TEAM_BEAR_MAX = 0.35, 0.55, 1.1

RELOC_COOL = 2.0

SEED_WATCH = 2.5

PX = 3
COL_UNKNOWN, COL_FREE, COL_OCC = 0x9E9E9E, 0xFFFFFF, 0x202020
COL_SELF, COL_TEAM, COL_ODOM = 0xFF2020, 0xE020E0, 0xFF9500
COL_PATH, COL_GOAL = 0x60A0FF, 0x1050FF
CLS_COL = (COL_UNKNOWN, COL_FREE, COL_OCC)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def bresenham(x0, y0, x1, y1):
    cells = [(x0, y0)]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while x != x1 or y != y1:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        cells.append((x, y))
    return cells


def get_dev(robot, name):
    try:
        return robot.getDevice(name)
    except Exception:
        return None


def cell_of(x, y):
    cx = int((x + MAP_SIZE / 2.0) / CELL)
    cy = int((y + MAP_SIZE / 2.0) / CELL)
    return min(GRID_N - 1, max(0, cx)), min(GRID_N - 1, max(0, cy))


class SensorRig:
    """Lidar (any FOV) or distance-sensor ring -> (angle, range) rays."""

    def __init__(self, robot, ts):
        lid = get_dev(robot, LIDAR_NAME)
        if lid is not None:
            lid.enable(ts)
            n, fov = lid.getHorizontalResolution(), lid.getFov()
            self.kind = "lidar %d rays / %.0f deg" % (n, math.degrees(fov))
            self.angles = [SCAN_DIRECTION * (-fov / 2.0 + i * fov / n)
                           for i in range(n)]
            self.rmin, self.rmax = lid.getMinRange(), lid.getMaxRange()
            self.lidar, self.sonars = lid, None
        else:
            self.sonars = []
            k = 0
            while True:
                s = get_dev(robot, "ds%d" % k)
                if s is None:
                    break
                s.enable(ts)
                self.sonars.append(s)
                k += 1
            if not self.sonars:
                raise RuntimeError("no lidar and no ds0..dsN distance sensors")
            self.kind = "%d distance sensors (30-deg ring)" % len(self.sonars)
            self.angles = [math.radians(30.0 * i)
                           for i in range(len(self.sonars))]
            self.rmin, self.rmax = 0.05, SONAR_MAX
            self.lidar = None
        self.n_rays = len(self.angles)
        self.sm_min = max(8, min(30, self.n_rays // 3))

    def scan(self):
        if self.lidar is not None:
            img = self.lidar.getRangeImage() or []
            return list(zip(self.angles, img))
        out = []
        for a, s in zip(self.angles, self.sonars):
            v = s.getValue()
            out.append((a, v if self.rmin < v < self.rmax else math.inf))
        return out

    def points(self, scan):
        return [(r * math.cos(a), r * math.sin(a))
                for a, r in scan if self.rmin < r < self.rmax]


class QTable:
    """Tabular Q(s,a): disk persistence, optimistic (max) fleet fusion."""

    def __init__(self, kind, robot_id, base, n_act,
                 alpha, gamma, eps0, eps_min, eps_tau):
        self.n_act, self.alpha, self.gamma = n_act, alpha, gamma
        self.eps0, self.eps_min, self.eps_tau = eps0, eps_min, eps_tau
        self.path = os.path.join(base, "q_%s_%s.json" % (kind, robot_id))
        self.q = {}
        self.n = 0
        for p in glob.glob(os.path.join(base, "q_%s_*.json" % kind)):
            self._absorb_file(p)

    def _absorb(self, table):
        if not isinstance(table, dict):
            return
        for s, vec in table.items():
            cur = self.q.setdefault(str(s), [0.0] * self.n_act)
            for i in range(min(self.n_act, len(vec))):
                cur[i] = max(cur[i], float(vec[i]))

    def _absorb_file(self, path):
        try:
            with open(path) as f:
                self._absorb(json.load(f))
        except (OSError, ValueError):
            pass

    def merge(self, table):
        self._absorb(table)

    def row(self, s):
        return self.q.setdefault(s, [0.0] * self.n_act)

    def act(self, s, eps):
        if random.random() < eps:
            return random.randrange(self.n_act)
        v = self.row(s)
        best = max(v)
        return random.choice([i for i, x in enumerate(v) if x == best])

    def learn(self, s, a, r, s2):
        v, v2 = self.row(s), self.row(s2)
        v[a] += self.alpha * (r + self.gamma * max(v2) - v[a])
        self.n += 1

    def learn_terminal(self, s, a, r):
        v = self.row(s)
        v[a] += self.alpha * (r - v[a])
        self.n += 1

    def eps(self):
        return self.eps_min + (self.eps0 - self.eps_min) * \
            math.exp(-self.n / self.eps_tau)

    def as_dict(self):
        return {s: list(v) for s, v in self.q.items()}

    def save(self):
        try:
            data = {s: [round(x, 3) for x in v] for s, v in self.q.items()}
            with open(self.path, "w") as f:
                json.dump(data, f)
        except OSError as e:
            print("could not save Q-table:", e)


class SlamBot:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--x", type=float, default=0.0)
        ap.add_argument("--y", type=float, default=0.0)
        ap.add_argument("--t", type=float, default=0.0)
        ap.add_argument("--r", type=float, default=WHEEL_RADIUS)
        ap.add_argument("--b", type=float, default=AXLE_LENGTH)
        ap.add_argument("--greedy", action="store_true")
        args, _ = ap.parse_known_args()

        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        self.id = self.robot.getName() or "robot"
        self.wr, self.b = args.r, args.b
        self.greedy = args.greedy

        self.rig = SensorRig(self.robot, self.ts)

        self.motors = (get_dev(self.robot, LEFT_MOTOR),
                       get_dev(self.robot, RIGHT_MOTOR))
        self.enc = (get_dev(self.robot, LEFT_ENCODER),
                    get_dev(self.robot, RIGHT_ENCODER))
        if None in self.motors or None in self.enc:
            raise RuntimeError("motor/encoder device not found")
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        for e in self.enc:
            e.enable(self.ts)

        self.display = get_dev(self.robot, DISPLAY_NAME)
        self.emitter = get_dev(self.robot, EMITTER_NAME)
        self.receiver = get_dev(self.robot, RECEIVER_NAME)
        if self.receiver is not None:
            self.receiver.enable(self.ts)

        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
        self.base = base

        self.pose = [args.x, args.y, args.t]
        self.raw = list(self.pose)
        self.enc_prev = None
        self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
        self.teammates = {}
        self.team_goals = {}
        self.last_hits = []

        self.react_q = QTable("react", self.id, base, N_ACT,
                              ALPHA, GAMMA, EPS0, EPS_MIN, EPS_TAU)
        self.strat_q = QTable("strat", self.id, base, N_SACT,
                              S_ALPHA, S_GAMMA, S_EPS0, S_EPS_MIN, S_EPS_TAU)
        self.cur_action = 0
        self.prev_state = None
        self.rewards = deque(maxlen=20)
        self.own_new = 0
        self.pending_r = None
        self.pending_t = 0.0
        self.prev_ss = None
        self.prev_sa = None
        self.collisions = 0
        self.coll_cool = 0.0
        self.milestones = {50: None, 80: None}
        self.outcomes = {"reached": 0, "timeout": 0, "stuck": 0,
                         "blocked": 0, "known": 0}

        self.mode = "WANDER"
        self.goal = None
        self.path = None
        self.path_cells = None
        self.wp_i = 0
        self.p_t0 = 0.0
        self.p_m = 0.0
        self.abandoned = []
        self.n_front = 0
        self.last_gain = 0.0
        self.select_period = max(1, int(round(1000.0 / self.ts)))
        self.select_tick = 1

        self.stk_cmd = self.stk_act = 0.0
        self.stk_t = 0.0
        self.stuck = 0

        self.avoid = False
        self.rev_until = 0.0
        self.reloc_cool = 0.0

        self.cfg_episode = None
        cfg = self._read_cfg()
        if cfg:
            self._apply_cfg(cfg)

        self._disp = bytearray([0xFF]) * (GRID_N * GRID_N)

        self.map_path = os.path.join(base, "map_%s.pgm" % self.id)
        self.csv_path = os.path.join(base, "log_%s.csv" % self.id)
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w") as f:
                    f.write("time_s,mode,mapped_pct,n_front,gain_rate,"
                            "eps_react,eps_strat,collisions,outcomes\n")
            except OSError:
                pass

        self.n = 0
        self.t = 0.0

        print("[%s] %s | sm_min=%d | reflex+reactive-Q(%d)+A*+strategy-Q(%d)%s"
              % (self.id, self.rig.kind, self.rig.sm_min,
                 len(self.react_q.q), len(self.strat_q.q),
                 " | GREEDY" if self.greedy else ""))

    # --------------------------------------------------- supervisor config

    def _read_cfg(self):
        p = os.path.join(os.path.dirname(self.base),
                         "benchmark_supervisor", "episode_config.json")
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _apply_cfg(self, cfg):
        s = cfg.get("robots", {}).get(self.id)
        if s is None:
            return
        self.pose = [s["x"], s["y"], s["t"]]
        self.raw = list(self.pose)
        self.enc_prev = None
        self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
        random.seed(cfg.get("seed", 0) * 31 + sum(ord(c) for c in self.id))
        self.cfg_episode = cfg.get("episode")
        print("[%s] spawn reseeded from episode config (ep %s)"
              % (self.id, self.cfg_episode))

    # ------------------------------------------------------------- actuation

    def actuate(self, wl, wr):
        cl = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, wl))
        cr = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, wr))
        self.motors[0].setVelocity(cl)
        self.motors[1].setVelocity(cr)
        self.stk_cmd += 0.5 * (abs(cl) + abs(cr)) * self.wr * self.ts / 1000.0

    def apply_action(self, a):
        wl, wr = ACTIONS[a]
        self.actuate(wl, wr)

    # -------------------------------------------------------------- odometry

    def update_odometry(self):
        l = self.enc[0].getValue()
        r = self.enc[1].getValue()
        if self.enc_prev is None:
            self.enc_prev = (l, r)
            return
        dl = (l - self.enc_prev[0]) * self.wr
        dr = (r - self.enc_prev[1]) * self.wr
        self.enc_prev = (l, r)
        dist = 0.5 * (dl + dr)
        dth = (dr - dl) / self.b
        self.stk_act += abs(dist)
        if self.mode == "GOTO":
            self.p_m += abs(dist)
        for p in (self.pose, self.raw):
            mid = p[2] + 0.5 * dth
            p[0] += dist * math.cos(mid)
            p[1] += dist * math.sin(mid)
            p[2] = wrap(p[2] + dth)

    # -------------------------------------------------------------- SLAM core

    def sector(self, scan, a0, a1):
        best = math.inf
        for a, r in scan:
            if a0 <= a <= a1 and math.isfinite(r) and r < best:
                best = r
        return best

    def _match_score(self, pts, x, y, th):
        c, s = math.cos(th), math.sin(th)
        tot = 0.0
        for lx, ly in pts:
            cx, cy = cell_of(x + lx * c - ly * s, y + lx * s + ly * c)
            tot += max(0.0, self.grid[cy][cx])
        return tot

    def scan_match(self, hits):
        if len(hits) < self.rig.sm_min:
            return
        pts = hits[::2] if len(hits) >= 30 else hits[:]
        for _ in range(3):
            px, py, pth = self.pose
            base = self._match_score(pts, px, py, pth)
            best, corr = base, None
            for dth in (-0.03, 0.0, 0.03):
                for dx in (-0.04, 0.0, 0.04):
                    for dy in (-0.04, 0.0, 0.04):
                        if dth == dx == dy == 0.0:
                            continue
                        sc = self._match_score(pts, px + dx, py + dy,
                                               pth + dth)
                        if sc > best:
                            best, corr = sc, (dth, dx, dy)
            if corr is None or best <= base + 0.5:
                break
            dth, dx, dy = corr
            self.pose[0] += dx
            self.pose[1] += dy
            self.pose[2] = wrap(self.pose[2] + dth)

    def relocalize(self, hits):
        if self.rig.n_rays < 40 or self.t < self.reloc_cool:
            return
        pts = hits[::3]
        if len(pts) < 20:
            return

        def conf(x, y, th):
            c, s = math.cos(th), math.sin(th)
            tot = 0.0
            for lx, ly in pts:
                cx, cy = cell_of(x + lx * c - ly * s, y + lx * s + ly * c)
                tot += min(1.0, max(0.0, self.grid[cy][cx]))
            return tot / len(pts)

        px, py, pth = self.pose
        base = conf(px, py, pth)
        if base >= 0.45:
            return
        best, bc = base, None
        for dth in (-0.07, -0.035, 0.0, 0.035, 0.07):
            for dx in (-0.12, -0.06, 0.0, 0.06, 0.12):
                for dy in (-0.12, -0.06, 0.0, 0.06, 0.12):
                    if dth == dx == dy == 0.0:
                        continue
                    cc = conf(px + dx, py + dy, pth + dth)
                    if cc > best:
                        best, bc = cc, (dth, dx, dy)
        if bc is not None and best >= 0.6 and best - base >= 0.2:
            dth, dx, dy = bc
            self.pose[0] += dx
            self.pose[1] += dy
            self.pose[2] = wrap(self.pose[2] + dth)
        self.reloc_cool = self.t + RELOC_COOL

    def _paint(self, cx, cy, delta, own):
        row = self.grid[cy]
        before = row[cx] >= OCC_TH or row[cx] <= FREE_TH
        v = row[cx] + delta
        row[cx] = L_MIN if v < L_MIN else (L_MAX if v > L_MAX else v)
        if own and not before and (row[cx] >= OCC_TH or row[cx] <= FREE_TH):
            self.own_new += 1

    def integrate(self, points, pose, own=False):
        px, py, pth = pose
        c, s = math.cos(pth), math.sin(pth)
        x0, y0 = cell_of(px, py)
        mates = list(self.teammates.values()) if own else []
        for lx, ly in points:
            rng = math.hypot(lx, ly)
            dyn = False
            if mates and rng > 0.3:
                wx = px + lx * c - ly * s
                wy = py + lx * s + ly * c
                for tx, ty, _ in mates:
                    if math.hypot(wx - tx, wy - ty) < TEAM_FILTER_R:
                        dyn = True
                        break
            if dyn:
                k = (rng - 0.25) / rng
                lx, ly = lx * k, ly * k
                x1, y1 = cell_of(px + lx * c - ly * s, py + lx * s + ly * c)
                for cx, cy in bresenham(x0, y0, x1, y1)[:-1]:
                    self._paint(cx, cy, L_FREE, own)
                continue
            x1, y1 = cell_of(px + lx * c - ly * s, py + lx * s + ly * c)
            for cx, cy in bresenham(x0, y0, x1, y1)[:-1]:
                self._paint(cx, cy, L_FREE, own)
            self._paint(x1, y1, L_OCC, own)

    def known_frac(self):
        k = 0
        for row in self.grid:
            for v in row:
                if v >= OCC_TH or v <= FREE_TH:
                    k += 1
        return k / float(GRID_N * GRID_N)

    # ---------------------------------------------------------- collaboration

    def broadcast(self):
        if self.emitter is None:
            return
        msg = {"i": self.id,
               "p": [round(v, 3) for v in self.pose],
               "s": [(round(x, 2), round(y, 2))
                     for x, y in self.last_hits[::3]][:90],
               "g": ([round(self.goal["x"], 2), round(self.goal["y"], 2)]
                     if self.goal else None)}
        if self.n % (COMM_EVERY * 10) == 0:
            msg["q"] = self.react_q.as_dict()
            msg["qs"] = self.strat_q.as_dict()
        try:
            self.emitter.send(json.dumps(msg))
        except Exception:
            pass

    def drain_comms(self):
        if self.receiver is None:
            return
        while self.receiver.getQueueLength():
            try:
                msg = json.loads(self.receiver.getString())
                if msg.get("i") != self.id:
                    p = msg["p"]
                    self.teammates[msg["i"]] = (p[0], p[1], p[2])
                    if msg.get("g"):
                        self.team_goals[msg["i"]] = (msg["g"][0], msg["g"][1],
                                                     self.t)
                    pts = [(float(a), float(o)) for a, o in msg.get("s", [])]
                    if pts:
                        self.integrate(pts, (p[0], p[1], p[2]), own=False)
                    if "q" in msg:
                        self.react_q.merge(msg["q"])
                    if "qs" in msg:
                        self.strat_q.merge(msg["qs"])
            except (ValueError, KeyError, TypeError):
                pass
            self.receiver.nextPacket()

    # ---------------------------------------------------- frontiers & planning

    def find_frontiers(self):
        g = self.grid
        lab = [[0] * GRID_N for _ in range(GRID_N)]
        for cy in range(GRID_N):
            for cx in range(GRID_N):
                v = g[cy][cx]
                if FREE_TH < v < OCC_TH and (
                        (cx > 0 and g[cy][cx - 1] <= FREE_TH) or
                        (cx < GRID_N - 1 and g[cy][cx + 1] <= FREE_TH) or
                        (cy > 0 and g[cy - 1][cx] <= FREE_TH) or
                        (cy < GRID_N - 1 and g[cy + 1][cx] <= FREE_TH)):
                    lab[cy][cx] = -1
        clusters, nid = [], 0
        for cy in range(GRID_N):
            for cx in range(GRID_N):
                if lab[cy][cx] != -1:
                    continue
                nid += 1
                cells = [(cx, cy)]
                lab[cy][cx] = nid
                dq = deque([(cx, cy)])
                while dq:
                    x, y = dq.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < GRID_N and 0 <= ny < GRID_N
                                    and lab[ny][nx] == -1):
                                lab[ny][nx] = nid
                                cells.append((nx, ny))
                                dq.append((nx, ny))
                if len(cells) >= PLAN_MIN_CLUSTER:
                    mx = sum(c[0] for c in cells) / float(len(cells))
                    my = sum(c[1] for c in cells) / float(len(cells))
                    clusters.append({"cells": cells, "size": len(cells),
                                     "x": (mx + 0.5) * CELL - MAP_SIZE / 2,
                                     "y": (my + 0.5) * CELL - MAP_SIZE / 2})
        return clusters

    def _blocked_mask(self):
        blocked = bytearray(GRID_N * GRID_N)
        for cy in range(GRID_N):
            for cx in range(GRID_N):
                if self.grid[cy][cx] < OCC_TH:
                    continue
                for dy in range(-CLEARANCE, CLEARANCE + 1):
                    ny = cy + dy
                    if 0 <= ny < GRID_N:
                        for dx in range(-CLEARANCE, CLEARANCE + 1):
                            nx = cx + dx
                            if 0 <= nx < GRID_N:
                                blocked[ny * GRID_N + nx] = 1
        return blocked

    def _snap(self, wx, wy, blocked):
        sx, sy = cell_of(wx, wy)
        if not blocked[sy * GRID_N + sx]:
            return (sx, sy)
        seen = {(sx, sy)}
        dq = deque([(sx, sy)])
        while dq:
            cx, cy = dq.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < GRID_N and 0 <= ny < GRID_N
                            and (nx, ny) not in seen):
                        if not blocked[ny * GRID_N + nx]:
                            return (nx, ny)
                        seen.add((nx, ny))
                        dq.append((nx, ny))
        return None

    def _astar(self, start, goal, blocked):
        W = GRID_N
        gx, gy = goal
        sidx, gidx = start[1] * W + start[0], goal[1] * W + goal[0]
        if sidx == gidx:
            return [start]

        def h(i):
            dx, dy = abs(i % W - gx), abs(i // W - gy)
            return (dx + dy) - 0.5858 * min(dx, dy)

        def cost(i):
            v = self.grid[i // W][i % W]
            return UNKNOWN_COST if FREE_TH < v < OCC_TH else 1.0

        gs = {sidx: 0.0}
        parent = {}
        openq = [(h(sidx), 0.0, sidx)]
        closed = set()
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == gidx:
                idxs = [cur]
                while idxs[-1] != sidx:
                    idxs.append(parent[idxs[-1]])
                idxs.reverse()
                return [(i % W, i // W) for i in idxs]
            cx, cy = cur % W, cur // W
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < W and 0 <= ny < W):
                        continue
                    ni = ny * W + nx
                    if blocked[ni]:
                        continue
                    ng = gc + (1.4142 if dx and dy else 1.0) * cost(ni)
                    if ng < gs.get(ni, 1e18):
                        gs[ni] = ng
                        parent[ni] = cur
                        heapq.heappush(openq, (ng + h(ni), ng, ni))
        return None

    def _los(self, a, b, blocked):
        for cx, cy in bresenham(a[0], a[1], b[0], b[1]):
            if blocked[cy * GRID_N + cx]:
                return False
        return True

    def _smooth(self, cells, blocked):
        out = [cells[0]]
        i = 0
        last = len(cells) - 1
        while i < last:
            j = last
            while j > i + 1 and not self._los(cells[i], cells[j], blocked):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def plan_to(self, wx, wy):
        blocked = self._blocked_mask()
        start = cell_of(self.pose[0], self.pose[1])
        goal = self._snap(wx, wy, blocked)
        if goal is None:
            return None
        cells = self._astar(start, goal, blocked)
        if cells is None:
            return None
        cells = self._smooth(cells, blocked)
        path = [((cx + 0.5) * CELL - MAP_SIZE / 2,
                 (cy + 0.5) * CELL - MAP_SIZE / 2) for cx, cy in cells]
        return cells, path

    def revalidate(self):
        if self.goal is None:
            return
        gx, gy = cell_of(self.goal["x"], self.goal["y"])
        blocked = self._blocked_mask()
        if blocked[gy * GRID_N + gx]:
            self.end_pursuit("blocked")
            return
        if self._los(cell_of(self.pose[0], self.pose[1]), (gx, gy), blocked):
            return
        plan = self.plan_to(self.goal["x"], self.goal["y"])
        if plan is None:
            self.end_pursuit("blocked")
            return
        self.path_cells, self.path = plan
        self.wp_i = 0

    # ---------------------------------------------------- strategy Q (layer 4)

    def _rank(self, clusters, action):
        if action == 0:
            return sorted(clusters, key=lambda c: c["dist"])
        if action == 1:
            return sorted(clusters, key=lambda c: -c["size"])
        if action == 2:
            return sorted(clusters, key=lambda c: -c["size"] / (c["dist"] + 0.5))
        order = list(clusters)
        random.shuffle(order)
        return order

    def _reserved(self, c):
        for x, y, t0 in self.abandoned:
            if (self.t - t0 < ABANDON_COOLDOWN
                    and math.hypot(x - c["x"], y - c["y"]) < ABANDON_R):
                return True
        for x, y, t0 in self.team_goals.values():
            if (self.t - t0 < 6.0
                    and math.hypot(x - c["x"], y - c["y"]) < CLAIM_R):
                return True
        return False

    def start_pursuit(self, clusters):
        px, py, pth = self.pose
        for c in clusters:
            c["dist"] = math.hypot(c["x"] - px, c["y"] - py)
            c["bearing"] = wrap(math.atan2(c["y"] - py, c["x"] - px) - pth)
        n = len(clusters)
        nb = "1" if n == 1 else ("2" if n <= 3 else "3")
        behind = sum(1 for c in clusters if abs(c["bearing"]) > 2.0)
        near = 0
        for tx, ty, _ in self.teammates.values():
            if math.hypot(tx - px, ty - py) < NEAR_TEAM_R:
                near = 1
                break
        s = nb + ("1" if behind / float(n) > 0.4 else "0") + str(near)
        if self.pending_r is not None and self.prev_ss is not None:
            self.strat_q.learn(self.prev_ss, self.prev_sa, self.pending_r, s)
            self.pending_r = None
        self.prev_ss = s
        eps = 0.0 if self.greedy else self.strat_q.eps()
        a = self.strat_q.act(s, eps)
        self.prev_sa = a
        order = self._rank(clusters, a)
        for c in order[:4]:
            if self._reserved(c):
                continue
            plan = self.plan_to(c["x"], c["y"])
            if plan:
                self._begin(c, plan)
                return True
            self.abandoned.append((c["x"], c["y"], self.t))
        for c in order[:2]:
            plan = self.plan_to(c["x"], c["y"])
            if plan:
                self._begin(c, plan)
                return True
        return False

    def _begin(self, c, plan):
        self.path_cells, self.path = plan
        self.wp_i = 0
        self.goal = c
        self.p_t0 = self.t
        self.p_m = 0.0
        self.own_new = 0
        self.mode = "GOTO"
        self.draw_map_full()

    def end_pursuit(self, outcome):
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        m = max(0.2, self.p_m)
        rate = self.own_new / m
        r = 0.1 * rate
        if outcome == "timeout":
            r -= 1.0
        elif outcome == "stuck":
            r -= 2.0
        elif outcome == "blocked":
            r -= 0.8
        elif outcome == "known":
            r -= 0.5
        self.pending_r = max(-3.0, min(6.0, r))
        self.pending_t = self.t
        self.last_gain = rate
        if outcome in ("timeout", "stuck", "blocked") and self.goal:
            self.abandoned.append((self.goal["x"], self.goal["y"], self.t))
        if len(self.abandoned) > 20:
            self.abandoned.pop(0)
        self.goal = None
        self.path = None
        self.own_new = 0
        self.mode = "WANDER"
        self.select_tick = 0

    def pursue(self):
        g = self.goal
        if g is None or self.path is None:
            self.mode = "WANDER"
            return
        if self.t - self.p_t0 > PURSUIT_MAX_T or self.p_m > PURSUIT_MAX_M:
            self.end_pursuit("timeout")
            return
        gcx, gcy = cell_of(g["x"], g["y"])
        v = self.grid[gcy][gcx]
        if v <= FREE_TH or v >= OCC_TH:
            self.end_pursuit("known")
            return
        if math.hypot(g["x"] - self.pose[0], g["y"] - self.pose[1]) < GOAL_TOL:
            self.end_pursuit("reached")
            return
        self._follow()

    def _follow(self):
        px, py, pth = self.pose
        while self.wp_i < len(self.path) - 1:
            wx, wy = self.path[self.wp_i]
            if math.hypot(wx - px, wy - py) < 0.18:
                self.wp_i += 1
            else:
                break
        tx, ty = self.path[self.wp_i]
        e = wrap(math.atan2(ty - py, tx - px) - pth)
        if abs(e) > 1.0:
            s = SPIN_W if e > 0 else -SPIN_W
            self.actuate(-s, s)
        else:
            base = V_LIN / self.wr
            self.actuate(base * (1.0 - TURN_K * e), base * (1.0 + TURN_K * e))

    # ------------------------------------------------- reactive Q (layer 2)

    def state_of(self, front, left, right):
        def bkt(d):
            return 0 if d < BUCKET_LO else (1 if d < BUCKET_HI else 2)
        return "%d%d%d" % (bkt(front), bkt(left), bkt(right))

    def decide(self, front, left, right):
        s = self.state_of(front, left, right)
        if self.prev_state is not None:
            r = R_CELL * self.own_new - R_TIME
            self.react_q.learn(self.prev_state, self.cur_action, r, s)
            self.rewards.append(r)
        self.own_new = 0
        self.prev_state = s
        eps = 0.0 if self.greedy else self.react_q.eps()
        self.cur_action = self.react_q.act(s, eps)

    # --------------------------------------------------------- reflex (layer 1)

    def _teammate_threat(self):
        px, py, pth = self.pose
        best = None
        for tx, ty, _ in self.teammates.values():
            d = math.hypot(tx - px, ty - py)
            if d < TEAM_AVOID_R + 0.15:
                bear = wrap(math.atan2(ty - py, tx - px) - pth)
                if abs(bear) < TEAM_BEAR_MAX and (best is None or d < best[0]):
                    best = (d, bear)
        return best

    def reflex(self, front, left, right):
        mt = self._teammate_threat()
        if mt is not None:
            d, bear = mt
            if d < 0.28:
                if self.t > self.coll_cool:
                    self.collisions += 1
                    self.coll_cool = self.t + 1.0
                self.rev_until = self.t + 0.5
                self.actuate(-4.5, -4.5)
                return True
            if d < TEAM_AVOID_R:
                if bear > 0:
                    self.actuate(6.0, -6.0)
                else:
                    self.actuate(-6.0, 6.0)
                return True
        if self.rev_until > self.t:
            self.actuate(-4.5, -4.5)
            return True
        if front < 0.15:
            if self.t > self.coll_cool:
                self.collisions += 1
                self.coll_cool = self.t + 1.0
            self.rev_until = self.t + 0.6
            self.actuate(-4.5, -4.5)
            return True
        if front < FRONT_STOP:
            self.avoid = True
        if self.avoid:
            if front > FRONT_CLEAR:
                self.avoid = False
            else:
                w = 6.0 if left >= right else -6.0
                self.actuate(-w, w)
                return True
        return False

    # ------------------------------------------------------------- behaviour

    def try_explore(self):
        clusters = self.find_frontiers()
        self.n_front = len(clusters)
        if clusters and self.start_pursuit(clusters):
            return
        if self.known_frac() >= 0.995:
            if self.pending_r is not None and self.prev_ss is not None:
                self.strat_q.learn_terminal(self.prev_ss, self.prev_sa,
                                            self.pending_r)
                self.pending_r = None
            self.mode = "IDLE"
        else:
            self.mode = "WANDER"

    # ---------------------------------------------------------------- display

    def _cell_color(self, v):
        if v >= OCC_TH:
            return COL_OCC
        if v <= FREE_TH:
            return COL_FREE
        return COL_UNKNOWN

    def _restore(self, cx, cy, rad=3):
        for yy in range(max(0, cy - rad), min(GRID_N, cy + rad + 1)):
            row = self.grid[yy]
            y = (GRID_N - 1 - yy) * PX
            base = yy * GRID_N
            for xx in range(max(0, cx - rad), min(GRID_N, cx + rad + 1)):
                v = row[xx]
                cls = 2 if v >= OCC_TH else (1 if v <= FREE_TH else 0)
                self._disp[base + xx] = cls
                self.display.setColor(CLS_COL[cls])
                self.display.fillRectangle(xx * PX, y, PX, PX)

    def _marker(self, pose, color):
        cx, cy = cell_of(pose[0], pose[1])
        sx = (cx + 0.5) * PX
        sy = (GRID_N - 0.5 - cy) * PX
        self.display.setColor(color)
        self.display.fillOval(int(sx) - 3, int(sy) - 3, 6, 6)
        self.display.drawLine(int(sx), int(sy),
                              int(sx + 9 * math.cos(pose[2])),
                              int(sy - 9 * math.sin(pose[2])))

    def draw_local(self):
        if self.display is None:
            return
        spots = [cell_of(self.pose[0], self.pose[1]),
                 cell_of(self.raw[0], self.raw[1])]
        spots += [cell_of(p[0], p[1]) for p in self.teammates.values()]
        for cx, cy in set(spots):
            self._restore(cx, cy)
        for p in self.teammates.values():
            self._marker(p, COL_TEAM)
        self._marker(self.raw, COL_ODOM)
        self._marker(self.pose, COL_SELF)

    def draw_map_full(self):
        if self.display is None:
            return
        d = self.display
        for cy in range(GRID_N):
            row = self.grid[cy]
            y = (GRID_N - 1 - cy) * PX
            base = cy * GRID_N
            for cx in range(GRID_N):
                v = row[cx]
                cls = 2 if v >= OCC_TH else (1 if v <= FREE_TH else 0)
                if self._disp[base + cx] == cls:
                    continue
                self._disp[base + cx] = cls
                d.setColor(CLS_COL[cls])
                d.fillRectangle(cx * PX, y, PX, PX)
        if self.path_cells and len(self.path_cells) > 1:
            d.setColor(COL_PATH)
            pts = [((cx + 0.5) * PX, (GRID_N - 1 - cy + 0.5) * PX)
                   for cx, cy in self.path_cells]
            for i in range(1, len(pts)):
                d.drawLine(int(pts[i - 1][0]), int(pts[i - 1][1]),
                           int(pts[i][0]), int(pts[i][1]))
        if self.goal:
            gx, gy = cell_of(self.goal["x"], self.goal["y"])
            d.setColor(COL_GOAL)
            d.fillRectangle(gx * PX, (GRID_N - 1 - gy) * PX, PX, PX)
        self.draw_local()

    # -------------------------------------------------------------- telemetry

    def telemetry(self):
        frac = 100.0 * self.known_frac()
        for m in self.milestones:
            if self.milestones[m] is None and frac >= m:
                self.milestones[m] = self.t
        eps_r = 0.0 if self.greedy else self.react_q.eps()
        eps_s = 0.0 if self.greedy else self.strat_q.eps()
        o = self.outcomes
        ostr = "R%d/T%d/S%d/B%d/K%d" % (o["reached"], o["timeout"],
                                        o["stuck"], o["blocked"], o["known"])
        print("[%s %6.1f s] %-6s cov=%3.0f%% front=%2d gain=%5.1f c/m "
              "eps=%.2f/%.2f coll=%d %s" % (
                  self.id, self.t, self.mode, frac, self.n_front,
                  self.last_gain, eps_r, eps_s, self.collisions, ostr))
        try:
            with open(self.csv_path, "a") as f:
                f.write("%.1f,%s,%.1f,%d,%.2f,%.3f,%.3f,%d,%s\n"
                        % (self.t, self.mode, frac, self.n_front,
                           self.last_gain, eps_r, eps_s, self.collisions,
                           ostr))
        except OSError:
            pass
        self.react_q.save()
        self.strat_q.save()

    def save_map(self):
        try:
            with open(self.map_path, "wb") as f:
                f.write(("P5\n%d %d\n255\n" % (GRID_N, GRID_N)).encode("ascii"))
                data = bytearray(GRID_N * GRID_N)
                i = 0
                for cy in range(GRID_N - 1, -1, -1):
                    row = self.grid[cy]
                    for cx in range(GRID_N):
                        v = row[cx]
                        data[i] = (0 if v >= OCC_TH
                                   else (254 if v <= FREE_TH else 205))
                        i += 1
                f.write(bytes(data))
        except OSError as e:
            print("could not write map:", e)

    # -------------------------------------------------------------------- run

    def run(self):
        r = self.robot
        while r.step(self.ts) != -1:
            self.n += 1
            self.t = self.n * self.ts / 1000.0

            self.update_odometry()
            scan = self.rig.scan()
            front = self.sector(scan, -0.62, 0.62)
            left = self.sector(scan, 0.65, 1.75)
            right = self.sector(scan, -1.75, -0.65)
            self.drain_comms()

            if self.t < SEED_WATCH and self.n % 4 == 0:
                cfg = self._read_cfg()
                if cfg and cfg.get("episode") != self.cfg_episode:
                    self._apply_cfg(cfg)

            if self.n % SCAN_EVERY == 0:
                self.last_hits = self.rig.points(scan)
                self.scan_match(self.last_hits)
                self.relocalize(self.last_hits)
                self.integrate(self.last_hits, self.pose, own=True)

            if not self.reflex(front, left, right):
                if self.mode == "GOTO":
                    if self.n % REPLAN_EVERY == 0:
                        self.revalidate()
                    if self.mode == "GOTO":
                        self.pursue()
                else:
                    self.select_tick -= 1
                    if self.select_tick <= 0:
                        self.select_tick = self.select_period
                        self.try_explore()
                    if self.mode == "WANDER":
                        if self.n % DECIDE_EVERY == 0:
                            self.decide(front, left, right)
                        self.apply_action(self.cur_action)
                    else:
                        self.actuate(0.0, 0.0)

            if self.t - self.stk_t >= 1.0:
                if self.stk_cmd > 0.04 and self.stk_act < 0.3 * self.stk_cmd:
                    self.stuck += 1
                    if self.mode == "GOTO" and self.stuck >= 2:
                        self.end_pursuit("stuck")
                        self.stuck = 0
                else:
                    self.stuck = 0
                self.stk_cmd = 0.0
                self.stk_act = 0.0
                self.stk_t = self.t

            if (self.pending_r is not None and self.prev_ss is not None
                    and self.t - self.pending_t > PENDING_TTL):
                self.strat_q.learn_terminal(self.prev_ss, self.prev_sa,
                                            self.pending_r)
                self.pending_r = None

            self.draw_local()
            if self.n % DRAW_FULL_EVERY == 0:
                self.draw_map_full()
            if self.n % COMM_EVERY == 0:
                self.broadcast()
            if self.n % TELEMETRY_EVERY == 0:
                self.telemetry()
            if self.n % MAP_SAVE_STEPS == 0:
                self.save_map()

        self.telemetry()
        self.save_map()
        print("[%s] done. 50%% at %s s, 80%% at %s s | outcomes=%s" % (
            self.id, self.milestones[50], self.milestones[80], self.outcomes))


if __name__ == "__main__":
    SlamBot().run()
