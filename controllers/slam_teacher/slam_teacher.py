"""SLAM classroom controller v4 -- eleven lessons + live demo tools.

Lessons (keys 0-9, X): L0 manual | L1 dead reckoning | L2 mapping |
L3 scan match | L4 particle filter + kidnap | L5 everything |
L6 LOOP CLOSURE with an animated Gauss-Newton pose-graph back-end |
L7 sensor lab (noise/dropout/bias/ghosts + odom damage + map decay) |
L8 frontier exploration | L9 EKF on virtual beacons (labeled cheat) |
L10 (key X) CAMERA landmarks -- real blob detection, no cheat.

Keys: 0-9,X lessons | SPACE pause | S sm | M mcl | E ekf | U ekf-loc |
C closure | Y graph-vs-linear | D decay | N noise | O odom-dmg | F frontier |
L amcl-load | B beams | G truth | P particles | R reset | K kidnap | [ ] - =

Honesty labels: GPS = intentional ground-truth cheat. L9 beacons = virtual
cheat (estimator math is the lesson). L10 = real camera + real pillars.

Outputs: pose_log.csv, cov_log.csv, events.csv, map_classroom.pgm,
ate_final.txt (read by ablation_supervisor and grader.py).
"""

import argparse
import heapq
import json
import math
import os
import random
from collections import deque

from controller import Robot, Keyboard

LIDAR_NAME, DISPLAY_NAME, CAMERA_NAME = "lidar", "display", "camera"
LEFT_MOTOR, RIGHT_MOTOR = "left wheel motor", "right wheel motor"
LEFT_ENCODER, RIGHT_ENCODER = "left wheel sensor", "right wheel sensor"
GPS_NAME = "gps"

WHEEL_RADIUS, AXLE_LENGTH, MAX_WHEEL_SPEED = 0.03, 0.30, 6.0
SCAN_DIRECTION = 1.0

# keep in sync with CLASS_ROBOT's pose in slam_classroom.wbt
SPAWN = (-3.0, -4.0, 0.0)

GRID_N, MAP_SIZE = 100, 10.0
CELL = MAP_SIZE / GRID_N
L_OCC, L_FREE, L_MIN, L_MAX = 0.9, -0.4, -5.0, 5.0
FREE_TH, OCC_TH = -1.0, 1.5

SCAN_EVERY, DRAW_FULL_EVERY, LOG_EVERY, HUD_EVERY = 4, 24, 25, 6

SM_DTH, SM_DT = 0.03, 0.04

MCL_N = 200
MCL_DIST_J, MCL_DIST_A = 0.05, 0.003
MCL_TH_J, MCL_TH_A = 0.15, 0.004
MCL_JITTER_X, MCL_JITTER_T = 0.005, 0.01
MCL_ESS_FRAC = 0.5
KIDNAP_SPREAD_J = 0.35

LC_MIN_SEP, LC_RADIUS, LC_COOLDOWN, LC_MAX_ERR = 12, 0.55, 6.0, 0.9
PG_NODE_GAP, PG_MAX_NODES, PG_WINDOW_PAD = 0.8, 140, 2
PG_MAX_ITER, PG_EPS, PG_LOOP_W = 15, 1e-3, 3.0

NOISE_PROFILES = [
    ("clean", 0.0, 0.0, 0.0, 0.0),
    ("gauss5", 0.05, 0.0, 0.0, 0.0),
    ("gauss15", 0.15, 0.0, 0.0, 0.0),
    ("drop30", 0.0, 0.30, 0.0, 0.0),
    ("bias25", 0.0, 0.0, 0.25, 0.0),
    ("ghosts", 0.0, 0.0, 0.0, 0.08),
]
ODOM_SLIP_D, ODOM_SLIP_T = 0.012, 0.003
DECAY_RATE = 0.05

PLAN_MIN_CLUSTER, CLEARANCE, UNKNOWN_COST = 3, 2, 3.0
GOAL_TOL, FR_TIMEOUT = 0.25, 45.0
ABANDON_R, ABANDON_COOL = 1.0, 20.0

LANDMARKS = [(-3.5, 3.5), (3.5, 3.5), (3.5, -3.5), (-3.5, -3.5)]
BEA_RANGE, BEA_SR, BEA_SB = 3.2, 0.03, 0.02
CAM_WIDTH = 0.30
CAM_COLS = (0xFF3030, 0x30D030, 0x3050FF, 0xFFD020)

V_LIN, FRONT_STOP, TURN_W = 5.0, 0.45, 5.0
HELD_TIMEOUT = 0.65

PX, MAP_X, MAP_Y, LEG_X = 2, 0, 44, 204
HUD_COL = 0x101010
COL_TEXT, COL_DIM = 0xE0E0E0, 0x808080
COL_TRAIL, COL_TRAIL_RAW, COL_TRUTH = 0xFF3030, 0xFF9500, 0x20D060
COL_BEAM, COL_FREEPATH, COL_HIT = 0xFFD020, 0x20A0FF, 0xFF2020
COL_PART, COL_NODE, COL_EDGE = 0x7A00FF, 0x4060FF, 0x3050FF
COL_PATH, COL_FRONT, COL_EKFC = 0x60A0FF, 0xFF8000, 0xB060FF
COL_EDGE_OK, COL_EDGE_MID, COL_EDGE_BAD = 0x30C030, 0xE0C020, 0xFF4040


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


def cell_of(x, y):
    cx = int((x + MAP_SIZE / 2.0) / CELL)
    cy = int((y + MAP_SIZE / 2.0) / CELL)
    return min(GRID_N - 1, max(0, cx)), min(GRID_N - 1, max(0, cy))


def rel_pose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def comp_rel(p, q):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (p[0] + c * q[0] - s * q[1],
            p[1] + s * q[0] + c * q[1], wrap(p[2] + q[2]))


def inv_rel(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-c * p[0] - s * p[1], s * p[0] - c * p[1], wrap(-p[2]))


def mat_zero(r, c):
    return [[0.0] * c for _ in range(r)]


def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = mat_zero(n, p)
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += a * Bk[j]
    return C


def mat_T(A):
    return [list(r) for r in zip(*A)]


def mat_add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mat_sub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mat_eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_inv2(A):
    d = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(d) < 1e-12:
        d = 1e-12
    return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]


def lin_solve(A, rhs):
    """Gaussian elimination with partial pivoting (H is small)."""
    n = len(A)
    M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f:
                Mr, Mc = M[r], M[col]
                for c in range(col, n + 1):
                    Mr[c] -= f * Mc[c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r] if abs(M[r][r]) > 1e-12 else 0.0
    return x


LESSON_TEXT = {
    0: "L0 MANUAL: arrows steer. Console shows odometry vs green truth.",
    1: "L1 DEAD RECKONING: red=estimate green=truth orange=raw odom. "
       "Watch them split -- that drift is why SLAM exists.",
    2: "L2 MAPPING: B shows beams (yellow), free cells (blue), hits (red). "
       "Retune with [ ] and - = ; R to reset the map.",
    3: "L3 SCAN MATCHING: press S to switch it on and watch red snap to green.",
    4: "L4 PARTICLE FILTER: press M. Purple=hypotheses, ESS in HUD. "
       "Press K to KIDNAP and watch recovery.",
    5: "L5 EVERYTHING: S+M together; HUD compares both ATEs every frame.",
    6: "L6 LOOP CLOSURE: drive a lap. On revisit a loop CONSTRAINT is added "
       "and the pose graph is optimized LIVE -- watch edges turn green. "
       "Y toggles Gauss-Newton vs the old linear hack.",
    7: "L7 SENSOR LAB: N cycles profiles, O damages ODOMETRY, D enables map "
       "DECAY. Toggle S: which layer survives?",
    8: "L8 EXPLORATION: the robot picks frontier goals and plans. F toggles.",
    9: "L9 LANDMARKS: EKF-SLAM on 4 VIRTUAL beacons (labeled cheat). "
       "Ellipses = 2-sigma. U toggles SLAM vs LOC (pinned beacons).",
    10: "L10 CAMERA LANDMARKS (no cheat): real blob detection -- bearing from "
        "the centroid, range from apparent size. EKF-SLAM on the four colored "
        "pillars. U toggles SLAM/LOC. K to kidnap.",
}


class Teacher:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--lesson", type=int, default=1)
        ap.add_argument("--no-truth", action="store_true")
        ap.add_argument("--particles", type=int, default=MCL_N)
        ap.add_argument("--seed", type=int, default=1)
        args, _ = ap.parse_known_args()
        random.seed(args.seed)

        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        self.kb = self.robot.getKeyboard()
        self.kb.enable(self.ts)

        self.lidar = self.robot.getDevice(LIDAR_NAME)
        self.lidar.enable(self.ts * SCAN_EVERY)
        n, fov = self.lidar.getHorizontalResolution(), self.lidar.getFov()
        self.ray_angles = [SCAN_DIRECTION * (-fov / 2 + i * fov / n)
                           for i in range(n)]
        self.rmin, self.rmax = self.lidar.getMinRange(), self.lidar.getMaxRange()

        try:
            self.cam = self.robot.getDevice(CAMERA_NAME)
            self.cam.enable(self.ts)
        except Exception:
            self.cam = None

        self.display = self.robot.getDevice(DISPLAY_NAME)
        self.motors = (self.robot.getDevice(LEFT_MOTOR),
                       self.robot.getDevice(RIGHT_MOTOR))
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.enc = (self.robot.getDevice(LEFT_ENCODER),
                    self.robot.getDevice(RIGHT_ENCODER))
        for e in self.enc:
            e.enable(self.ts)
        self.gps = self.robot.getDevice(GPS_NAME)
        self.gps.enable(self.ts)
        self.truth_on = not args.no_truth

        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
        self.base = base
        self.sup_dir = os.path.join(os.path.dirname(base),
                                    "ablation_supervisor")

        self.odom = list(SPAWN)
        self.sm = list(SPAWN)
        self.mcl = list(SPAWN)
        self.truth = list(SPAWN)
        self.true_prev = list(SPAWN)
        self.tv = [SPAWN[0], SPAWN[1]]
        self.th = SPAWN[2]
        self.enc_prev = None
        self.kidnap_flash = 0.0
        self.n_kidnaps = 0

        self.L_OCC, self.L_FREE = L_OCC, L_FREE
        self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
        self._disp = bytearray([0xFF]) * (GRID_N * GRID_N)
        self._map_dirty = True
        self.mapping_on = True
        self.decay_on = False

        self.nodes = []
        self.pg_edges = []
        self.pg_on = False
        self.pg_solving = False
        self.pg_map = []
        self.pg_nodes = []
        self.pg_local = []
        self.pg_iter = 0
        self.pg_J0 = self.pg_J = self.pg_prevJ = None
        self.pg_live_rel = None
        self.pg_flash = 0.0
        self.closure_on = False
        self.lc_cool = 0.0
        self.lc_edge = None
        self.lc_err = None

        self.noise_idx = 0
        self.odom_damage = False

        self.n_particles = max(20, args.particles)
        self.particles = []
        self.ess = float(self.n_particles)
        self.resample_flash = False
        self.seed_particles(self.odom, 0.05)

        self.ekf_on = False
        self.ekf_known = False
        self.cam_on = False
        self.ekf_x = None
        self.ekf_P = None
        self.ekf_seen = None
        self.ekf_meas = []
        self.cam_meas = []
        self.cam_logged = set()

        self.frontier_on = False
        self.goal = None
        self.path_cells = None
        self.wp_i = 0
        self.g_t0 = 0.0
        self.abandoned = []
        self.n_front = 0
        self.fr_tick = 0
        self.front_cache, self.front_cache_t = [], -1e9

        self.lesson = max(0, min(10, args.lesson))
        self.sm_on = self.mcl_on = False
        self.paused = False
        self.show_beams = False
        self.show_truth = True
        self.show_particles = True
        self.est_trail = deque(maxlen=300)
        self.raw_trail = deque(maxlen=300)
        self.true_trail = deque(maxlen=300)
        self.trail_dirty = True
        self._overlay_cells = []
        self._held_t = {}
        self._held = set()
        self._prev_held = set()
        self.sm_flash = 0.0
        self._snap_info = ""
        self.cov_cache, self.cov_t = 0.0, -1e9
        self.apply_lesson(self.lesson, announce=False)

        self.log_path = os.path.join(base, "pose_log.csv")
        self.cov_path = os.path.join(base, "cov_log.csv")
        self.ev_path = os.path.join(base, "events.csv")
        self.map_path = os.path.join(base, "map_classroom.pgm")
        self.fin_path = os.path.join(base, "ate_final.txt")
        with open(self.log_path, "w") as f:
            f.write("t,ex,ey,eth,rx,ry,rth,tx,ty,tth,kx,ky,kth\n")
        with open(self.cov_path, "w") as f:
            f.write("t,coverage,lesson,mode\n")
        with open(self.ev_path, "w") as f:
            f.write("t,event\n")

        self.n = 0
        self.t = 0.0
        self.cfg_tag = None
        self._apply_teach_config()
        self.log_event("lesson%d" % self.lesson)

        print("=== SLAM classroom v4 -- %s" % LESSON_TEXT[self.lesson])
        print("keys: 0-9,X lessons | SPACE pause | S sm | M mcl | E ekf | "
              "U ekf-loc | C closure | Y graph | D decay | N noise | O dmg | "
              "F frontier | L amcl | B/G/P | R reset | K kidnap | [ ] - =")
        print("grade:  python ../grader/grader.py")

    # ----------------------------------------------------- autopilot config

    def _read_teach_config(self):
        p = os.path.join(self.sup_dir, "teach_config.json")
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _apply_teach_config(self):
        cfg = self._read_teach_config()
        if not cfg:
            return
        if cfg.get("tag") != self.cfg_tag:
            self.cfg_tag = cfg.get("tag")
            with open(self.log_path, "w") as f:
                f.write("t,ex,ey,eth,rx,ry,rth,tx,ty,tth,kx,ky,kth\n")
            with open(self.cov_path, "w") as f:
                f.write("t,coverage,lesson,mode\n")
            with open(self.ev_path, "w") as f:
                f.write("t,event\n")
        if "lesson" in cfg:
            self.apply_lesson(int(cfg["lesson"]), announce=False)
        for key, attr in (("sm", "sm_on"), ("mcl", "mcl_on"),
                          ("closure", "closure_on"), ("ekf", "ekf_on"),
                          ("frontier", "frontier_on"),
                          ("odom_damage", "odom_damage"),
                          ("decay", "decay_on"), ("posegraph", "pg_on")):
            if key in cfg:
                setattr(self, attr, bool(cfg[key]))
        if "noise" in cfg:
            self.noise_idx = max(0, min(len(NOISE_PROFILES) - 1,
                                       int(cfg["noise"])))
        if cfg.get("ekf_known") and self.ekf_on:
            self.ekf_set_known(True)
        print("[cfg] autopilot run '%s' applied" % self.cfg_tag)

    # ------------------------------------------------------------- lessons

    def apply_lesson(self, l, announce=True):
        self.lesson = l
        self.manual = (l == 0)
        self.sm_on = l in (3, 5, 6, 8)
        self.mcl_on = l in (4, 5)
        self.closure_on = (l == 6)
        self.pg_on = (l == 6)
        self.ekf_on = l in (9, 10)
        self.cam_on = (l == 10)
        self.frontier_on = (l == 8)
        self.show_beams = l in (0, 2)
        self.noise_idx = 0
        self.odom_damage = False
        self.goal = None
        self.path_cells = None
        self.mapping_on = True
        self.trail_dirty = True
        self.cam_meas = []
        self.cam_logged = set()
        if l == 6:
            self.nodes = []
            self.pg_edges = []
        if self.ekf_on:
            self.ekf_init()
        if announce:
            self.log_event("lesson%d" % l)
        print(LESSON_TEXT[l])

    def active_pose(self):
        if self.ekf_on:
            return [self.ekf_x[0], self.ekf_x[1], self.ekf_x[2]]
        if self.sm_on:
            return self.sm
        if self.mcl_on:
            return self.mcl
        return self.odom

    def set_active(self, p):
        tgt = (self.ekf_x if self.ekf_on else
               self.sm if self.sm_on else
               self.mcl if self.mcl_on else self.odom)
        tgt[0], tgt[1], tgt[2] = p[0], p[1], wrap(p[2])

    # ------------------------------------------------------------- controls

    def handle_keys(self, edges):
        for k in edges:
            if ord("0") <= k <= ord("9"):
                self.apply_lesson(k - ord("0"))
            elif k in (ord("x"), ord("X")):
                self.apply_lesson(10)
            elif k == 32:
                self.paused = not self.paused
            elif k in (ord("s"), ord("S")):
                self.sm_on = not self.sm_on
                self.sm = list(self.active_pose())
                self.log_event("sm-%s" % self.sm_on)
            elif k in (ord("m"), ord("M")):
                self.mcl_on = not self.mcl_on
                if self.mcl_on:
                    self.seed_particles(self.active_pose(), 0.05)
                self.log_event("mcl-%s" % self.mcl_on)
            elif k in (ord("e"), ord("E")):
                self.ekf_on = not self.ekf_on
                if self.ekf_on:
                    self.ekf_init()
                self.log_event("ekf-%s" % self.ekf_on)
            elif k in (ord("u"), ord("U")):
                self.ekf_set_known(not self.ekf_known)
                self.log_event("ekf-known-%s" % self.ekf_known)
            elif k in (ord("c"), ord("C")):
                self.closure_on = not self.closure_on
                self.log_event("closure-%s" % self.closure_on)
            elif k in (ord("y"), ord("Y")):
                if not self.pg_solving:
                    self.pg_on = not self.pg_on
                    self.log_event("posegraph-%s" % self.pg_on)
                    print("back-end: %s" %
                          ("Gauss-Newton pose graph" if self.pg_on
                           else "linear redistribution (naive)"))
            elif k in (ord("d"), ord("D")):
                self.decay_on = not self.decay_on
                self.log_event("decay-%s" % self.decay_on)
                print("map decay: %s" % self.decay_on)
            elif k in (ord("n"), ord("N")):
                self.noise_idx = (self.noise_idx + 1) % len(NOISE_PROFILES)
                self.log_event("noise-%s" % NOISE_PROFILES[self.noise_idx][0])
                print("sensor profile: %s" % NOISE_PROFILES[self.noise_idx][0])
            elif k in (ord("o"), ord("O")):
                self.odom_damage = not self.odom_damage
                self.log_event("odom-damage-%s" % self.odom_damage)
                print("odometry damage: %s" % self.odom_damage)
            elif k in (ord("f"), ord("F")):
                self.frontier_on = not self.frontier_on
                self.goal = None
                self.log_event("frontier-%s" % self.frontier_on)
            elif k in (ord("b"), ord("B")):
                self.show_beams = not self.show_beams
            elif k in (ord("g"), ord("G")):
                self.show_truth = not self.show_truth
                self.trail_dirty = True
            elif k in (ord("p"), ord("P")):
                self.show_particles = not self.show_particles
            elif k in (ord("r"), ord("R")):
                self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
                self.nodes = []
                self.pg_edges = []
                self.invalidate_view()
                self.mapping_on = True
                self.seed_particles(self.active_pose(), 0.05)
                self.log_event("map-reset")
            elif k in (ord("k"), ord("K")):
                try:
                    with open(os.path.join(os.path.dirname(self.base),
                                           "kidnap_supervisor",
                                           "kidnap.now"), "w") as f:
                        f.write("1")
                except OSError:
                    pass
            elif k in (ord("l"), ord("L")):
                self.amcl_load()
            elif k == ord("["):
                self.L_OCC = max(0.1, self.L_OCC - 0.2)
                self.log_event("l_occ=%.1f" % self.L_OCC)
            elif k == ord("]"):
                self.L_OCC = min(3.0, self.L_OCC + 0.2)
                self.log_event("l_occ=%.1f" % self.L_OCC)
            elif k == ord("-"):
                self.L_FREE = max(-2.5, self.L_FREE + 0.1)
                self.log_event("l_free=%.1f" % self.L_FREE)
            elif k == ord("="):
                self.L_FREE = min(-0.05, self.L_FREE - 0.1)
                self.log_event("l_free=%.1f" % self.L_FREE)

    # -------------------------------------------------------------- motion

    def actuate(self, wl, wr):
        self.motors[0].setVelocity(max(-MAX_WHEEL_SPEED,
                                       min(MAX_WHEEL_SPEED, wl)))
        self.motors[1].setVelocity(max(-MAX_WHEEL_SPEED,
                                       min(MAX_WHEEL_SPEED, wr)))

    def drive(self, front, left, right):
        if self.paused:
            self.actuate(0.0, 0.0)
            return
        if self.manual:
            held = self._held
            v = 0.0
            w = 0.0
            if Keyboard.UP in held:
                v = V_LIN
            if Keyboard.DOWN in held:
                v = -0.7 * V_LIN
            if Keyboard.LEFT in held:
                w = TURN_W
            if Keyboard.RIGHT in held:
                w = -TURN_W
            if w:
                self.actuate(-w if v >= 0 else w, w if v >= 0 else -w)
            else:
                self.actuate(v, v)
            return
        if self.frontier_on and self.goal and self.path_cells:
            self.follow()
            return
        if front < FRONT_STOP:
            w = TURN_W if left >= right else -TURN_W
            self.actuate(-w, w)
        else:
            steer = 0.8 * math.sin(0.25 * self.t)
            self.actuate(V_LIN * (1 - steer), V_LIN * (1 + steer))

    # ------------------------------------------------------------- odometry

    def update_odometry(self):
        l = self.enc[0].getValue()
        r = self.enc[1].getValue()
        if self.enc_prev is None:
            self.enc_prev = (l, r)
            return 0.0, 0.0
        dl = (l - self.enc_prev[0]) * WHEEL_RADIUS
        dr = (r - self.enc_prev[1]) * WHEEL_RADIUS
        self.enc_prev = (l, r)
        if self.odom_damage:
            dl *= 1.0 + random.gauss(0, ODOM_SLIP_D) - 0.015
            dr *= 1.0 + random.gauss(0, ODOM_SLIP_D) + 0.005
        dist = 0.5 * (dl + dr)
        dth = (dr - dl) / AXLE_LENGTH
        if self.odom_damage:
            dth += random.gauss(0, ODOM_SLIP_T)
        for p in (self.odom, self.sm, self.mcl):
            mid = p[2] + 0.5 * dth
            p[0] += dist * math.cos(mid)
            p[1] += dist * math.sin(mid)
            p[2] = wrap(p[2] + dth)
        return dist, dth

    def read_truth(self):
        if not self.truth_on:
            return
        g = self.gps.getValues()
        self.true_prev = list(self.truth)
        self.truth[0], self.truth[1] = g[0], g[1]
        if math.hypot(g[0] - self.tv[0], g[1] - self.tv[1]) > 0.02:
            self.th = wrap(math.atan2(g[1] - self.tv[1], g[0] - self.tv[0]))
            self.tv = [g[0], g[1]]
        self.truth[2] = self.th
        if math.hypot(self.truth[0] - self.true_prev[0],
                      self.truth[1] - self.true_prev[1]) > 0.5:
            self.on_kidnap()

    def on_kidnap(self):
        self.n_kidnaps += 1
        self.kidnap_flash = self.t + 2.0
        self.log_event("kidnap-detected")
        print("!! KIDNAPPED ( #%d ) -- belief is now wrong on purpose !!"
              % self.n_kidnaps)
        if self.mcl_on:
            self.spread_particles()
        if self.ekf_on:
            for i in range(3):
                self.ekf_P[i][i] = 1.0
            if self.ekf_known:
                print("EKF-LOC: pose covariance inflated; pinned landmarks "
                      "will re-anchor me")
            else:
                nn = len(self.ekf_x)
                for i in range(3, nn):
                    self.ekf_x[i] = 0.0
                    self.ekf_P[i][i] = 1e6
                self.ekf_seen = [False] * len(LANDMARKS)
                print("EKF-SLAM: landmarks re-initialized from the (wrong) "
                      "pose -- expect a WRONG but self-consistent frame")

    # ------------------------------------------------------- sensor lab (L7)

    def corrupt_scan(self):
        img = self.lidar.getRangeImage() or []
        _, sigma, drop, bias, ghost = NOISE_PROFILES[self.noise_idx]
        out = []
        for a, r in zip(self.ray_angles, img):
            if not (self.rmin < r < self.rmax):
                continue
            r2 = r + bias + random.gauss(0, sigma)
            if drop and random.random() < drop:
                continue
            if self.rmin < r2 < self.rmax:
                out.append((a, r2))
        if ghost:
            for _ in range(int(round(ghost * max(1, len(out))))):
                out.append((random.uniform(-math.pi, math.pi),
                            random.uniform(0.6, self.rmax)))
        return out

    # -------------------------------------------------------------- SLAM: map

    def scan_hits(self, corrupted):
        return [(r * math.cos(a), r * math.sin(a))
                for a, r in corrupted if self.rmin < r < self.rmax]

    def _paint(self, cx, cy, delta):
        v = self.grid[cy][cx] + delta
        self.grid[cy][cx] = L_MIN if v < L_MIN else (L_MAX if v > L_MAX else v)
        self._map_dirty = True

    def integrate(self, hits, pose):
        px, py, th = pose
        c, s = math.cos(th), math.sin(th)
        x0, y0 = cell_of(px, py)
        for lx, ly in hits:
            x1, y1 = cell_of(px + lx * c - ly * s, py + lx * s + ly * c)
            for cx, cy in bresenham(x0, y0, x1, y1)[:-1]:
                self._paint(cx, cy, self.L_FREE)
            self._paint(x1, y1, self.L_OCC)

    def apply_decay(self, dt):
        d = DECAY_RATE * dt
        for cy in range(GRID_N):
            row = self.grid[cy]
            for cx in range(GRID_N):
                v = row[cx]
                if v > 0.0:
                    row[cx] = 0.0 if v <= d else v - d
                elif v < 0.0:
                    row[cx] = 0.0 if -v <= d else v + d
        self._map_dirty = True

    def coverage(self):
        if self.t - self.cov_t >= 1.0:
            k = sum(1 for row in self.grid for v in row
                    if v >= OCC_TH or v <= FREE_TH)
            self.cov_cache = 100.0 * k / (GRID_N * GRID_N)
            self.cov_t = self.t
        return self.cov_cache

    # ------------------------------------------------- SLAM: scan matching

    def _score(self, pts, x, y, th):
        c, s = math.cos(th), math.sin(th)
        tot = 0.0
        for lx, ly in pts:
            cx, cy = cell_of(x + lx * c - ly * s, y + lx * s + ly * c)
            tot += max(0.0, self.grid[cy][cx])
        return tot

    def scan_match(self, hits):
        if len(hits) < 30:
            return
        pts = hits[::2]
        px, py, pth = self.sm
        base = self._score(pts, px, py, pth)
        best, corr = base, (0.0, 0.0, 0.0)
        for dth in (-SM_DTH, 0.0, SM_DTH):
            for dx in (-SM_DT, 0.0, SM_DT):
                for dy in (-SM_DT, 0.0, SM_DT):
                    sc = self._score(pts, px + dx, py + dy, pth + dth)
                    if sc > best:
                        best, corr = sc, (dth, dx, dy)
        dth, dx, dy = corr
        if dth or dx or dy:
            self.sm[0] += dx
            self.sm[1] += dy
            self.sm[2] = wrap(self.sm[2] + dth)
            mag = math.hypot(dx, dy)
            if mag > 0.03 or abs(dth) > 0.05:
                self.sm_flash = self.t + 0.6
                self._snap_info = "+%.0fcm %.0fdeg" % (mag * 100,
                                                       math.degrees(dth))

    # -------------------------------------------- loop closure + pose graph

    def _world(self, lx, ly, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return (pose[0] + lx * c - ly * s, pose[1] + lx * s + ly * c)

    def record_node(self, hits):
        p = self.active_pose()
        if self.nodes:
            lp = self.nodes[-1]["pose"]
            if math.hypot(p[0] - lp[0], p[1] - lp[1]) < PG_NODE_GAP:
                return
            self.pg_edges.append({"i": len(self.nodes) - 1,
                                  "j": len(self.nodes),
                                  "z": rel_pose(lp, p), "w": 1.0,
                                  "loop": False})
        step = max(1, len(hits) // 60)
        self.nodes.append({"pose": list(p), "hits": hits[::step]})
        if len(self.nodes) > PG_MAX_NODES:
            del self.nodes[0]
            keep = []
            for ed in self.pg_edges:
                i, j = ed["i"] - 1, ed["j"] - 1
                if i >= 0 and j >= 0:
                    keep.append({"i": i, "j": j, "z": ed["z"],
                                 "w": ed["w"], "loop": ed["loop"]})
            self.pg_edges = keep

    def _edge_residual(self, ed):
        a = self.nodes[ed["i"]]["pose"]
        b = self.nodes[ed["j"]]["pose"]
        rx, ry, rt = rel_pose(a, b)
        return math.hypot(rx - ed["z"][0], ry - ed["z"][1]) \
            + 0.3 * abs(wrap(rt - ed["z"][2]))

    def try_closure(self, hits):
        if self.t < self.lc_cool or len(hits) < 25 or len(self.nodes) < 20:
            return
        p = self.active_pose()
        cand = None
        for i in range(len(self.nodes) - LC_MIN_SEP):
            np_ = self.nodes[i]["pose"]
            if math.hypot(p[0] - np_[0], p[1] - np_[1]) < LC_RADIUS:
                cand = i
                break
        if cand is None:
            return
        cpts = self.nodes[cand]["hits"][::2]
        buckets = set()
        for lx, ly in cpts:
            buckets.add(cell_of(*self._world(lx, ly,
                                             self.nodes[cand]["pose"])))
        samples = hits[::3]

        def overlap(px, py, pth):
            c, s = math.cos(pth), math.sin(pth)
            hit = 0
            for lx, ly in samples:
                cx, cy = cell_of(px + lx * c - ly * s, py + lx * s + ly * c)
                near = False
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if (nx, ny) in buckets:
                            near = True
                            break
                    if near:
                        break
                hit += 1 if near else 0
            return hit / max(1, len(samples))

        base = overlap(*p)
        best, bcorr = base, (0.0, 0.0, 0.0)
        for dth in (-0.12, -0.06, 0.0, 0.06, 0.12):
            for dx in (-0.24, -0.12, 0.0, 0.12, 0.24):
                for dy in (-0.24, -0.12, 0.0, 0.12, 0.24):
                    if dth == dx == dy == 0.0:
                        continue
                    o = overlap(p[0] + dx, p[1] + dy, p[2] + dth)
                    if o > best:
                        best, bcorr = o, (dth, dx, dy)
        dth, dx, dy = bcorr
        err = math.hypot(dx, dy)
        if best - base < 0.15 or err > LC_MAX_ERR:
            return
        self.lc_cool = self.t + LC_COOLDOWN
        self.lc_err = err
        self.lc_edge = ((self.nodes[cand]["pose"][0],
                         self.nodes[cand]["pose"][1]),
                        (p[0], p[1]), self.t + 2.5)
        last = len(self.nodes) - 1
        corr_live = [p[0] + dx, p[1] + dy, wrap(p[2] + dth)]
        z_cl = rel_pose(self.nodes[cand]["pose"], corr_live)
        live_rel = rel_pose(self.nodes[last]["pose"], self.active_pose())
        z_edge = comp_rel(z_cl, inv_rel(live_rel))
        self.pg_edges.append({"i": cand, "j": last, "z": z_edge,
                              "w": PG_LOOP_W, "loop": True})
        if self.pg_on:
            self.pg_start(cand, last, live_rel)
            self.log_event("loop-detected cand=%d" % cand)
            print("LOOP detected (drift %.2f m): adding constraint edge, "
                  "optimizing pose graph ..." % err)
        else:
            self._redistribute(cand, dx, dy, dth)
            self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
            for node in self.nodes:
                self.integrate(node["hits"], node["pose"])
            self.invalidate_view()
            self.log_event("closure-err=%.3f" % err)
            print("LOOP CLOSURE (linear hack): drift %.2f m redistributed"
                  % err)

    def _redistribute(self, ci, dx, dy, dth):
        cx, cy, _ = self.nodes[ci]["pose"]
        n = len(self.nodes)
        for i in range(ci + 1, n):
            a = (i - ci) / float(n - 1 - ci)
            x, y, th = self.nodes[i]["pose"]
            ox, oy = x - cx, y - cy
            ca, sa = math.cos(a * dth), math.sin(a * dth)
            self.nodes[i]["pose"] = [
                cx + ox * ca - oy * sa + a * dx,
                cy + ox * sa + oy * ca + a * dy,
                wrap(th + a * dth)]
        p = self.active_pose()
        self.set_active([p[0] + dx, p[1] + dy, p[2] + dth])

    # ---- the real back-end: Gauss-Newton on the pose graph, animated -----

    def pg_start(self, cand, last, live_rel):
        w0 = max(0, cand - PG_WINDOW_PAD)
        self.pg_map = list(range(w0, len(self.nodes)))
        lmap = {g: i for i, g in enumerate(self.pg_map)}
        self.pg_nodes = [list(self.nodes[g]["pose"]) for g in self.pg_map]
        self.pg_local = [
            {"i": lmap[ed["i"]], "j": lmap[ed["j"]], "z": ed["z"],
             "w": ed["w"], "loop": ed["loop"]}
            for ed in self.pg_edges
            if ed["i"] in lmap and ed["j"] in lmap]
        self.pg_live_rel = live_rel
        self.pg_iter = 0
        self.pg_J0 = self.pg_J = self.pg_prevJ = None
        self.pg_solving = True

    def _gn_iter(self, nodes, edges):
        n = len(nodes)
        m = 3 * n
        H = [[0.0] * m for _ in range(m)]
        b = [0.0] * m
        J = 0.0
        for ed in edges:
            i, j, z, w = ed["i"], ed["j"], ed["z"], ed["w"]
            xi, yi, ti = nodes[i]
            xj, yj, tj = nodes[j]
            c, s = math.cos(ti), math.sin(ti)
            ddx, ddy = xj - xi, yj - yi
            rx = c * ddx + s * ddy
            ry = -s * ddx + c * ddy
            rt = wrap(tj - ti)
            ex, ey, et = rx - z[0], ry - z[1], wrap(rt - z[2])
            J += w * (ex * ex + ey * ey + et * et)
            Ji = ((-c, -s, ry), (s, -c, -rx), (0.0, 0.0, -1.0))
            Jj = ((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))
            e3 = (ex, ey, et)
            for a in range(3):
                bi = bj = 0.0
                for k in range(3):
                    bi += Ji[k][a] * e3[k]
                    bj += Jj[k][a] * e3[k]
                b[3 * i + a] += w * bi
                b[3 * j + a] += w * bj
                for c2 in range(3):
                    hii = hij = hji = hjj = 0.0
                    for k in range(3):
                        hii += Ji[k][a] * Ji[k][c2]
                        hij += Ji[k][a] * Jj[k][c2]
                        hji += Jj[k][a] * Ji[k][c2]
                        hjj += Jj[k][a] * Jj[k][c2]
                    H[3 * i + a][3 * i + c2] += w * hii
                    H[3 * i + a][3 * j + c2] += w * hij
                    H[3 * j + a][3 * i + c2] += w * hji
                    H[3 * j + a][3 * j + c2] += w * hjj
        for a in range(3):                       # anchor the first node
            for c2 in range(m):
                H[a][c2] = 0.0
                H[c2][a] = 0.0
            H[a][a] = 1.0
            b[a] = 0.0
        d = lin_solve(H, [-v for v in b])
        return J, d

    def pg_iterate(self):
        J, d = self._gn_iter(self.pg_nodes, self.pg_local)
        for k in range(1, len(self.pg_nodes)):
            self.pg_nodes[k][0] += d[3 * k]
            self.pg_nodes[k][1] += d[3 * k + 1]
            self.pg_nodes[k][2] = wrap(self.pg_nodes[k][2] + d[3 * k + 2])
        if self.pg_J0 is None:
            self.pg_J0 = J
        self.pg_J = J
        self.pg_iter += 1
        if (self.pg_iter >= PG_MAX_ITER or
                (self.pg_prevJ is not None
                 and self.pg_prevJ - J < PG_EPS * max(1.0, self.pg_prevJ))):
            self.pg_finish()
        self.pg_prevJ = J

    def pg_finish(self):
        for li, g in enumerate(self.pg_map):
            self.nodes[g]["pose"] = list(self.pg_nodes[li])
        self.grid = [[0.0] * GRID_N for _ in range(GRID_N)]
        for node in self.nodes:
            self.integrate(node["hits"], node["pose"])
        last_pose = list(self.nodes[self.pg_map[-1]]["pose"])
        self.set_active(comp_rel(last_pose, self.pg_live_rel))
        self.invalidate_view()
        self.pg_solving = False
        self.pg_flash = self.t + 3.0
        self.log_event("pg-done iters=%d J0=%.2f J=%.2f"
                       % (self.pg_iter, self.pg_J0, self.pg_J))
        print("POSE GRAPH optimized: %d GN iterations, cost %.2f -> %.2f; "
              "map re-integrated" % (self.pg_iter, self.pg_J0, self.pg_J))

    # --------------------------------------------------------- SLAM: MCL

    def seed_particles(self, pose, sigma):
        self.particles = []
        for _ in range(self.n_particles):
            self.particles.append(
                [pose[0] + random.gauss(0, sigma),
                 pose[1] + random.gauss(0, sigma),
                 wrap(pose[2] + random.gauss(0, sigma)), 1.0])

    def spread_particles(self):
        free = [(cx, cy) for cy in range(GRID_N) for cx in range(GRID_N)
                if self.grid[cy][cx] < OCC_TH]
        if not free:
            return
        self.particles = []
        for _ in range(self.n_particles):
            cx, cy = random.choice(free)
            self.particles.append([
                (cx + 0.5) * CELL - MAP_SIZE / 2
                + random.gauss(0, KIDNAP_SPREAD_J),
                (cy + 0.5) * CELL - MAP_SIZE / 2
                + random.gauss(0, KIDNAP_SPREAD_J),
                wrap(random.uniform(-math.pi, math.pi)), 1.0])
        print("MCL: particles re-scattered over %d cells" % len(free))

    def mcl_predict(self, dist, dth):
        for p in self.particles:
            d = dist * (1 + random.gauss(0, MCL_DIST_J)) \
                + random.gauss(0, MCL_DIST_A)
            t = dth * (1 + random.gauss(0, MCL_TH_J)) \
                + random.gauss(0, MCL_TH_A)
            mid = p[2] + 0.5 * t
            p[0] += d * math.cos(mid)
            p[1] += d * math.sin(mid)
            p[2] = wrap(p[2] + t)

    def mcl_update(self, hits):
        if not hits:
            return
        pts = hits[::max(1, len(hits) // 40)]
        tot = 0.0
        for p in self.particles:
            p[3] = 1e-6 + self._score(pts, p[0], p[1], p[2])
            tot += p[3]
        if tot <= 0:
            return
        sq = 0.0
        for p in self.particles:
            p[3] /= tot
            sq += p[3] * p[3]
        self.ess = 1.0 / sq if sq > 0 else self.n_particles
        if self.ess < self.n_particles * MCL_ESS_FRAC:
            self._resample()

    def _resample(self):
        n = self.n_particles
        step = 1.0 / n
        r = random.uniform(0, step)
        cum, out = 0.0, []
        for p in self.particles:
            cum += p[3]
            while r <= cum and len(out) < n:
                out.append([p[0] + random.gauss(0, MCL_JITTER_X),
                            p[1] + random.gauss(0, MCL_JITTER_X),
                            wrap(p[2] + random.gauss(0, MCL_JITTER_T)),
                            1.0 / n])
                r += step
        while len(out) < n:
            out.append(list(self.particles[-1])[:3] + [1.0 / n])
        self.particles = out
        self.resample_flash = True

    def mcl_estimate(self):
        sx = sy = st = cs = 0.0
        for p in self.particles:
            sx += p[3] * p[0]
            sy += p[3] * p[1]
            st += p[3] * math.sin(p[2])
            cs += p[3] * math.cos(p[2])
        self.mcl[0], self.mcl[1] = sx, sy
        self.mcl[2] = wrap(math.atan2(st, cs))

    # ------------------------------------------------------- AMCL mode (L)

    def amcl_load(self):
        try:
            with open(self.map_path, "rb") as f:
                raw = f.read()
            i = raw.index(b"255\n") + 4
            hdr = raw[:i].split()
            w, h = int(hdr[1]), int(hdr[2])
            data = raw[i:]
            if w != GRID_N or h != GRID_N or len(data) < w * h:
                raise ValueError
        except (OSError, ValueError):
            print("AMCL: no usable %s -- run a mapping lesson first, "
                  "then press L" % self.map_path)
            return
        for cy in range(GRID_N):
            src = data[(GRID_N - 1 - cy) * GRID_N:(GRID_N - cy) * GRID_N]
            for cx in range(GRID_N):
                p = 1.0 - src[cx] / 255.0
                p = min(0.999, max(0.001, p))
                lo = math.log(p / (1.0 - p))
                self.grid[cy][cx] = min(L_MAX, max(L_MIN, lo))
        self.invalidate_view()
        self.mapping_on = False
        self.sm_on = False
        self.mcl_on = True
        self.frontier_on = False
        self.goal = None
        self.spread_particles()
        self.log_event("amcl-load")
        print("AMCL: map loaded, MAPPING OFF -- localization only. "
              "Cloud is GLOBAL (uniform). Press K to kidnap.")

    # --------------------------------------------------- EKF landmarks (L9/L10)

    def ekf_init(self):
        known = getattr(self, "ekf_known", False)
        n = 3 + 2 * len(LANDMARKS)
        self.ekf_x = [self.odom[0], self.odom[1], self.odom[2]] \
            + [0.0] * (n - 3)
        self.ekf_P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            self.ekf_P[i][i] = 0.0025 if i < 3 else 1e6
        self.ekf_seen = [False] * len(LANDMARKS)
        self.ekf_meas = []
        if known:
            for i, (bx, by) in enumerate(LANDMARKS):
                li = 3 + 2 * i
                self.ekf_x[li], self.ekf_x[li + 1] = bx, by
                self.ekf_P[li][li] = self.ekf_P[li + 1][li + 1] = 1e-6
                self.ekf_P[li][li + 1] = self.ekf_P[li + 1][li] = 0.0
                self.ekf_seen[i] = True

    def ekf_set_known(self, known):
        self.ekf_known = known
        if not self.ekf_on or self.ekf_x is None:
            return
        if known:
            for i, (bx, by) in enumerate(LANDMARKS):
                li = 3 + 2 * i
                self.ekf_x[li], self.ekf_x[li + 1] = bx, by
                self.ekf_P[li][li] = self.ekf_P[li + 1][li + 1] = 1e-6
                self.ekf_P[li][li + 1] = self.ekf_P[li + 1][li] = 0.0
                self.ekf_seen[i] = True
            print("EKF: landmarks PINNED to the map (localization mode) -- "
                  "kidnap recovery now possible (U to undo)")
        else:
            nn = len(self.ekf_x)
            for i in range(3, nn):
                self.ekf_P[i][i] = 1e6
            self.ekf_seen = [False] * len(LANDMARKS)
            print("EKF: landmarks UNKNOWN again (SLAM mode)")

    def beacon_measure(self):
        """L9 TEACHING CHEAT: simulated range+bearing to known-id beacons."""
        tx, ty, tth = self.truth
        out = []
        for i, (bx, by) in enumerate(LANDMARKS):
            dx, dy = bx - tx, by - ty
            r = math.hypot(dx, dy)
            if r > BEA_RANGE:
                continue
            zb = wrap(math.atan2(dy, dx) - tth + random.gauss(0, BEA_SB))
            zr = r * (1 + random.gauss(0, 0.015)) + random.gauss(0, BEA_SR)
            out.append((i, zr, zb))
        return out

    def cam_measure(self):
        """L10 REAL sensor: color-blob detection on the camera image.
        Bearing from the blob centroid; range from the blob's angular width
        (a 0.30 m pillar). Occlusion shrinks the blob -> range overestimates:
        a real monocular-metrology failure mode, deliberately kept."""
        if self.cam is None:
            return []
        img = self.cam.getImage()
        w, h = self.cam.getWidth(), self.cam.getHeight()
        fov = self.cam.getFov()
        gr = self.cam.imageGetRed
        gg = self.cam.imageGetGreen
        gb = self.cam.imageGetBlue
        stats = [[0, 0.0, w, -1] for _ in range(4)]
        for y in range(h):
            for x in range(w):
                r = gr(img, w, x, y) / 255.0
                g = gg(img, w, x, y) / 255.0
                b = gb(img, w, x, y) / 255.0
                if r + g + b < 0.4:
                    continue
                k = -1
                if r > 0.55 and r > 1.4 * g and r > 1.4 * b:
                    k = 0
                elif g > 0.5 and g > 1.4 * r and g > 1.4 * b:
                    k = 1
                elif b > 0.55 and b > 1.4 * r and b > 1.4 * g:
                    k = 2
                elif r > 0.5 and g > 0.7 * r and b < 0.35 * min(r, g):
                    k = 3
                if k >= 0:
                    st = stats[k]
                    st[0] += 1
                    st[1] += x
                    if x < st[2]:
                        st[2] = x
                    if x > st[3]:
                        st[3] = x
        out = []
        for k, st in enumerate(stats):
            if st[0] < 8:
                continue
            cx = st[1] / st[0]
            aw = (st[3] - st[2] + 1) / float(w) * fov
            if aw < 1e-3:
                continue
            zr = CAM_WIDTH / (2.0 * math.tan(aw / 2.0))
            if not (0.3 < zr < 4.5):
                continue
            zb = wrap((w / 2 - cx) / w * fov)
            out.append((k, zr, zb))
        return out

    def ekf_step(self, dist, dth, meas):
        x, P = self.ekf_x, self.ekf_P
        n = len(x)
        th = x[2]
        x[0] += dist * math.cos(th + 0.5 * dth)
        x[1] += dist * math.sin(th + 0.5 * dth)
        x[2] = wrap(th + dth)
        F = mat_eye(n)
        F[0][2] = -dist * math.sin(th + 0.5 * dth)
        F[1][2] = dist * math.cos(th + 0.5 * dth)
        qd = (0.02 * abs(dist) + 0.001) ** 2
        qt = (0.03 * abs(dth) + 0.001) ** 2
        Q = mat_zero(n, n)
        Q[0][0] = Q[1][1] = qd
        Q[2][2] = qt
        P = mat_add(mat_mul(mat_mul(F, P), mat_T(F)), Q)
        self.ekf_meas = []
        R = [[BEA_SR ** 2, 0.0], [0.0, BEA_SB ** 2]]
        for i, zr, zb in meas:
            self.ekf_meas.append(i)
            li = 3 + 2 * i
            if not self.ekf_seen[i]:
                x[li] = x[0] + zr * math.cos(x[2] + zb)
                x[li + 1] = x[1] + zr * math.sin(x[2] + zb)
                P[li][li] = P[li + 1][li + 1] = 0.0025
                P[li][li + 1] = P[li + 1][li] = 0.0
                self.ekf_seen[i] = True
            dx = x[li] - x[0]
            dy = x[li + 1] - x[1]
            q = dx * dx + dy * dy
            r = math.sqrt(q)
            bb = math.atan2(dy, dx) - x[2]
            y0 = zr - r
            y1 = wrap(zb - bb)
            H = mat_zero(2, n)
            H[0][0], H[0][1] = -dx / r, -dy / r
            H[1][0], H[1][1] = dy / q, -dx / q
            H[1][2] = -1.0
            H[0][li], H[0][li + 1] = dx / r, dy / r
            H[1][li], H[1][li + 1] = -dy / q, dx / q
            S = mat_add(mat_mul(mat_mul(H, P), mat_T(H)), R)
            K = mat_mul(mat_mul(P, mat_T(H)), mat_inv2(S))
            for j in range(n):
                x[j] += K[j][0] * y0 + K[j][1] * y1
            A = mat_sub(mat_eye(n), mat_mul(K, H))
            # Joseph form -- keeps P positive semi-definite
            P = mat_add(mat_mul(mat_mul(A, P), mat_T(A)),
                        mat_mul(mat_mul(K, R), mat_T(K)))
        x[2] = wrap(x[2])
        self.ekf_x, self.ekf_P = x, P

    # ------------------------------------------------- frontier planning (L8)

    def find_frontiers(self):
        if self.front_cache and self.t - self.front_cache_t < 1.0:
            return self.front_cache
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
                    clusters.append({"size": len(cells),
                                     "x": (mx + 0.5) * CELL - MAP_SIZE / 2,
                                     "y": (my + 0.5) * CELL - MAP_SIZE / 2})
        self.front_cache, self.front_cache_t = clusters, self.t
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

    def pick_frontier(self):
        clusters = self.find_frontiers()
        self.n_front = len(clusters)
        if not clusters:
            return
        p = self.active_pose()
        clusters.sort(key=lambda c: math.hypot(c["x"] - p[0], c["y"] - p[1]))
        blocked = self._blocked_mask()
        for c in clusters[:4]:
            if any(math.hypot(c["x"] - ax, c["y"] - ay) < ABANDON_R
                   and self.t - at < ABANDON_COOL
                   for ax, ay, at in self.abandoned):
                continue
            goal = self._snap(c["x"], c["y"], blocked)
            if goal is None:
                continue
            cells = self._astar(cell_of(p[0], p[1]), goal, blocked)
            if cells is None:
                continue
            while len(cells) > 2 and not self._los(cells[0], cells[2],
                                                   blocked):
                cells.pop(1)
            self.path_cells = cells
            self.wp_i = 0
            self.goal = c
            self.g_t0 = self.t
            return

    def abandon(self, why):
        if self.goal:
            self.abandoned.append((self.goal["x"], self.goal["y"], self.t))
            if len(self.abandoned) > 20:
                self.abandoned.pop(0)
            print("frontier abandoned (%s)" % why)
        self.goal = None
        self.path_cells = None

    def follow(self):
        if self.t - self.g_t0 > FR_TIMEOUT:
            self.abandon("timeout")
            return
        p = self.active_pose()
        if math.hypot(self.goal["x"] - p[0],
                      self.goal["y"] - p[1]) < GOAL_TOL:
            self.goal = None
            self.path_cells = None
            return
        while self.wp_i < len(self.path_cells) - 1:
            wx = (self.path_cells[self.wp_i][0] + 0.5) * CELL - MAP_SIZE / 2
            wy = (self.path_cells[self.wp_i][1] + 0.5) * CELL - MAP_SIZE / 2
            if math.hypot(wx - p[0], wy - p[1]) < 0.18:
                self.wp_i += 1
            else:
                break
        wx = (self.path_cells[self.wp_i][0] + 0.5) * CELL - MAP_SIZE / 2
        wy = (self.path_cells[self.wp_i][1] + 0.5) * CELL - MAP_SIZE / 2
        e = wrap(math.atan2(wy - p[1], wx - p[0]) - p[2])
        if abs(e) > 1.0:
            s = 5.0 if e > 0 else -5.0
            self.actuate(-s, s)
        else:
            self.actuate(V_LIN * (1 - 1.2 * e), V_LIN * (1 + 1.2 * e))

    # ---------------------------------------------------------------- drawing

    def invalidate_view(self):
        self._disp = bytearray([0xFF]) * (GRID_N * GRID_N)
        self._overlay_cells = []
        self._map_dirty = True
        self.trail_dirty = True
        self.front_cache, self.front_cache_t = [], -1e9
        self.cov_t = -1e9

    def _sx(self, cx):
        return MAP_X + cx * PX

    def _sy(self, cy):
        return MAP_Y + (GRID_N - 1 - cy) * PX

    def w2s(self, x, y):
        return (MAP_X + (x + MAP_SIZE / 2.0) / CELL * PX,
                MAP_Y + (MAP_SIZE / 2.0 - y) / CELL * PX)

    def _repaint_cell(self, cx, cy):
        v = self.grid[cy][cx]
        p = 1.0 / (1.0 + math.exp(-v))
        g = int(255 * (1.0 - p))
        self._disp[cy * GRID_N + cx] = g >> 3
        self.display.setColor(g * 0x010101)
        self.display.fillRectangle(self._sx(cx), self._sy(cy), PX, PX)

    def _track(self, cx, cy):
        if 0 <= cx < GRID_N and 0 <= cy < GRID_N:
            self._overlay_cells.append((cx, cy))

    def _tline(self, x0, y0, x1, y1, color):
        d = self.display
        d.setColor(color)
        ax, ay = self.w2s(x0, y0)
        bx, by = self.w2s(x1, y1)
        d.drawLine(int(ax), int(ay), int(bx), int(by))
        c0 = cell_of(x0, y0)
        c1 = cell_of(x1, y1)
        for cx, cy in bresenham(c0[0], c0[1], c1[0], c1[1]):
            self._track(cx, cy)

    def _trail_step(self, dq, p, col):
        d = self.display
        if dq:
            d.setColor(col)
            a = self.w2s(*dq[-1])
            b = self.w2s(*p)
            d.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        dq.append((p[0], p[1]))

    def _redraw_trails(self):
        d = self.display
        specs = [(self.raw_trail, COL_TRAIL_RAW),
                 (self.true_trail, COL_TRUTH),
                 (self.est_trail, COL_TRAIL)]
        for i, (tr, col) in enumerate(specs):
            if i == 1 and not self.show_truth:
                continue
            if len(tr) < 2:
                continue
            d.setColor(col)
            prev = None
            for p in tr:
                pt = self.w2s(*p)
                if prev:
                    d.drawLine(int(prev[0]), int(prev[1]),
                               int(pt[0]), int(pt[1]))
                prev = pt

    def draw_map_full(self):
        if not self._map_dirty:
            return
        d = self.display
        for cy in range(GRID_N):
            row = self.grid[cy]
            sy = self._sy(cy)
            base = cy * GRID_N
            for cx in range(GRID_N):
                v = row[cx]
                p = 1.0 / (1.0 + math.exp(-v))
                g = int(255 * (1.0 - p))
                q = g >> 3
                if self._disp[base + cx] == q:
                    continue
                self._disp[base + cx] = q
                d.setColor(g * 0x010101)
                d.fillRectangle(self._sx(cx), sy, PX, PX)
        self._map_dirty = False

    def _ellipse(self, x, y, cov, color):
        a, b, c = cov[0][0], cov[0][1], cov[1][1]
        tr = a + c
        dt = math.sqrt(max(0.0, ((a - c) / 2.0) ** 2 + b * b))
        l1 = (tr + dt) / 2.0
        l2 = max(1e-9, (tr - dt) / 2.0)
        ang = 0.5 * math.atan2(2 * b, a - c)
        r1 = min(math.sqrt(max(0.0, l1)) * 2.0, 5.0)
        r2 = min(math.sqrt(max(0.0, l2)) * 2.0, 5.0)
        prev = None
        for k in range(15):
            t = 2 * math.pi * k / 14.0
            ex, ey = r1 * math.cos(t), r2 * math.sin(t)
            wx = x + ex * math.cos(ang) - ey * math.sin(ang)
            wy = y + ex * math.sin(ang) + ey * math.cos(ang)
            if prev:
                self._tline(prev[0], prev[1], wx, wy, color)
            prev = (wx, wy)

    def draw_overlays(self, corrupted):
        d = self.display
        for cx, cy in self._overlay_cells:
            self._repaint_cell(cx, cy)
        self._overlay_cells = []
        if self.trail_dirty:
            self._redraw_trails()
            self.trail_dirty = False
        show_edges = self.pg_solving or self.t < self.pg_flash
        if self.closure_on:
            d.setColor(COL_NODE)
            for node in self.nodes:
                cx, cy = cell_of(node["pose"][0], node["pose"][1])
                self._track(cx, cy)
                d.fillRectangle(self._sx(cx), self._sy(cy), 2, 2)
            for ed in self.pg_edges:
                if ed["i"] >= len(self.nodes) or ed["j"] >= len(self.nodes):
                    continue
                if ed["loop"]:
                    col = COL_EDGE
                elif show_edges:
                    r = self._edge_residual(ed)
                    col = (COL_EDGE_OK if r < 0.05 else
                           COL_EDGE_MID if r < 0.15 else COL_EDGE_BAD)
                else:
                    continue
                a = self.nodes[ed["i"]]["pose"]
                b2 = self.nodes[ed["j"]]["pose"]
                self._tline(a[0], a[1], b2[0], b2[1], col)
        if self.lc_edge and self.t < self.lc_edge[2]:
            (x1, y1), (x2, y2), _ = self.lc_edge
            self._tline(x1, y1, x2, y2, COL_EDGE)
        if self.path_cells:
            d.setColor(COL_PATH)
            for cx, cy in self.path_cells:
                self._track(cx, cy)
                d.fillRectangle(self._sx(cx), self._sy(cy), PX, PX)
        if self.frontier_on and self.n_front:
            d.setColor(COL_FRONT)
            for c in self.find_frontiers()[:8]:
                cx, cy = cell_of(c["x"], c["y"])
                self._track(cx, cy)
                self._track(cx, min(GRID_N - 1, cy + 1))
                d.fillRectangle(self._sx(cx) - 1, self._sy(cy) - 1, 4, 4)
        if self.ekf_on and self.ekf_x is not None:
            for i, seen in enumerate(self.ekf_seen):
                if not seen:
                    continue
                bx, by = self.ekf_x[3 + 2 * i], self.ekf_x[4 + 2 * i]
                col = CAM_COLS[i] if self.cam_on else COL_EKFC
                self._tline(bx - 0.12, by, bx + 0.12, by, col)
                self._tline(bx, by - 0.12, bx, by + 0.12, col)
                self._ellipse(bx, by,
                              [[self.ekf_P[3 + 2 * i][3 + 2 * i],
                                self.ekf_P[3 + 2 * i][4 + 2 * i]],
                               [self.ekf_P[4 + 2 * i][3 + 2 * i],
                                self.ekf_P[4 + 2 * i][4 + 2 * i]]], col)
            self._ellipse(self.ekf_x[0], self.ekf_x[1],
                          [[self.ekf_P[0][0], self.ekf_P[0][1]],
                           [self.ekf_P[1][0], self.ekf_P[1][1]]], COL_EKFC)
            for i in self.ekf_meas:
                self._tline(self.ekf_x[0], self.ekf_x[1],
                            LANDMARKS[i][0], LANDMARKS[i][1], 0xFFD020)
        if self.cam_on:
            est = self.active_pose()
            for k, zr, zb in self.cam_meas:
                self._tline(est[0], est[1],
                            est[0] + min(zr, 3.0) * math.cos(est[2] + zb),
                            est[1] + min(zr, 3.0) * math.sin(est[2] + zb),
                            CAM_COLS[k])
        for trail, col in ((self.raw_trail, COL_TRAIL_RAW),
                           (self.true_trail, COL_TRUTH) if self.show_truth
                           else ((), 0),
                           (self.est_trail, COL_TRAIL)):
            if len(trail) < 2:
                continue
            d.setColor(col)
            prev = None
            for p in trail:
                pt = self.w2s(*p)
                if prev:
                    d.drawLine(int(prev[0]), int(prev[1]),
                               int(pt[0]), int(pt[1]))
                prev = pt
        if self.mcl_on and self.show_particles:
            for p in self.particles:
                cx, cy = cell_of(p[0], p[1])
                self._track(cx, cy)
                d.setColor(COL_PART if p[3] > 1.5 / self.n_particles
                           else 0x301040)
                d.fillRectangle(self._sx(cx), self._sy(cy), 2, 2)
        if self.show_truth:
            self._tline(self.truth[0] - 0.10, self.truth[1],
                        self.truth[0] + 0.10, self.truth[1], COL_TRUTH)
            self._tline(self.truth[0], self.truth[1] - 0.10,
                        self.truth[0], self.truth[1] + 0.10, COL_TRUTH)
        est = self.active_pose()
        ex, ey = cell_of(est[0], est[1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self._track(ex + dx, ey + dy)
        d.setColor(COL_TRAIL)
        sxp, syp = self.w2s(est[0], est[1])
        d.fillOval(int(sxp) - 2, int(syp) - 2, 5, 5)
        self._tline(est[0], est[1],
                    est[0] + 0.14 * math.cos(est[2]),
                    est[1] + 0.14 * math.sin(est[2]), COL_TRAIL)
        if self.show_beams:
            est2 = self.active_pose()
            for i in range(0, len(corrupted), 25):
                a, r = corrupted[i]
                wx, wy = self._world(r * math.cos(a), r * math.sin(a), est2)
                hx, hy = cell_of(wx, wy)
                x0, y0 = cell_of(est2[0], est2[1])
                path = bresenham(x0, y0, hx, hy)
                d.setColor(COL_FREEPATH)
                for bx, by in path[:-1]:
                    self._track(bx, by)
                    d.fillRectangle(self._sx(bx), self._sy(by), PX, PX)
                self._track(hx, hy)
                d.setColor(COL_HIT)
                d.fillRectangle(self._sx(hx), self._sy(hy), PX, PX)

    def draw_hud(self):
        d = self.display
        d.setColor(HUD_COL)
        d.fillRectangle(0, 0, 300, MAP_Y)
        d.fillRectangle(0, MAP_Y + GRID_N * PX, 300, 56)
        est = self.active_pose()

        def fa(v):
            return "%5.2f" % v if v == v else "  n/a"

        d.setColor(COL_TEXT)
        mode = "MAP" if self.mapping_on else "LOC"
        ekfmode = ""
        if self.ekf_on:
            ekfmode = " EKF-LOC" if self.ekf_known else " EKF-SLAM"
        d.drawText("L%d %s%s%s%s%s%s%s" % (
            self.lesson, mode,
            " SM" if self.sm_on else "", " MCL" if self.mcl_on else "",
            ekfmode, " LCL" if self.closure_on else "",
            " GN" if self.pg_on else "",
            " DEC" if self.decay_on else ""), 4, 3)
        ate = self.ate(est)
        d.setColor(0x20D060 if ate == ate and ate < 0.10 else
                   (0xFFD020 if ate == ate and ate < 0.30 else 0xFF3030))
        d.drawText("ATE %s   raw %s" % (fa(ate), fa(self.ate(self.odom))),
                   4, 12)
        d.setColor(COL_TEXT)
        lm = ("  lm %d/4" % sum(self.ekf_seen)) if self.ekf_on else ""
        d.drawText("cov %3.0f%%  noise %s%s  ess %d  front %d%s" % (
            self.coverage(), NOISE_PROFILES[self.noise_idx][0],
            " DMG" if self.odom_damage else "", int(self.ess),
            self.n_front, lm), 4, 21)
        pg = ""
        if self.pg_solving:
            pg = "  GN k=%d J %.1f->%.1f" % (self.pg_iter, self.pg_J0
                                             or 0, self.pg_J or 0)
        elif self.pg_J is not None and self.t < self.pg_flash + 10:
            pg = "  GN J %.1f->%.1f" % (self.pg_J0 or 0, self.pg_J)
        d.drawText("Lo %.2f Lf %.2f  closure %s%s" % (
            self.L_OCC, self.L_FREE,
            "%.2fm" % self.lc_err if self.lc_err is not None else "--",
            pg), 4, 30)
        d.setColor(COL_DIM)
        for i, line in enumerate([
                "0-9,X lessons  SPACE pause",
                "S sm  M mcl  E ekf  U ekf-loc",
                "C closure  Y graph  D decay",
                "N noise  O dmg  F frontier  L amcl",
                "B beams  G truth  P particles",
                "R reset  K kidnap",
                "[ ] L_OCC   - = L_FREE"]):
            d.drawText(line, LEG_X, 6 + i * 9)
        if self.t < self.kidnap_flash:
            d.setColor(0xFF3030)
            d.drawText("** KIDNAPPED **", 84, 262)
        elif self.pg_solving:
            d.setColor(0x60A0FF)
            d.drawText("OPTIMIZING GRAPH ...", 86, 262)
        elif self.t < self.sm_flash:
            d.setColor(0xFFD020)
            d.drawText("SM SNAP %s" % self._snap_info, 90, 262)
        elif self.resample_flash:
            d.setColor(0xFFD020)
            d.drawText("RESAMPLE", 122, 262)
            self.resample_flash = False
        elif self.paused:
            d.setColor(0xFFD020)
            d.drawText("PAUSED", 128, 262)

    # ---------------------------------------------------------------- logging

    def ate(self, pose):
        if not self.truth_on:
            return float("nan")
        return math.hypot(pose[0] - self.truth[0],
                          pose[1] - self.truth[1])

    def log_event(self, name):
        try:
            with open(self.ev_path, "a") as f:
                f.write("%.1f,%s\n" % (self.t, name))
        except OSError:
            pass

    def log_pose(self):
        def d(v):
            return "%.3f" % v if self.truth_on else "nan"
        est = self.active_pose()
        kx = ky = kth = "nan"
        if self.ekf_on and self.ekf_x is not None:
            kx, ky, kth = ("%.3f" % v for v in self.ekf_x[:3])
        try:
            with open(self.log_path, "a") as f:
                f.write("%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,"
                        "%s,%s,%s\n" % (
                            self.t, est[0], est[1], est[2],
                            self.odom[0], self.odom[1], self.odom[2],
                            d(self.truth[0]), d(self.truth[1]),
                            d(self.truth[2]), kx, ky, kth))
        except OSError:
            pass

    def log_cov(self):
        try:
            with open(self.cov_path, "a") as f:
                f.write("%.1f,%.1f,%d,%s\n" % (
                    self.t, self.coverage(), self.lesson,
                    "LOC" if not self.mapping_on else "MAP"))
        except OSError:
            pass

    def write_final(self):
        try:
            with open(self.fin_path, "w") as f:
                f.write("%.3f,%.3f,%.3f,%.3f,%.1f\n" % (
                    self.ate(self.sm), self.ate(self.odom),
                    self.ate(self.ekf_x[:3]) if self.ekf_on else -1.0,
                    self.ate(self.mcl), self.coverage()))
        except OSError:
            pass

    def save_map(self):
        try:
            with open(self.map_path, "wb") as f:
                f.write(("P5\n%d %d\n255\n" % (GRID_N, GRID_N))
                        .encode("ascii"))
                data = bytearray(GRID_N * GRID_N)
                i = 0
                for cy in range(GRID_N - 1, -1, -1):
                    for cx in range(GRID_N):
                        p = 1.0 / (1.0 + math.exp(-self.grid[cy][cx]))
                        data[i] = int(255 * (1.0 - p))
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

            if self.t < 2.5 and self.n % 4 == 0:
                self._apply_teach_config()

            events = set()
            k = self.kb.getKey()
            while k > 0:
                events.add(k)
                self._held_t[k] = self.t
                k = self.kb.getKey()
            self._held = set(k for k, tt in self._held_t.items()
                             if self.t - tt <= HELD_TIMEOUT)
            edges = set(k for k in events if k not in self._prev_held)
            self._prev_held = set(self._held)
            if len(self._held_t) > 64:
                self._held_t = dict(self._held)
            self.handle_keys(edges)

            self.read_truth()
            moved = self.update_odometry()
            corrupted = self.corrupt_scan()
            hits = self.scan_hits(corrupted)
            front = min([rg for a, rg in corrupted if -0.62 <= a <= 0.62]
                        or [math.inf])
            left = min([rg for a, rg in corrupted if 0.65 <= a <= 1.75]
                       or [math.inf])
            right = min([rg for a, rg in corrupted if -1.75 <= a <= -0.65]
                        or [math.inf])

            if self.decay_on and self.n % SCAN_EVERY == 0:
                self.apply_decay(SCAN_EVERY * self.ts / 1000.0)

            if self.n % SCAN_EVERY == 0 and hits:
                if self.ekf_on:
                    meas = self.cam_measure() if self.cam_on \
                        else self.beacon_measure()
                    if self.cam_on:
                        self.cam_meas = meas
                    self.ekf_step(moved[0], moved[1], meas)
                    if self.cam_on:
                        for i, sn in enumerate(self.ekf_seen):
                            if sn and i not in self.cam_logged:
                                self.cam_logged.add(i)
                                self.log_event("cam-seen-%d" % i)
                if self.mcl_on:
                    self.mcl_predict(*moved)
                    self.mcl_update(hits)
                    self.mcl_estimate()
                if self.sm_on:
                    self.scan_match(hits)
                if self.mapping_on:
                    self.integrate(hits, self.active_pose())
                if self.closure_on and not self.pg_solving:
                    self.record_node(hits)
                    self.try_closure(hits)
            if self.pg_solving:
                self.pg_iterate()

            if self.frontier_on and self.goal is None:
                self.fr_tick -= 1
                if self.fr_tick <= 0:
                    self.fr_tick = 30
                    self.pick_frontier()

            self.drive(front, left, right)

            if self.n % DRAW_FULL_EVERY == 0:
                self._trail_step(self.est_trail, self.active_pose()[:2],
                                 COL_TRAIL)
                self._trail_step(self.raw_trail, self.odom[:2],
                                 COL_TRAIL_RAW)
                if self.truth_on:
                    self._trail_step(self.true_trail, self.truth[:2],
                                     COL_TRUTH)

            self.draw_map_full()
            self.draw_overlays(corrupted)
            if self.n % HUD_EVERY == 0:
                self.draw_hud()

            if self.n % LOG_EVERY == 0:
                self.log_pose()
                self.write_final()
            if self.n % (5 * 1000 // self.ts) == 0:
                self.log_cov()

        self.log_pose()
        self.write_final()
        self.save_map()
        print("done: ATE(sm)=%.3f ekf=%.3f odom=%.3f kidnaps=%d" % (
            self.ate(self.sm),
            self.ate(self.ekf_x[:3]) if self.ekf_on else -1,
            self.ate(self.odom), self.n_kidnaps))
        print("analyze:  python analysis.py pose_log.csv --plot --cov")
        print("grade:     python ../grader/grader.py")


if __name__ == "__main__":
    Teacher().run()
