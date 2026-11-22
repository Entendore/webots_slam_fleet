"""Teleports the classroom robot to demonstrate the kidnapped-robot problem.

Triggers: the robot's K key (writes kidnap.now here) or --every SECONDS.
--random picks an unpredictable target (at least 1.5 m away) instead of a
fixed offset.
"""

import argparse
import math
import os
import random

from controller import Supervisor

DEF_NAME = "CLASS_ROBOT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=0.0)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=-0.7)
    args, _ = ap.parse_known_args()
    random.seed(args.seed or None)

    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())
    here = os.path.dirname(os.path.abspath(__file__))
    flag = os.path.join(here, "kidnap.now")
    node = sup.getFromDef(DEF_NAME)
    if node is None:
        print("[kidnap] no DEF %s in world -- idling" % DEF_NAME)
        return

    def pick(cur):
        if not args.random:
            return (max(-4.2, min(4.2, cur[0] + args.dx)),
                    max(-4.2, min(4.2, cur[1] + args.dy)))
        for _ in range(200):
            nx, ny = random.uniform(-4.0, 4.0), random.uniform(-4.0, 4.0)
            if math.hypot(nx - cur[0], ny - cur[1]) > 1.5:
                return nx, ny
        return cur[0] + args.dx, cur[1] + args.dy

    def kidnap(tag):
        f = node.getField("translation")
        p = f.getSFVec3f()
        nx, ny = pick(p)
        f.setSFVec3f([nx, ny, p[2]])
        node.resetPhysics()
        print("[kidnap] %s: robot -> (%.2f, %.2f)" % (tag, nx, ny))

    nxt = args.every
    while sup.step(ts) != -1:
        if os.path.exists(flag):
            try:
                os.remove(flag)
            except OSError:
                pass
            kidnap("key request")
        if args.every > 0:
            t = sup.getTime()
            if t >= nxt:
                kidnap("timer")
                nxt = t + args.every


if __name__ == "__main__":
    main()
