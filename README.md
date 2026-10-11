# Webots SLAM Course Project

Two tracks, one project:

## Classroom track (start here)
`webots worlds/slam_classroom.wbt` — one robot, 11 lessons (keys `0`–`9`, `X`).
Dead reckoning → occupancy mapping → scan matching → particle filter (with
kidnapping) → loop closure with a live Gauss-Newton pose graph → sensor-failure
lab (with map decay) → frontier exploration → EKF landmarks (virtual + real
camera capstone) → AMCL localization mode. Full lesson plan and key map:
`README_LESSONS.md`.

## Advanced track
`webots worlds/slam_fleet_world.wbt` — three heterogeneous robots (360° lidar,
sonar ring, 240° lidar) run collaborative SLAM with map sharing, A* frontier
planning, and two learned layers (reactive Q + frontier-strategy Q). The
supervisor runs episodic randomized training; Q-tables persist across resets.

## Tools
| Tool | Use |
|---|---|
| `controllers/grader/grader.py` | auto-grades the 12 lesson exercises from logs; `--root students/` for class mode |
| `controllers/analysis/analysis.py` | ATE RMSE, trajectory/ATE/coverage plots (`--plot --cov`) |
| `controllers/ablation_supervisor/` | one-button lecture suites: `drift`, `noise`, `damage`, `explore`, `ekf`, `backends` |
| `controllers/benchmark_supervisor/` | episodic fleet training benchmark |

## Typical session
```bash
webots worlds/slam_classroom.wbt      # do lessons 0-10
python controllers/grader/grader.py  # score yourself
python controllers/analysis/analysis.py controllers/slam_teacher/pose_log.csv --plot --cov
webots worlds/slam_fleet_world.wbt    # then the multi-robot track
