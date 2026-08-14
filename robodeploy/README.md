# robodeploy

Robotics deployment & data collection toolkit — based on LeRobot, adapted for real-world robot control and policy inference.

---

## Features

- **Multi-robot support** — S1 (single/bimanual), Innov Arm, ARX X5
- **Bimanual teleoperation** — single-arm and dual-arm leader-follower setups
- **Multi-mode data collection** — teleop, policy inference, mixed (hot-switch with P key)
- **Camera integration** — OpenCV USB cameras, Intel RealSense depth cameras
- **Policy inference** — OpenPI WebSocket client with temporal smoothing and DAgger alignment
- **WebUI** — FastAPI WebSocket real-time monitoring and control dashboard

## Project Layout

```
robodeploy/
├── src/robodeploy/         # core package: cameras, datasets, motors, robots, teleoperators, webui, scripts
├── scripts/                   # hardware deploy/test tools (replay, inference check, RTC smoke test)
├── examples/                  # usage examples
└── pyproject.toml             # project metadata & dependencies
```

## Quick Start

### Install

```bash
pip install -e ".[dev,test]"           # base + dev/test
pip install -e ".[feetech]"            # Feetech servo driver
pip install -e ".[dynamixel]"          # Dynamixel servo driver
pip install -e ".[intelrealsense]"     # Intel RealSense camera driver
pip install -e ".[innov]"              # Innov Arm (mujoco renderer)
pip install -e ".[qt]"                 # Qt desktop UI (PyQt6)
pip install -e ".[all]"                # everything
```

### Lint & format

```bash
ruff check src/
ruff format src/
```

(The `tests/` suite has been removed; dev/test deps remain in `pyproject.toml` for future use.)

### Start WebUI

```bash
python -m robodeploy.webui           # defaults to http://localhost:5000
```

### Record data

Two recording script variants, both using the NPY storage backend (O(1) RAM) with a 30fps control loop:

| | Leader-follower `record_dataset.py` | Body-teaching `record_body_teaching.py` |
|---|---|---|
| Target robots | S1 and other leader-follower setups | Innov Arm / ARX X5 (body teaching) |
| Demonstration source | Separate teleoperator (`--teleop.type`, leader arms) | No teleop — robot provides `get_action()`; hand-guiding under gravity compensation with `--robot.mode=collect` |
| Recorded action | teleop mode: leader target joint positions; policy mode: inferred actions | collect mode: the arm's own current positions (produced by hand-guiding, state/action from the same arm); policy mode: inferred actions |
| Control modes | `teleop` / `policy` / `mixed` | `collect` / `policy` / `mixed` |
| P-key switch behavior | On policy→teleop, `interpolate_leader_to_follower()` cosine-blends the leader arms onto follower positions for a smooth handoff | Also switches hardware mode: `set_mode("collect")` enables gravity compensation / `set_mode("control")` position control |
| Inference input preprocessing | Gripper binarization (joint indices 6/13, <0.2→0); `front_1` rotated 180° before vertical stacking with `front` | No binarization; `front_1` stacked without rotation |
| Front-ends | Keyboard / WebUI / Qt (`--front_end=qt`) | Keyboard / WebUI |

Both variants additionally write `is_infer_data` (whether the frame's action came from policy inference) and `is_failure_data` (set by the success/failure label at save time) into every recorded frame.

```bash
# Leader-follower: bimanual S1 — pure teleoperation
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --teleop.type=bi_s1_leader \
    --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --control_mode teleop \
    --task="pick and place"

# Leader-follower: mixed mode — policy inference + teleop (P key to toggle)
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --teleop.type=bi_s1_leader \
    --policy.type=openpi \
    --policy.host=localhost --policy.port=8000 \
    --task="pick and place"

# Body-teaching: bimanual Innov Arm (mode=collect, no leader arms needed)
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=bi_innov_arm_v1 \
    --robot.left_port=/dev/ttyACM0 --robot.right_port=/dev/ttyACM1 \
    --robot.mode=collect \
    --task="pick and place"

# Body-teaching: policy inference deployment (mode=control), supports --use_rtc
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=innov_arm_v1 --robot.port=/dev/ttyACM0 --robot.mode=control \
    --policy.type=openpi --policy.host=localhost --policy.port=8000 \
    --task="pick and place"
```

#### Keyboard controls (same for both variants)

Controls have two equivalent entry points: **terminal keys** or **front-end buttons** (WebUI / Qt — both go through the same command channel, executed synchronously by the main loop):

| Key | WebUI/Qt button | Action |
|---|---|---|
| `Enter` | — | Start the control loop |
| `R` | Record toggle | Start / stop recording the current episode |
| `S` | Save (with label) | Finish and save the episode, then label it: `1` success / `0` failure / `2` discard |
| `P` or `Tab` | Mode switch | Hot-switch demonstration ↔ inference in mixed mode (DAgger-style correction; no-op in fixed modes) |
| `Z` | Zero reset | Move arms to zero position (unavailable while recording) |
| `Esc` / `Ctrl+C` | Stop | Exit |

When an episode reaches `--episode_time_s`, recording stops automatically and the success/failure label prompt appears. Note the body-teaching variant supports keyboard + WebUI only — no Qt front-end.

See [scripts/README.md](scripts/README.md) for details.

## Tech Stack

| Component | Stack |
|-----------|-------|
| Language | Python 3.10+ |
| ML Backend | PyTorch, HuggingFace datasets & hub |
| Vision | OpenCV, Intel RealSense |
| Config | draccus (dataclass-based CLI) |
| WebUI | FastAPI / uvicorn |
| Video | PyAV (av) |
| Serial | pyserial, Dynamixel SDK, Feetech SDK |
| Policy | OpenPI WebSocket client, msgpack |

## Conventions

- **Double quotes** for all Python strings (ruff-enforced)
- **110-char line width**
- **Google-style docstrings**
- **Apache 2.0 header** on every `.py` file
- **`src` layout** — import from `from robodeploy import ...`

## Notes

- Hardware tests auto-skip in `conftest.py` when no physical motor/camera is connected
- `torchcodec` is unavailable on Windows, ARM Linux, and macOS x86_64 — this is expected
- `[feetech]`, `[dynamixel]`, `[intelrealsense]` are optional extras, not in the base install
- Top-level `scripts/` is **not** part of the Python package — standalone hardware deploy/test entry points only. Data collection & dataset processing scripts live in `src/robodeploy/scripts/` (invoked via `python -m robodeploy.scripts.<module>`)

## License

Apache 2.0 — derived from [LeRobot](https://github.com/huggingface/lerobot).
