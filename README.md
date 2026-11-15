# MuJoCo SayCan — Franka Panda

An evolving integration of the MiniGrid SayCan architecture in a continuous-space Franka Panda tabletop manipulation scene.

## Setup

~~~bash
conda activate mojoco
cd /mnt/data2/zhiyu/embodied-lab/mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/mnt/data2/zhiyu/embodied-lab/mujoco/05_franka_panda/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
~~~

## Tests

Model and RGB-D smoke test:

~~~bash
python scripts/smoke_test.py
~~~

Planner state unit tests:

~~~bash
python -m unittest tests/test_planner_state.py -v
~~~

Camera localization and physical IK reach test:

~~~bash
python scripts/test_ik_reach.py
~~~

Detailed diagnosis of IK joint tracking, robot/table collision contacts, finger geometry and gripper actuator:

~~~bash
python scripts/diagnose_manipulation.py
~~~

Run this diagnostic before trying another full Pick-and-Place episode if the IK reach test fails.

## SayCan task

~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube"
~~~

For Qwen:

~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
export QWEN_MODEL=qwen-plus
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube"
~~~

## Current limitations

This is a limited two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. IK is position-only and does not perform global collision-aware planning. The fixed camera and hand-frame waypoints need calibration. Grasp friction, contact stability, and placement tolerance must be validated through physical trajectory tests before using success rates as research evidence.
