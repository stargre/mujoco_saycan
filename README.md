# MuJoCo SayCan — Franka Panda

An evolving integration of MiniGrid SayCan in a continuous-space Franka Panda tabletop scene.

## Setup

~~~bash
conda activate mojoco
cd /mnt/data2/zhiyu/embodied-lab/mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/mnt/data2/zhiyu/embodied-lab/mujoco/05_franka_panda/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
~~~

## Visualization and collision audit

~~~bash
python scripts/visualize_scene.py
python scripts/diagnose_manipulation.py
~~~

The tabletop now defaults to x=[0.34, 0.90] m, retaining both cubes while moving its front edge away from the Panda shoulder region. This is a candidate scene-clearance correction; verify the contact report on the target machine. The table can be tuned without editing code:

~~~bash
export SAYCAN_TABLE_X_MIN=0.34
export SAYCAN_TABLE_X_MAX=0.90
export SAYCAN_TABLE_Y_HALF=0.48
export SAYCAN_TABLE_TOP=0.36
export SAYCAN_TABLE_THICKNESS=0.08
~~~

The visualizer marks robot-table contacts in PNG/GIF frames and reports contact locations in `visualization_report.json`. Cube-table support contacts are excluded from this collision warning.

## Other tests

~~~bash
python scripts/smoke_test.py
python -m unittest tests/test_planner_state.py -v
python scripts/test_ik_reach.py
~~~

## SayCan task

Only run the full task after the arm reaches the pre-grasp target without robot-table contacts and the grasp test is validated.

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

This is a limited two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. IK is position-only and not collision-aware. The hand body origin is not the gripper TCP. The table clearance change must be validated on the target model; grasp and placement still require physical contact and postcondition verification.
