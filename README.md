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

## Stage 1: TCP reach test

The controlled end-effector point is the midpoint of the two main fingertip pads, not the wrist/hand body origin. For the Menagerie Panda model this is 0.1029 m along the hand-local +Z axis: 0.0584 m from the hand to each finger body plus 0.0445 m to the main fingertip pad center.

~~~bash
python scripts/test_ik_reach.py
python scripts/diagnose_manipulation.py
~~~

The test checks IK state preservation, TCP forward kinematics, actual trajectory tracking, and robot-table contacts. It does not yet attempt to grasp an object.

## Visualization

~~~bash
python scripts/visualize_scene.py
~~~

Outputs go to `outputs/visualization/`: overhead/side PNGs, reach GIFs and a JSON report with TCP and collision measurements.

## Stage 2: planner state tests

~~~bash
python -m unittest tests/test_planner_state.py -v
~~~

## Stage 3: SayCan task (do not run until TCP grasp calibration passes)

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

This is a limited two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. TCP IK is position-only; there is no global collision-aware planner or controlled orientation target yet. Perception calibration, robust grasp stability and placement postcondition checks still need validation before claiming end-to-end task success.
