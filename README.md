# MuJoCo SayCan — Franka Panda

An evolving integration of the MiniGrid SayCan architecture in a continuous-space Franka Panda tabletop manipulation scene.

## Setup

Use the MuJoCo Menagerie Franka Emika Panda directory containing `scene.xml`, `panda.xml` and `assets/`.

~~~bash
conda activate mojoco
cd /mnt/data2/zhiyu/embodied-lab/mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/mnt/data2/zhiyu/embodied-lab/mujoco/05_franka_panda/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
~~~

## Stage 1: model and camera smoke test

~~~bash
python scripts/smoke_test.py
~~~

## Stage 2: camera localization and physical IK test

~~~bash
python scripts/test_ik_reach.py
~~~

This compares camera estimates with simulator ground truth for evaluation only, verifies that IK does not teleport the robot, and executes a real joint trajectory to a pre-grasp waypoint. Ground truth is not provided to the planner.

## Stage 3: SayCan task

First use the keyword baseline to debug the skill pipeline:

~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube"
~~~

Then test Qwen semantic relevance:

~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
export QWEN_MODEL=qwen-plus
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube"
~~~

## Important implementation notes

- SayCan now transitions to `Done` after a verified placement, instead of repeating `PickUp` and `PlaceNear`.
- IK is a pure computation: it restores the live qpos before the joint trajectory is executed.
- A grasp is accepted only when the target cube contacts a finger geom after lifting; cube-table contact is not accepted.
- Placement is accepted only after the released cube is observed near the reference cube and is no longer in finger contact.

## Current limitations

This remains a limited two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. IK is position-only and does not perform global collision-aware planning. The fixed camera and hand-frame waypoint offsets need calibration on the target model. Grasp friction, contact stability, and placement tolerance must be validated through the physical trajectory tests before using success rates as research evidence.
