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

## Stage 1: TCP calibration and reach

The controlled end-effector point is the midpoint of the two main fingertip pads, not the wrist/hand body origin. For the Menagerie Panda MJCF this point is 0.1029 m along the hand-local +Z axis.

~~~bash
python scripts/test_ik_reach.py
python scripts/diagnose_manipulation.py
python scripts/visualize_scene.py
~~~

## Stage 2: Isolated physical grasp

~~~bash
python scripts/test_grasp.py --target red_cube
~~~

A grasp is accepted only when both fingers contact the target, the cube rises at least 6 cm, and it is no longer in contact with the tabletop.

## Stage 3: PickUp -> PlaceNear -> Done

Run the isolated placement integration test. It performs a fresh grasp, transports the cube while maintaining the validated top-down grasp orientation, releases it beside the blue cube, retreats, and checks the physical/perceptual postconditions.

~~~bash
python scripts/test_place.py
~~~

Outputs are saved to `outputs/place_test/`:
- `place_overhead.gif` and `place_side_view.gif`
- Stage-by-stage PNG images
- `place_report.json`

The placement target is derived from the RGB-D belief for the blue cube. The release height uses the known table and cube geometry. The current TCP-to-object offset is an empirical calibration for this fixed scene and grasp orientation; recalibrate it if the cube dimensions, gripper orientation, or grasp strategy changes. MuJoCo ground-truth object poses are used only by the test script for final evaluation, not by SayCan planning or the PlaceNear target calculation.

Acceptance requires:
- PickUp, PlaceNear and Done all succeed
- Red cube is in contact with the table and no longer in contact with either finger
- No robot-table collision
- Final red/blue center distance is between 5 cm and 13 cm
- Placement target XY error is at most 4 cm
- Cube height error and post-release settling displacement are each at most 1 cm

## Stage 4: Planner state tests

~~~bash
python -m unittest tests/test_planner_state.py -v
~~~

## Stage 5: Full SayCan task (after Stage 3 passes)

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

This is a two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. IK supports a local 6D pose target but there is not yet a global collision-aware motion planner. Camera localization remains centimetre-scale, and the fixed grasp transform is specific to the current cube and orientation. These limitations must be addressed before transferring the controller to hardware or claiming general manipulation robustness.
