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

The tests check camera localization, TCP forward kinematics, IK state preservation, physical trajectory tracking, pad-midpoint calibration and robot-table contacts.

## Stage 2: Isolated physical grasp

Do not start with the full SayCan loop. First validate the PickUp skill independently:

~~~bash
python scripts/test_grasp.py --target red_cube
~~~

This executes pre-grasp, vertical approach, bilateral finger contact, and a 12 cm lift. The script saves stage-by-stage PNGs, overhead/side GIFs and `outputs/grasp_test/grasp_report.json`.

A grasp is accepted only when both `left_finger` and `right_finger` contact the target, the cube rises at least 6 cm, and it is no longer in contact with the tabletop. The cube's MuJoCo body pose is used only as a simulator-side test oracle; it is not fed into SayCan candidate scoring.

The grasp controller holds the Home hand orientation during approach and lift. The current scene is calibrated for a top-down approach with the finger closing axis approximately aligned to world X.

## Stage 3: Planner state tests

~~~bash
python -m unittest tests/test_planner_state.py -v
~~~

## Stage 4: Full SayCan task (only after Stage 2 passes)

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

This is a two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. IK supports a local 6D pose target but there is not yet a global collision-aware motion planner. Camera localization has centimetre-scale error in the current calibration and needs further refinement. Robust grasping must be validated on the target machine before claiming end-to-end task success.
