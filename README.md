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

~~~bash
python scripts/test_place.py
~~~

Outputs are saved to \`outputs/place_test/\`:
- \`place_overhead.gif\` and \`place_side_view.gif\`
- Stage-by-stage PNG images
- \`place_report.json\`

The placement target is derived from the RGB-D belief for the blue cube. The release height uses the known table and cube geometry. The TCP-to-object offset is an empirical calibration for this fixed scene and grasp orientation; recalibrate it if the cube dimensions, gripper orientation, or grasp strategy changes.

## Stage 4: Planner state tests

~~~bash
python -m unittest tests/test_planner_state.py -v
~~~

## Stage 5: End-to-end SayCan task

Keyword baseline:

~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube"
~~~

Qwen:

~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
export QWEN_MODEL=qwen-plus
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube"
~~~

The output is saved to \`outputs/last_run.json\`.

## Stage 6: Multi-candidate SayCan planning

The planner now exposes multiple candidates at each relevant state. For the initial two-cube scene, candidates include \`PickUp(red_cube)\`, \`PickUp(blue_cube)\`, and \`Done()\`. Qwen scores semantic relevance (Say), while Can independently evaluates gripper/object state and IK reachability for the candidate approach and manipulation poses. The selected action maximizes \`Say × Can\`.

~~~bash
python -m unittest tests/test_planner_state.py -v

python scripts/run_saycan.py \
  --scorer keyword \
  --task "Pick up the red cube and place it near the blue cube"

python scripts/run_saycan.py \
  --scorer qwen \
  --task "Pick up the red cube and place it near the blue cube"
~~~

Every round in \`outputs/last_run.json\` includes:
- \`belief_before\` and \`belief_after\` immutable snapshots
- \`candidate_scores\`: candidate ID, arguments, Say, Can, combined score and rationale
- \`selected_candidate_id\` and selected action

Can's current motion feasibility check is IK reachability for pre-grasp/grasp or pre-place/place poses. It is not a collision-aware path planner; that remains a limitation. The current \`Find\` primitive only refreshes the camera and does not yet perform active viewpoint search, so it is not marked feasible by Can.

## Current limitations

This is a two-cube tabletop prototype, not a verified full reproduction of Google's real-robot system. The scene-specific grasp transform and fixed top-down orientation require recalibration if object geometry or grasp strategy changes. Camera localization is centimetre-scale. There is no global collision-aware motion planner or general active-search policy yet. Dynamic recovery after a physically failed skill is the next development stage.
