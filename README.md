# MuJoCo SayCan — Franka Panda

A first integration baseline porting the MiniGrid SayCan architecture to a continuous-space Franka Panda tabletop manipulation scene in MuJoCo. It combines semantic skill selection, Say × Can scoring, camera segmentation/depth localization, Jacobian IK, joint trajectory control, gripper control and closed-loop replanning.

## 1. Dependencies

Use the MuJoCo Menagerie Franka Emika Panda directory containing `scene.xml`, `panda.xml` and `assets/`. The robot model is not vendored in this repository.

~~~bash
conda activate mojoco
cd /mnt/data2/zhiyu/embodied-lab/mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/mnt/data2/zhiyu/embodied-lab/mujoco/05_franka_panda/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
~~~

## 2. Smoke test

~~~bash
python scripts/smoke_test.py
~~~

The test loads the Menagerie Panda model, creates a tabletop scene, moves the arm to its home configuration, renders the overhead RGB-D camera, and checks that both cubes are visible.

## 3. Run SayCan

Offline keyword baseline (for debugging only; not an LLM experiment):

~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube"
~~~

Qwen relevance scorer:

~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
export QWEN_MODEL=qwen-plus
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube"
~~~

## 4. Code layout

- `mujoco_env.py`: scene composition and simulation/rendering gateway.
- `perception.py`: segmentation and depth back-projection.
- `robot.py`: Panda joint control, damped-least-squares IK and gripper.
- `context.py`: robot context and observation-derived belief.
- `skills.py`: Explore, Find, NavigateTo, Reach, PickUp, PlaceNear and Done.
- `saycan.py`: semantic relevance and SayCan selection loop.

The planner uses camera-derived object estimates and robot proprioception. Object ground-truth poses are not intentionally supplied to semantic planning.

## 5. Current limitations

This is an initial integration baseline, not a verified full reproduction of Google's real-robot system. It currently targets the included red/blue cube scene and a fixed overhead camera. IK is position-only damped least squares; there is no global collision-aware motion planner, robust grasp controller, learned affordance model, or broad task benchmark yet. The placement routine still needs an independent post-release object-pose check. Validate the smoke test and inspect trajectories on the target host before treating a run as successful experimental evidence.
