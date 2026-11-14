# MuJoCo SayCan — Franka Panda

This is a first integration baseline that ports the MiniGrid SayCan architecture to a continuous-space Franka Panda manipulation scene in MuJoCo. It preserves semantic skill selection and Say × Can scoring, with RGB-D segmentation/depth localization, Jacobian IK, joint trajectory control, gripper control, and replanning.

## Setup

Use the MuJoCo Menagerie Franka Emika Panda model (the directory containing scene.xml and assets/). It is not vendored here.

~~~bash
conda create -n mujoco-saycan python=3.11 -y
conda activate mujoco-saycan
git clone https://github.com/stargre/mujoco_saycan.git
cd mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/path/to/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
python scripts/smoke_test.py
~~~

## Run

Qwen-backed planner:
~~~bash
export DASHSCOPE_API_KEY=YOUR_KEY
export QWEN_MODEL=qwen-plus
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube"
~~~

Offline smoke baseline (not an LLM experiment):
~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube"
~~~

## Architecture

- mujoco_env.py: scene composition and MuJoCo stepping/rendering.
- perception.py: camera segmentation and depth back-projection.
- robot.py: Panda joint control, damped-least-squares IK and gripper.
- context.py: observation-derived belief and robot context.
- skills.py: Explore, Find, NavigateTo, Reach, PickUp, PlaceNear and Done.
- saycan.py: Qwen relevance and SayCan selection loop.

The planner uses camera-derived object locations and robot proprioception. MuJoCo ground truth is not intentionally passed to the semantic planner. Contact checks use the contact manifold for grasp verification.

## Scope and limitations

This is an initial reproducible integration, not a verified reproduction of Google's full real-robot system. It currently targets the included red/blue cube tabletop scene and a fixed overhead camera. IK is position-only damped least squares; there is no global collision-aware motion planner, robust grasp controller, learned affordance model, or broad task benchmark yet. Placement currently marks release completion after the release sequence; add an independent post-release pose oracle before using results as research claims. Run the smoke test on the target host and inspect the robot trajectory before trusting full task success.
