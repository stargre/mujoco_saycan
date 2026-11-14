# MuJoCo SayCan: Franka Panda Manipulation

A modular SayCan-style reproduction for a Franka Emika Panda arm in MuJoCo. It preserves the MiniGrid project's core design—semantic skill candidates, independent Say/Can scores, a parameterized skill library, observation-grounded state, closed-loop execution, and replanning—while replacing grid navigation with continuous-space robot control.

The project uses the official MuJoCo Menagerie Franka Panda model. It does not vendor the upstream robot meshes; point PANDA_MODEL_DIR at your existing model directory or run scripts/bootstrap_menagerie.sh.

## Quick start

~~~bash
conda activate mojoco
cd /mnt/data2/zhiyu/embodied-lab
git clone https://github.com/stargre/mujoco_saycan.git
cd mujoco_saycan
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export PANDA_MODEL_DIR=/mnt/data2/zhiyu/embodied-lab/mujoco/05_franka_panda/mujoco_menagerie/franka_emika_panda
export MUJOCO_GL=egl
python scripts/smoke_test.py
~~~

Run the Qwen-backed SayCan planner:

~~~bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
export QWEN_MODEL="qwen-plus"
python scripts/run_saycan.py --scorer qwen --task "Pick up the red cube and place it near the blue cube" --save-gif
~~~

Offline baseline (not an LLM experiment):

~~~bash
python scripts/run_saycan.py --scorer keyword --task "Pick up the red cube and place it near the blue cube" --save-gif
~~~

See the full README below for architecture, supported tasks, tests, and troubleshooting.
