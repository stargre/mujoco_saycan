# MuJoCo SayCan — Franka Panda 复刻

这是一个可运行、可验证、可视化的 SayCan 桌面操作复刻。Qwen 负责 **Say**（下一项技能与自然语言任务的相关性），确定性状态检查和 IK 负责 **Can**（物理可行性），最终选择 `Say × Can` 得分最高的技能，并在 MuJoCo 中闭环执行。

当前验收任务：

> 拿起红色方块，把它放到蓝色方块旁边。

系统使用 Franka Panda、RGB-D/分割感知、TCP 位姿 IK、双指接触与抬升验证、放置后视觉/接触后置条件验证。支持英文和上述中文任务表达。

## 架构

```text
任务文本
  └─ build_candidates（基于信念状态生成技能候选）
      ├─ Say：Keyword 或 Qwen 语义相关性
      ├─ Can：观测、夹爪状态、接触、IK 可达性
      └─ argmax(Say × Can)
           └─ PickUp / PlaceNear / Done
                └─ MuJoCo → RGB-D 更新 → 重新规划
```

Qwen **不控制关节，也不判断碰撞或 IK**；它只给候选技能打语义分。物理动作始终由确定性技能控制器执行。

## 5 分钟开始

推荐 Python 3.10–3.12。

Linux 服务器（不要使用 PowerShell 的 `Activate.ps1`）：

```bash
git clone https://github.com/stargre/mujoco_saycan.git
cd mujoco_saycan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_assets.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_assets.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

如果服务器已经有 `mujoco_menagerie/franka_emika_panda`，无需下载：

```bash
export PANDA_MODEL_DIR=/path/to/mujoco_menagerie/franka_emika_panda
```

Linux 无显示器运行前通常需要 `export MUJOCO_GL=egl`；Windows 本地图形环境保持默认即可。

运行离线关键词基线，并生成 GIF、HTML 和 JSON：

```powershell
python scripts/run_saycan.py --scorer keyword --visualize record
```

结果位于 `outputs/saycan_run/`：

- `result.json`：每轮候选的 Say、Can、乘积、选择与执行结果；
- `run_overhead.gif`：完整仿真过程；
- `report.html`：带动画和候选分数表的可视化报告；
- `timeline.json`：规划与技能阶段时间线。

要同时录制俯视和侧视角：

```powershell
python scripts/run_saycan.py --scorer keyword --cameras overhead,side_view
```

要打开 MuJoCo 原生窗口：

```powershell
python scripts/run_saycan.py --scorer keyword --visualize both
```

## Qwen API

在 PowerShell 中只对当前终端设置密钥：

```powershell
$env:DASHSCOPE_API_KEY="sk-你的密钥"
$env:QWEN_MODEL="qwen-plus"
python scripts/check_qwen_api.py
python scripts/run_saycan.py --scorer qwen --visualize record `
  --task "把红色方块拿起来，放到蓝色方块旁边"
```

Linux/macOS：

```bash
export DASHSCOPE_API_KEY="sk-你的密钥"
export QWEN_MODEL="qwen-plus"
python scripts/check_qwen_api.py
python scripts/run_saycan.py --scorer qwen --visualize record
```

也可复制 `.env.example` 作为变量清单，但程序不会自动读取 `.env`，以避免无意加载密钥。不要提交 `.env`；它已被 `.gitignore` 忽略。

完整的逐步验收、Qwen 接口说明、成功判据和故障排查见 [docs/VERIFICATION_ZH.md](docs/VERIFICATION_ZH.md)。

## 分层验证

```powershell
python -m unittest discover -s tests -v
python scripts/smoke_test.py
python scripts/test_ik_reach.py
python scripts/test_grasp.py --target red_cube
python scripts/test_place.py
python scripts/run_saycan.py --scorer keyword --visualize record
```

每一层只在上一层通过后继续。`test_place.py` 会验证：抓取和放置成功、方块接触桌面、手指已释放、机械臂未碰桌、目标位置误差和稳定时间均在阈值内。

## 数据集批量评测

单个红方块任务只是物理 smoke test。仓库同时提供
`benchmarks/saycan_tabletop_v1.jsonl`：12 个数据集 episode，覆盖 Pick/Place、
红/蓝源物体、4 种物理布局、中英文和同义改写。

```bash
python scripts/evaluate_dataset.py --scorer keyword
```

输出位于 `outputs/benchmark/summary.json`、`results.csv` 以及每个 episode
的完整规划轨迹。Qwen 会产生 API 调用，建议先跑两个 episode：

```bash
python scripts/evaluate_dataset.py --scorer qwen --max-episodes 2
```

这是仓库内的**评测数据集**，不是技能训练数据。当前 Can 使用确定性 IK、接触和
状态后置条件，没有声称使用原始 SayCan 的学习型 affordance/value-function 数据。

## 目录

- `mujoco_env.py`：Menagerie 场景组合、运行时、渲染和原生 viewer；
- `robot.py`：Panda TCP、DLS IK、关节轨迹和夹爪控制；
- `perception.py`：从分割图和深度图重建物体位置；
- `skills.py`：PickUp / PlaceNear / Done 及物理后置条件；
- `saycan.py`：候选生成、Keyword/Qwen 打分、Can 与闭环规划；
- `visualization.py`：运行过程 GIF、PNG、时间线和 HTML 报告；
- `scripts/setup_assets.py`：只下载 MuJoCo Menagerie 的 Panda 资产；
- `benchmarks/`：JSONL 多任务、多布局评测集及数据卡；
- `scripts/evaluate_dataset.py`：批量运行数据集并生成成功率/CSV/逐例轨迹；
- `tests/`：规划状态、中文任务与 Qwen JSON 契约测试。

## 适用边界

这是双方块桌面场景的研究复刻，不是 Google 真实机器人系统的完整复制。当前 Can 是接触/状态检查加 IK 可达性，不是训练出的 value function，也不是全局避障运动规划器；抓取变换针对当前夹爪姿态和 5 cm 方块标定。更换物体尺寸、相机或抓取姿态后需要重新标定并重新跑分层验收。

