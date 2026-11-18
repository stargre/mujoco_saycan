# MuJoCo SayCan 完整验证教程

本教程从干净环境开始，依次验证依赖、模型资产、感知、IK、抓取、放置、SayCan 规划、可视化和 Qwen API。不要跳过中间层；后层失败时，上一层的报告能直接定位问题。

## 1. 环境准备

推荐 Python 3.10–3.12、64 位系统、支持 OpenGL 的本地桌面或 Linux EGL 环境。

### Windows PowerShell

```powershell
git clone https://github.com/stargre/mujoco_saycan.git
cd mujoco_saycan
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可在当前进程临时允许：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
git clone https://github.com/stargre/mujoco_saycan.git
cd mujoco_saycan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

无桌面的 Linux 使用 `export MUJOCO_GL=egl`；有桌面的 Linux 可使用默认 GLFW。macOS 不要设置 `MUJOCO_GL=egl`。

检查版本：

```powershell
python -c "import mujoco,numpy,openai,PIL; print(mujoco.__version__, numpy.__version__, openai.__version__, PIL.__version__)"
```

## 2. 安装 Franka Panda 资产

仓库不重复提交 MuJoCo Menagerie 的网格资产。运行：

```powershell
python scripts/setup_assets.py
```

脚本通过 Git sparse checkout 只检出 `google-deepmind/mujoco_menagerie` 的
`franka_emika_panda` 目录，不使用容易触发匿名限流的 GitHub Contents API，
并复制上游许可证。成功后应存在：

```text
assets/franka_emika_panda/scene.xml
assets/franka_emika_panda/panda.xml
assets/franka_emika_panda/assets/...
assets/MUJOCO_MENAGERIE_LICENSE
```

如果已经有 Menagerie，可直接指定：

```powershell
$env:PANDA_MODEL_DIR="D:\models\mujoco_menagerie\franka_emika_panda"
```

Linux 使用 `export PANDA_MODEL_DIR=/path/to/.../franka_emika_panda`。

## 3. 纯逻辑与 API 契约测试

```powershell
python -m unittest discover -s tests -v
```

这些测试不调用真实 Qwen，也不消耗 Token。它们验证信念快照、状态化候选、中文任务、Keyword 目标区分，以及 Qwen 必须覆盖全部候选且分数位于 `[0,1]`。成功标志是最后一行 `OK`。

## 4. MuJoCo 场景与 RGB-D 感知

```powershell
python scripts/smoke_test.py
```

成功标志：输出同时包含 `red_cube`、`blue_cube` 和 `SMOKE TEST PASSED`。

只生成场景可视化：

```powershell
python scripts/visualize_scene.py
```

检查 `outputs/visualization/` 中的 PNG、GIF 和 `visualization_report.json`。`robot_table_contact_samples` 应为 0。

## 5. TCP 与 IK

```powershell
python scripts/test_ik_reach.py
```

主要判据：IK 求解不改变实时关节状态；执行后 TCP 误差不超过 1 cm；关节跟踪误差不超过 0.02 rad；最后输出 `TCP REACH TEST PASSED`。

## 6. 物理抓取

```powershell
python scripts/test_grasp.py --target red_cube
```

结果在 `outputs/grasp_test/`。成功需要双指都接触红方块、方块抬升至少 6 cm、方块脱离桌面，并生成两视角 GIF 和 `grasp_report.json`。

## 7. 放置闭环

```powershell
python scripts/test_place.py
```

`outputs/place_test/place_report.json` 的 `success` 应为 `true`，并满足：

- PickUp、PlaceNear、Done 均成功；
- 红方块最终在桌面上且不再接触手指；
- 机械臂不接触桌面；
- 红方块到规划目标的 XY 误差不超过 4 cm；
- 红蓝方块中心距为 5–13 cm；
- 静置阶段位移不超过 1 cm。

## 8. 完整 SayCan 与可视化

先跑不需要网络的基线：

```powershell
python scripts/run_saycan.py `
  --scorer keyword `
  --task "把红色方块拿起来，放到蓝色方块旁边" `
  --visualize record `
  --cameras overhead,side_view `
  --output outputs/keyword_run
```

Linux/macOS 把 PowerShell 的反引号换成反斜杠，或写成一行。

成功时 `result.json` 的 `success` 为 `true`、`status` 为 `task_complete`。`history` 通常依次包含 `PickUp(red_cube)`、`PlaceNear(red_cube, blue_cube)`、`Done()`。每轮：

- `say` 是语言相关性；
- `can` 是物理可行性；
- `score = say × can`；
- `selected_candidate_id` 是本轮执行项。

用浏览器打开 `report.html`，可同时查看动画和每轮分数表。本地图形桌面可用 `--visualize both` 同时打开原生 viewer；服务器无显示器只用 `record`。

## 9. 配置与验证 Qwen API

本项目通过阿里云百炼 DashScope 的 OpenAI 兼容接口调用 Qwen。程序优先读取 `DASHSCOPE_API_KEY`，也接受 `QWEN_API_KEY` 别名；不会输出密钥。

### Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY="sk-你的真实密钥"
$env:QWEN_MODEL="qwen-plus"
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python scripts/check_qwen_api.py
```

### Linux/macOS

```bash
export DASHSCOPE_API_KEY="sk-你的真实密钥"
export QWEN_MODEL="qwen-plus"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python scripts/check_qwen_api.py
```

`check_qwen_api.py` 只发送一个很小的候选打分请求。成功标志为 `"success": true` 和 `QWEN API CHECK PASSED`。账户可用模型不同就修改 `QWEN_MODEL`。可选网络参数：

```powershell
$env:QWEN_TIMEOUT_SECONDS="30"
$env:QWEN_MAX_RETRIES="2"
```

## 10. 用 Qwen 驱动 SayCan

保持同一个已配置密钥的终端：

```powershell
python scripts/run_saycan.py `
  --scorer qwen `
  --task "把红色方块拿起来，放到蓝色方块旁边" `
  --visualize record `
  --cameras overhead,side_view `
  --output outputs/qwen_run
```

验证 `outputs/qwen_run/result.json`：

- `success` 为 `true`，最终 `status` 为 `task_complete`；
- `run_metadata.scorer` 为 `qwen`；
- `run_metadata.qwen_request_count` 大于 0；
- `run_metadata.qwen_model` 与配置一致；
- 每轮 Qwen 都对全部候选给出分数。

Qwen 模式遇到缺少密钥、HTTP 错误或不完整 JSON 会明确失败，不会静默换成 Keyword。因此元数据能区分真正的 Qwen 实验和离线基线。

## 11. 数据集批量评测

单例 PlaceNear 是物理验收，不是数据集结论。完整离线评测运行 12 个 JSONL
episode，覆盖 Pick/Place、红/蓝两种源物体、4 种布局以及中英文：

```bash
python scripts/evaluate_dataset.py --scorer keyword
```

输出：

- `outputs/benchmark/summary.json`：总成功率及按语言、任务类型分组的指标；
- `outputs/benchmark/results.csv`：逐 episode 结果；
- `outputs/benchmark/episodes/*.json`：完整候选分数、技能序列和物理结果。

Qwen 批量评测会产生多次付费调用，先验证两个 episode：

```bash
python scripts/evaluate_dataset.py --scorer qwen --max-episodes 2
```

该 JSONL 是本仓库的 acceptance benchmark，不是训练数据，也不是 Google
真实机器人数据。当前实现的 Can 是确定性 IK/接触可行性；若要复刻学习型 Can，
还需要另行采集技能 rollout 数据、训练 value/affordance 模型并做 held-out 评测。

## 12. 常见故障

### `Franka Panda MJCF was not found`

运行 `python scripts/setup_assets.py`，或检查 `PANDA_MODEL_DIR` 是否直接指向含 `scene.xml` 的 `franka_emika_panda` 目录。

### Windows 报 OpenGL / GLFW 错误

不要设置 `MUJOCO_GL=egl`；在本地图形会话运行。远程桌面缺少 OpenGL 时可换有硬件加速的会话，或在 Linux 服务器使用 EGL。

### Linux `libGL.so` / EGL 初始化失败

安装显卡驱动和 EGL 运行库，确认 `export MUJOCO_GL=egl`。纯逻辑测试不加载渲染器，可先运行第 3 节。

### `401` / `invalid_api_key`

确认密钥来自阿里云百炼/DashScope，并在启动 Python 的同一个终端中设置。不要把引号本身复制进密钥内容。

### `model_not_found` / `AccessDenied`

账户没有 `QWEN_MODEL` 指定模型的权限。改为控制台中已开通的模型，再运行 `check_qwen_api.py`。

### Qwen 返回 JSON 契约错误

保存完整报错和模型名。代码要求每个候选 ID 恰好出现一次、`relevance` 是 `[0,1]` 的有限数值；这是实验有效性约束，不建议绕过。

### 抓取或放置偶发失败

按第 4–7 节逐层重跑并查看对应 JSON/GIF。不要先调 Qwen：语言分数不会修复相机、IK、双指接触或放置标定问题。

## 13. 最小验收清单

1. 单元测试 → `OK`；
2. smoke test → 两个方块均检测到；
3. IK test → `TCP REACH TEST PASSED`；
4. `grasp_report.json` → `success: true`；
5. `place_report.json` → `success: true`；
6. Keyword `result.json` → `task_complete`；
7. Qwen API check → `QWEN API CHECK PASSED`；
8. Qwen `result.json` → `task_complete` 且请求数大于 0；
9. 保存 `report.html`、GIF、JSON 和 Python/MuJoCo 版本。
10. `evaluate_dataset.py --scorer keyword` → 全部 episode 通过。

