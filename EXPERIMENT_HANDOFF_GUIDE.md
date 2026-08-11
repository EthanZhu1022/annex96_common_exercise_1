# Annex 96 CE1 实验项目交接与运行指南

本文档面向第一次接触本仓库的实验人员，说明项目结构、MAPPO 各版本的作用、实验结果文件，以及当前仓库内所有可用的训练启动方式。

所有命令默认从仓库根目录执行：

```text
annex96_common_exercise_1/
```

除非是在复现历史实验，否则优先使用下面两条主线：

- 固定分组主线：`mappo_grouped_tarmac_hybrid_grouping`
- 动态 Actor 选择主线：`mappo_grouped_tarmac_soft_router`

不要把不同训练月份、训练集长度、reward 权重或 seed 数量的结果直接放在同一个公平对比表中。

## 1. 最快开始方式

### 1.1 推荐环境

- Python 3.10 或 3.11。`pyproject.toml` 要求 Python `>=3.8,<3.12`。
- 不要另外执行 `pip install citylearn`。仓库已经包含本项目使用的 `citylearn/` 源码，安装外部 CityLearn 可能导入错误版本。
- Linux 长时间实验建议安装 `screen` 和 `taskset`；大规模 3F 消融还需要 GNU Parallel。

使用 `uv`：

```bash
uv python install 3.11
uv sync --python 3.11 --frozen
uv pip install -r test_requirements.txt
uv run python -c "import torch, pandas, sklearn, citylearn; print('environment ready')"
```

也可以使用普通虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch, pandas, sklearn, citylearn; print('environment ready')"
```

Windows PowerShell 激活方式：

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch, pandas, sklearn, citylearn; print('environment ready')"
```

W&B 是可选的。需要在线记录时先运行 `wandb login`；只保存在本地时可以设置：

```bash
export WANDB_MODE=disabled
```

Windows：

```powershell
$env:WANDB_MODE = "disabled"
```

### 1.2 推荐的固定分组 3fA 实验

下面的实验使用 Vermont、1 月训练、2 月测试、凝聚聚类和 3fA 特征：battery capacity、mean heating demand、mean non-shiftable load。

```bash
uv run python -m mappo_grouped_tarmac_hybrid_grouping.train \
  --climate VT \
  --n_episodes 500 \
  --train_month 1 \
  --test_month 2 \
  --grouping_feature_month 1 \
  --seed 42 \
  --group_k_candidates 4 5 \
  --cluster_seed 0 \
  --cluster_retries 10 \
  --grouping_method agglomerative \
  --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean \
  --comm_fusion_mode linear \
  --wandb_name handoff_tarmac_3fA_vt_seed42 \
  --save_dir results/handoff_tarmac_3fA_vt_seed42
```

Texas 泛化实验将 heating 换为 cooling，并使用 8 月训练、9 月测试：

```bash
uv run python -m mappo_grouped_tarmac_hybrid_grouping.train \
  --climate TX \
  --n_episodes 500 \
  --train_month 8 \
  --test_month 9 \
  --grouping_feature_month 8 \
  --seed 42 \
  --group_k_candidates 4 5 \
  --cluster_seed 0 \
  --cluster_retries 10 \
  --grouping_method agglomerative \
  --grouping_feature_columns bes_capacity_kwh cooling_mean nsl_mean \
  --comm_fusion_mode linear \
  --wandb_name handoff_tarmac_3fA_tx_seed42 \
  --save_dir results/handoff_tarmac_3fA_tx_seed42
```

### 1.3 推荐的稳定 Soft Router 实验

该实验先读取训练好的固定 3fA Actor，训练 router，然后只微调 Actor 输出 head。默认脚本比较 feed-forward router 和 GRU router，并运行 seeds 42、0、1：

```bash
PYTHON_EXE="$PWD/.venv/bin/python" \
EXPERT_DIR=results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final \
bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

只做最终 feed-forward stable Soft Router：

```bash
SEEDS="42 0 1" VARIANTS="ff" \
PYTHON_EXE="$PWD/.venv/bin/python" \
bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

只做 smoke test：

```bash
SEEDS="42" VARIANTS="ff" \
PYTHON_EXE="$PWD/.venv/bin/python" \
bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

## 2. 项目顶层目录与文件

### 2.1 主要目录

| 路径 | 大致功能 |
|---|---|
| `citylearn/` | 本仓库内置的 CityLearn 环境源码。实验实际导入的是这里，不是外部 pip 包。 |
| `data/datasets/annex96_ce1_vt_neighborhood/` | Vermont 25 栋建筑数据和 schema。 |
| `data/datasets/annex96_ce1_tx_neighborhood/` | Texas 25 栋建筑数据和 schema。 |
| `annex96_rewards/` | CE1 自定义 reward 公式、权重和 reward 构造函数。 |
| `mappo*/` | MAPPO 主体、固定分组、不同通信结构、分组方法和 Soft Router 实现。第 4 节逐个说明。 |
| `mappo_grouping_variants/` | 可复用的特征提取与 K-means、GMM、Agglomerative、Balanced Spectral 分组代码。 |
| `on-policy-main/` | MAPPO/R-MAPPO 的第三方训练后端。当前多个 MAPPO 版本复用其中的 policy、trainer 和 buffer 接口。不要随意修改。 |
| `independent_sac/` | 自编写的独立 SAC 对照。 |
| `rllib_independent_ppo/`, `rllib_sac/` | RLlib 独立智能体 PPO/SAC 基线。 |
| `sb3_independent_ppo/`, `sb3_independent_sac/` | Stable-Baselines3 独立智能体基线。 |
| `scripts/` | 批量实验、multi-seed、screen、两阶段/三阶段 router 的启动脚本。第 7 节列出全部命令。 |
| `tools/` | 汇总实验指标、重新测试 checkpoint、生成和合并负荷跟踪图。 |
| `tests/` | reward、Balanced Spectral、SOC regrouping、CityLearn 环境等自动测试。 |
| `results/` | 每个实验的 checkpoint、配置、指标、曲线和分组结果。通常体积最大。 |
| `wandb/` | W&B 本地缓存与离线记录。 |
| `experiment_metric_summary/` | 从 selected results 汇总出的 CSV 和 Markdown 排名表。 |
| `experiment_queue_logs/` | 批量脚本的 master/stdout/stderr/screen 日志；目录在首次运行脚本后生成。 |
| `target_folder/` | `generate_load_tracking_figures.py` 重新测试模型后生成的单实验时序和图。 |
| `combined_target_figures/` | 多实验负荷跟踪对比图。 |
| `target_folder_temperature_smoke/` | 温度导出 smoke test 结果，不用于正式结论。 |
| `report/` | 第一阶段 LaTeX 报告。 |
| `report2/` | 当前英文第二阶段 LaTeX 报告及其图。 |
| `report2withCN/` | 保留中文翻译的报告副本。 |
| `reference/` | 本项目收集的论文和参考资料。 |
| `notebooks/` | CityLearn quickstart 和交互式教程。 |
| `examples/` | CityLearn 基础示例。 |
| `docs/` | CityLearn 文档构建源文件。 |
| `assets/`, `_ppt_page_images/` | 报告、PPT 和说明文档使用的图片资源或中间图片。 |
| `.venv/`, `.pytest_cache/`, `__pycache__/` | 本地环境和缓存，不是实验数据，不需要交给结果分析人员。 |

### 2.2 根目录文件

| 文件 | 大致功能 |
|---|---|
| `README.md` | Annex 96 CE1 官方任务、数据、月份和评价指标说明。 |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python 版本与 uv 可复现依赖环境。 |
| `requirements.txt` | pip 安装依赖列表。 |
| `test_requirements.txt` | 测试和开发相关依赖。 |
| `setup.py`, `MANIFEST.in` | Python 包安装和打包配置。 |
| `.gitignore` | Git 忽略规则。 |
| `LICENSE`, `CODE_OF_CONDUCT.md` | 许可证和协作规范。 |
| `annex96_reporting.py` | 温度、舒适度和测试时序的共用导出函数。 |
| `training_progress.py` | 训练耗时、ETA 和进度显示。 |
| `sb3_independent_common.py` | SB3 独立智能体实验的共用环境与辅助逻辑。 |
| `selected_result_folders.md` | 推荐用于正式比较的结果目录及不推荐结果说明。 |
| `table_summary.md` | 已整理的实验表格摘要。 |
| `presentation_script_comp9991.md` | 第一阶段汇报讲稿。 |
| `COMP9991_*.pdf`, `COMP9993_*.pdf` | 已导出的报告和演示文稿。 |
| `report.zip`, `report2_overleaf_*.zip` | Overleaf/报告归档，不是训练入口。 |

## 3. 实验数据约定

### 3.1 气候和月份

| 气候 | 训练月 | 测试月 | 3fA 热需求特征 | 舒适范围 |
|---|---:|---:|---|---|
| Vermont (`VT`) | 1 月 | 2 月 | `heating_mean` | 20--24 °C |
| Texas (`TX`) | 8 月 | 9 月 | `cooling_mean` | 22--26 °C |

### 3.2 主要指标

- CV-RMSE：逐时负荷跟踪误差，越低越好。
- absolute NMBE：整个测试期的总体负荷偏差，越接近 0 越好。
- comfort exceedance：超出舒适区间的建筑小时比例，越低越好。
- degree-hours per building-day：每栋建筑每天累计超出舒适区间的温度程度，越低越好。
- `test_reward_sum`：只适合相同 reward 权重的实验。改变 reward 权重后，reward 数值尺度也会改变，不能直接跨设置排名。

### 3.3 当前常用分组设置

- 3fA：`bes_capacity_kwh heating_mean nsl_mean`；Texas 将 `heating_mean` 换成 `cooling_mean`。
- 4F：3fA 加 `hvac_total_kw`。
- 5F：`bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean`。
- 分组方法：`kmeans`、`gmm`、`agglomerative`、`balanced_spectral`。
- TarMAC Hybrid 融合方式：`relu`、`linear`、`gated`。

## 4. MAPPO 版本总览

| 包/入口 | 用途 | 建议状态 |
|---|---|---|
| `mappo.train` | 早期自编写的层次化 MAPPO：cluster Actor、集中 Critic、可选拼接通信。 | 历史实现，用于理解早期结构。 |
| `mappo_standard.train` | 不分组的标准 MAPPO，每栋建筑作为 agent，集中训练、分散执行。 | 标准基线。 |
| `mappo_grouped.train` | K-means 固定分组，每组共享 Actor，集中 Critic。 | 固定分组基线。 |
| `mappo_grouped_comm.train` | 在每组 Actor 中插入 none/CommNet/PowerNet 通信 wrapper。 | 第一版通信消融。 |
| `mappo_grouped_comm_v2.train` | 全体建筑联合通信的 global actor，并在 PPO batch 中保持同一时刻的 agent 对齐。 | 改进通信基线。 |
| `mappo_grouped_comm_weighted.train` | 分别加权同组信息和其他组信息，参数为 alpha/beta。 | 权重通信消融。 |
| `mappo_grouped_dial.train` | MAPPO 适配的 DIAL：训练时连续可微消息，评估时可离散化。 | 通信方法对照。 |
| `mappo_grouped_gat.train` | 在建筑相似度图上使用多头 GAT 通信。该图不是实际电网拓扑。 | 图通信对照。 |
| `mappo_grouped_powernet.train` | 组内 PowerNet 邻居通信，默认 ring topology。 | 局部邻居对照。 |
| `mappo_grouped_powernet_global.train` | 全体建筑上的 PowerNet 通信，默认 full topology。 | 全局邻居对照。 |
| `mappo_grouped_powernet_global_grouping.train` | PowerNet Global 加可选分组方法和可选分组特征。 | 分组/通信联合对照。 |
| `mappo_grouped_tarmac.train` | TarMAC query-key-value 定向注意力通信。 | TarMAC 基线。 |
| `mappo_grouped_tarmac_hybrid.train` | TarMAC Hybrid，加入 local projection、fusion 和 residual。 | 架构消融入口。 |
| `mappo_grouped_tarmac_hybrid_grouping.train` | TarMAC Hybrid 加可选 grouping method、feature columns 和 reward weights。 | **当前固定分组主入口。** |
| `mappo_grouped_tarmac_soc_regrouping.train` | 先运行固定策略收集 SOC，再根据策略行为重新分组并从头训练。 | 两阶段探索。 |
| `mappo_grouped_tarmac_soft_router.train` | 根据每栋建筑当前状态，在多个完整 Actor/Actor head 之间进行概率选择。支持两阶段、三阶段、stable heads 和 GRU router。 | **当前动态选择主入口。** |
| `mappo_grouping_variants.cluster` | 分组特征提取与四种无监督分组方法，不单独训练策略。 | 多个主入口共享。 |

## 5. 所有 MAPPO 文件的作用

每个包中的 `__init__.py` 都用于将目录声明为 Python 包或导出公共对象，本身通常不包含训练流程。下面列出其余文件。很多小型 `env.py` 和 `cluster.py` 只是 re-export，共用真实实现，修改时应先检查文件内容。

### 5.1 `mappo/`

- `__init__.py`：包标记。
- `agent.py`：早期自编写 Gaussian Actor 和 centralized Critic。
- `communication.py`：将每个 cluster 的观测编码成固定长度消息，并将所有消息拼接给 Actor。
- `utils.py`：rollout buffer、日级 KPI、SOC、成本、排放、公平性等共用计算。
- `train.py`：早期自编写 PPO rollout、更新、checkpoint、测试和绘图入口。

### 5.2 `mappo_standard/`

- `__init__.py`：包标记。
- `env.py`：实际的 `CityLearnMAPPOEnv` 适配器；把 25 栋建筑观测整理为 per-agent observation 和 centralized shared observation。
- `train.py`：使用 `on-policy-main` R-MAPPO 后端训练标准 MAPPO，保存 checkpoint、测试月指标和图。

### 5.3 `mappo_grouped/`

- `__init__.py`：包标记。
- `env.py`：re-export `mappo_standard.env`，环境接口相同。
- `cluster.py`：早期 K-means 分组；主要使用 BES capacity 和 HVAC power，选择较平衡的 K=4/5 结果并保存分组文件。
- `train.py`：每个固定组使用一个 policy/Actor，负责训练、测试、checkpoint 和完整指标导出。

### 5.4 `mappo_grouped_comm/`

- `__init__.py`：包标记。
- `env.py`：re-export 标准 CityLearn MAPPO 环境。
- `cluster.py`：第一版 K-means 分组和分组 artifact 导出。
- `actor_wrapper.py`：把通信模块插入 R-MAPPO Actor encoder 与 action head 之间。
- `buffer.py`：PPO mini-batch 中保持同一时刻的组内 agents 在一起，防止通信关系被随机 shuffle 打乱。
- `train.py`：第一版 grouped communication 的统一训练、测试、指标、checkpoint 和 CLI。
- `communication/__init__.py`：导出通信模块。
- `communication/base.py`：所有通信模块的抽象接口，输入输出均为 `(batch, agents, hidden)`。
- `communication/none.py`：不通信的 identity 对照。
- `communication/commnet.py`：对其他 agents 特征做 mean pooling，再通过 MLP/residual 更新。
- `communication/powernet.py`：按 ring、chain 或 full 邻接关系传递邻居信息。
- `communication/factory.py`：根据 `comm_method` 构造 none、commnet 或 powernet。

### 5.5 `mappo_grouped_comm_v2/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 标准环境和早期 grouped clustering。
- `global_actor.py`：把所有 group Actor encoder 输出合并，进行一次全局通信，再交回对应 action head。
- `train.py`：联合训练 global actor，正确组织 advantage 和 joint actor PPO batch；包含 checkpoint、测试和 CLI。

### 5.6 `mappo_grouped_comm_weighted/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `weighted_comm.py`：计算 `alpha * same_group_mean + beta * other_group_mean`，再 residual 更新。
- `global_actor.py`：将 weighted communication 接到 grouped Actor encoders/heads。
- `train.py`：训练和比较不同 alpha/beta 设置，并导出统一指标。

### 5.7 `mappo_grouped_dial/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `dial_comm.py`：简化 DIAL；训练时在连续消息上加噪声并 sigmoid，评估时可变为二值消息。
- `train.py`：DIAL communication 的 MAPPO 训练、测试和 checkpoint。

### 5.8 `mappo_grouped_gat/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `gat_comm.py`：在固定建筑相似度邻接矩阵上执行多头 masked graph attention。
- `train.py`：构建相似度图、保存 adjacency/summary，并训练 GAT communication MAPPO。

### 5.9 `mappo_grouped_powernet/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `train.py`：`mappo_grouped_comm.train` 的轻量 wrapper，固定 `comm_method=powernet`，默认组内 ring 邻居通信。

### 5.10 `mappo_grouped_powernet_global/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `train.py`：所有建筑先编码、全局 PowerNet 通信、再使用 grouped heads；默认 full topology。

### 5.11 `mappo_grouped_powernet_global_grouping/`

- `__init__.py`：包标记。
- `env.py`：re-export 标准环境。
- `cluster.py`：re-export `mappo_grouping_variants.cluster`。
- `train.py`：PowerNet Global 加四种 grouping method、五类 feature set 和显式 feature columns。

### 5.12 `mappo_grouped_tarmac/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `tarmac_comm.py`：TarMAC query/key/value、scaled similarity、soft attention 和 context 更新。
- `train.py`：TarMAC communication 的 grouped MAPPO 训练、测试和 checkpoint。

### 5.13 `mappo_grouped_tarmac_hybrid/`

- `__init__.py`：包标记。
- `env.py`, `cluster.py`：re-export 共用环境和 clustering。
- `hybrid_tarmac_comm.py`：TarMAC attention 加 local projection；提供 `relu`、`linear`、`gated` 三种 fusion，最后 residual 更新。
- `train.py`：固定旧分组下的 Hybrid 架构消融入口。

### 5.14 `mappo_grouped_tarmac_hybrid_grouping/`

- `__init__.py`：包标记。
- `env.py`：re-export 标准环境。
- `cluster.py`：re-export selectable grouping variants。
- `train.py`：当前固定分组主入口；支持四种分组方法、显式 feature columns、三种 Hybrid fusion、五个 reward 权重、训练/测试/checkpoint/指标导出。

### 5.15 `mappo_grouping_variants/`

- `__init__.py`：导出分组接口。
- `cluster.py`：读取 static 和 operational 特征，按月份切片、标准化，并执行 K-means、GMM、Agglomerative 或 Balanced Spectral；保存 assignments、centers 和 summary。

### 5.16 `mappo_grouped_tarmac_soc_regrouping/`

- `__init__.py`：包标记。
- `README.md`：两阶段 SOC regrouping 的原理、特征和输出说明。
- `env.py`：re-export Hybrid grouping 环境。
- `collect_soc.py`：加载固定 3fA checkpoint，确定性运行训练月并导出逐时 SOC trajectory。
- `features.py`：从 SOC trajectory 计算 mean、std、q10、low/high fraction、daily range 等统计量。
- `cluster.py`：构造 `soc6f` 或 `energy4f` 行为特征并重新分组。
- `train.py`：使用 SOC-derived grouping 从头训练新的 TarMAC Hybrid 模型；不会继承原模型权重。

### 5.17 `mappo_grouped_tarmac_soft_router/`

- `__init__.py`：包标记。
- `EXPERIMENTS.md`：stable feed-forward 与 GRU router 的设置和启动示例。
- `env.py`：re-export 标准环境。
- `cluster.py`：re-export selectable grouping，用于产生初始 expert prior。
- `hybrid_tarmac_comm.py`：Soft Router 包内使用的 Hybrid TarMAC communication 实现。
- `soft_router_actor.py`：共享 router、完整 Actor/Actor-head mixture、capacity-aware 输入和可选 GRU 状态。
- `train.py`：legacy、three-stage、pretrained-full-expert 三种训练 schedule；负责 expert freeze/unfreeze、router-only、head-only adaptation、anti-collapse loss、resume checkpoint、测试与 diagnostics。

## 6. 所有 MAPPO Python 启动入口

先用 `--help` 查看当前代码实际支持的参数。下面每个命令都是可运行入口：

```bash
uv run python -m mappo.train --help
uv run python -m mappo_standard.train --help
uv run python -m mappo_grouped.train --help
uv run python -m mappo_grouped_comm.train --help
uv run python -m mappo_grouped_comm_v2.train --help
uv run python -m mappo_grouped_comm_weighted.train --help
uv run python -m mappo_grouped_dial.train --help
uv run python -m mappo_grouped_gat.train --help
uv run python -m mappo_grouped_powernet.train --help
uv run python -m mappo_grouped_powernet_global.train --help
uv run python -m mappo_grouped_powernet_global_grouping.train --help
uv run python -m mappo_grouped_tarmac.train --help
uv run python -m mappo_grouped_tarmac_hybrid.train --help
uv run python -m mappo_grouped_tarmac_hybrid_grouping.train --help
uv run python -m mappo_grouped_tarmac_soc_regrouping.collect_soc --help
uv run python -m mappo_grouped_tarmac_soc_regrouping.train --help
uv run python -m mappo_grouped_tarmac_soft_router.train --help
```

仓库中的非 MAPPO 控制基线也有独立入口：

```bash
uv run python -m independent_sac.train --help
uv run python -m rllib_independent_ppo.train --help
uv run python -m rllib_sac.train --help
uv run python -m sb3_independent_ppo.train --help
uv run python -m sb3_independent_sac.train --help
```

这些基线的训练 budget 和 MAPPO 正式结果不一定相同。使用前应参考 `selected_result_folders.md`，不要仅根据算法名称直接做主结论比较。

通用训练参数主要包括：

- `--climate VT|TX`
- `--n_episodes`
- `--train_month`, `--test_month`
- `--seed`
- `--save_dir`, `--wandb_name`
- `--no_test`：训练后不执行测试月评估。
- `--test_only`：只加载 checkpoint 做测试。
- `--checkpoint_dir`：可以传结果目录或具体 `checkpoint.pt`。
- `--test_save_dir`：将重新测试结果写到新目录，防止覆盖原结果。

只重新测试一个固定分组 checkpoint：

```bash
uv run python -m mappo_grouped_tarmac_hybrid_grouping.train \
  --test_only \
  --climate VT \
  --train_month 1 \
  --test_month 2 \
  --checkpoint_dir results/EXPERIMENT_NAME \
  --test_save_dir results/EXPERIMENT_NAME/retest
```

改变分组方法时只修改：

```text
--grouping_method kmeans
--grouping_method gmm
--grouping_method agglomerative
--grouping_method balanced_spectral
```

改变 Hybrid fusion 时只修改：

```text
--comm_fusion_mode relu
--comm_fusion_mode linear
--comm_fusion_mode gated
```

显式 reward 参数示例：

```text
--weight_nmbe 1.0
--weight_cv_rmse 1.0
--weight_comfort 1.5
--comfort_binary_weight 3.0
--comfort_degree_weight 1.0
```

如果要做严格对照，除目标变量外，应保持 seed、月份、episodes、features、grouping method、fusion、网络尺寸和 reward 权重完全一致。

## 7. `scripts/` 中所有启动脚本

### 7.1 Linux Bash 脚本

| 启动命令 | 用途与注意事项 |
|---|---|
| `bash scripts/run_priority_experiment_queue.sh` | 顺序运行早期优先队列：4 个 Soft Router 5F 设置、TarMAC Hybrid 3F/4F、PowerNet Global 3F/4F。完成目录默认跳过。属于历史队列。 |
| `bash scripts/run_report2_missing_fair_comparison_seeds.sh` | 两批共 8 个公平对比实验：4F seeds 0--3，以及 K-means/GMM 5F seeds 0--1。每批 4 个并行、每个 3 核，并自动进入 detached screen。属于一次性补 seed 脚本。 |
| `bash scripts/run_soft_router_full_expert_stable_and_gru.sh` | 当前 stable full-expert feed-forward router 与 GRU router 对比；默认 seeds 42、0、1，顺序运行并支持 checkpoint resume。 |
| `bash scripts/run_tarmac_3fA_tx_degree_hours_3seeds_parallel.sh` | Texas 3fA、degree weight 1.5、seeds 0--2 并行，每个 2 核。 |
| `bash scripts/run_tarmac_3fA_vt_degree_hours_3seeds_parallel.sh` | Vermont 3fA、degree weight 1.5、seeds 0--2 并行，每个 2 核。 |
| `bash scripts/run_tarmac_3fA_vt_comfort_strict_3seeds_parallel.sh` | 历史 strict-comfort 脚本，实际同时启动 VT 和 TX 共 6 个作业。**当前不建议直接复现**：脚本没有显式传 reward 权重，而当前 `ce1.py` 默认值已经恢复为 0.8/1.3/0.3。若要 strict 设置，请直接使用第 6 节显式 reward 参数。 |
| `bash scripts/run_tarmac_balanced_spectral_5f_3fA_3seeds_parallel.sh` | Balanced Spectral 的 5F 和 3fA，各 seeds 0、1、42，共 6 个并行作业，需要 12 核和 `uv`。 |
| `bash scripts/run_tarmac_hybrid_3f_shortlist.sh` | 3F 组合 B--I，各 seeds 0--2，共 24 个实验；默认最多 14 个并行，需要 GNU Parallel 和足够 CPU。A 是已有基准，因此脚本不再运行 A。 |
| `bash scripts/run_tarmac_hybrid_cooling_3f_tx_aug_sep_sequential.sh` | Texas cooling 3fA seeds 0--2 顺序运行，适合核数较少机器。 |
| `bash scripts/run_tarmac_soc_regrouping_two_stage.sh` | 先收集固定 3fA 模型的 January SOC，再分别训练 `soc6f` 和 `energy4f`，每种 seeds 42、0、1。 |
| `bash scripts/run_three_stage_shared_router.sh` | 从头训练 shared encoder：500 static + 500 router-only + 500 dynamic actor，seed 42。 |
| `bash scripts/run_three_stage_full_expert_router.sh` | 加载已有完整 experts：500 router-only + 500 dynamic actor，seed 42。 |
| `bash scripts/run_twostage_router_followup_queue.sh` | seeds 0--3 的 router-only-500 与 no-capacity freeze-200 后续队列。 |
| `bash scripts/run_twostage_router_stability_queue.sh` | 先补固定 3F/5F seeds，再运行多种 freeze、temperature、prior、capacity-aware 两阶段 router 稳定性对照。历史消融队列。 |

### 7.2 Linux detached `screen` 启动器

| 启动命令 | 用途 |
|---|---|
| `bash scripts/start_tarmac_3fA_tx_degree_hours_screen.sh` | 在 detached screen 中启动 Texas degree-hours 三 seed。 |
| `bash scripts/start_tarmac_3fA_vt_degree_hours_screen.sh` | 在 detached screen 中启动 Vermont degree-hours 三 seed。 |
| `bash scripts/start_tarmac_3fA_vt_tx_degree_hours_screens.sh` | 同时启动 VT 与 TX 两个 screen，共需要 12 个可用 CPU 核。 |
| `bash scripts/start_tarmac_balanced_spectral_5f_3fA_3seeds_screen.sh` | 在 detached screen 中启动 6 个 Balanced Spectral 作业。 |

`run_report2_missing_fair_comparison_seeds.sh` 自己会创建 screen，不需要再套一层 screen。

常用 screen 命令：

```bash
screen -ls
screen -r SESSION_NAME
```

在 screen 内按 `Ctrl-a`，再按 `d`，可以退出界面但不停止实验。

### 7.3 Windows PowerShell 脚本

| 启动命令 | 用途 |
|---|---|
| `powershell -ExecutionPolicy Bypass -File scripts/run_priority_experiment_queue.ps1` | `run_priority_experiment_queue.sh` 的 Windows 版本。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_tarmac_soc_regrouping_two_stage.ps1` | SOC 两阶段 regrouping 的 Windows 版本。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_shared_router.ps1` | shared encoder 三阶段 router。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_full_expert_router.ps1` | pretrained full-expert 三阶段 router。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_router_comparison.ps1` | 顺序运行 shared 和 full-expert 两条三阶段路线。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_twostage_router_followup_queue.ps1` | 两阶段 router follow-up 队列。 |
| `powershell -ExecutionPolicy Bypass -File scripts/run_twostage_router_stability_queue.ps1` | 两阶段 router stability 队列。 |

PowerShell 脚本选择 Python 的方式：

```powershell
$env:PYTHON_EXE = (Resolve-Path .\.venv\Scripts\python.exe)
powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_full_expert_router.ps1
```

仓库其他位置也包含 shell/batch 文件，但不是 CE1 训练入口：

- `on-policy-main/onpolicy/scripts/` 下的 MPE、SMAC、Hanabi 和 football 脚本属于上游 MAPPO 仓库示例，使用其他环境，不能直接用于本项目的 CityLearn CE1 数据。
- `on-policy-main/onpolicy/envs/hanabi/clean_all.sh` 是上游 Hanabi 清理脚本。
- `tests/scripts/tacc_job.sh` 和其他 `tests/scripts/` 文件用于 CityLearn 测试/兼容性检查，不是正式 CE1 模型训练。
- `docs/make.bat` 用于构建 CityLearn 文档，不训练控制器。

### 7.4 常用脚本环境变量

| 变量 | 作用 |
|---|---|
| `PYTHON_EXE` | 指定 Python 解释器；运行 bash/PowerShell 队列前最重要。 |
| `EPISODES` | 覆盖部分脚本的训练 episode 数。 |
| `SEEDS` | stable Soft Router 脚本的 seed 列表，例如 `SEEDS="42 0 1"`。 |
| `VARIANTS` | stable Soft Router 选择 `ff`、`gru` 或两者。 |
| `EXPERT_DIR` | stable Soft Router 使用的固定 expert checkpoint 目录。 |
| `USE_GPU` | `0` 时隐藏 GPU，`1` 时允许 GPU；许多实验主要受 CityLearn CPU 模拟限制。 |
| `THREADS_PER_JOB` | 每个并行作业分配的 BLAS/CPU 线程数。部分脚本强制必须为 2 或 3。 |
| `CPU_OFFSET` | 多个队列共用机器时，从第几个允许 CPU 开始分配。 |
| `MAX_JOBS` | 3F shortlist 的最大并行数量。 |
| `FORCE=1` | 忽略已完成标志并重新运行。使用前先确认不会覆盖需要保留的结果。 |
| `NO_SKIP_COMPLETED=1` | 历史 queue 中不跳过已有 `latest_metrics.json`。 |
| `SESSION_NAME` / `SCREEN_NAME` | 自定义 screen 名称。 |
| `UV_EXE` | 指定 `uv` 可执行文件。 |

## 8. 结果目录如何判断和使用

一次完整训练通常在 `results/EXPERIMENT_NAME/` 中生成：

| 文件 | 用途 |
|---|---|
| `run_config.json` | 训练配置。做公平对比前先比较该文件。 |
| `checkpoint.pt` | Actor、critic、optimizer 和模型配置 checkpoint。 |
| `latest_metrics.json` | 训练结束和测试结果摘要。队列通常用它判断实验是否完成。 |
| `test_metrics.json`, `test_metrics.csv` | 测试月主要指标。 |
| `test_daily_metrics.csv`, `test_daily_metrics.png` | 每天的负荷跟踪指标和图。 |
| `test_daily_secondary_*` | 成本、排放、ramping 等次要指标。 |
| `test_building_comfort_metrics.csv` | 每栋建筑舒适度指标。 |
| `test_building_temperatures_full.*` | 每栋建筑完整温度和舒适区间时序。 |
| `training_curves.png` | reward、actor loss、critic loss 等训练曲线。 |
| `building_cluster_assignment.csv` | 每栋建筑被分到哪个固定组。 |
| `cluster_centers.csv`, `cluster_summary.json` | 分组中心、大小、特征和方法信息。 |
| `router_history.csv` 或 router diagnostics | Soft Router 每阶段的使用率、熵、prior、collapse 指标。 |
| `checkpoints/checkpoint_ep*.pt` | 三阶段/稳定 router 保留的中间阶段 checkpoint。 |

只有分组文件而没有 `checkpoint.pt`、`latest_metrics.json` 和测试文件，说明训练没有完整结束，不能放进 multi-seed 均值。

结果目录命名建议至少包含：算法、分组方法、特征、fusion、climate、episodes 和 seed，例如：

```text
mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed0
```

## 9. 汇总指标和生成图

### 9.1 重新生成 selected experiment 汇总表

```bash
uv run python tools/summarize_selected_experiment_metrics.py
```

自定义输出目录：

```bash
uv run python tools/summarize_selected_experiment_metrics.py \
  --output_root experiment_metric_summary_new \
  --wandb_root wandb
```

注意：这个脚本的 `SELECTED_EXPERIMENTS` 是源码中的显式列表。新增实验后，必须先将结果目录名加入该列表，脚本才会汇总它。

### 9.2 从 checkpoint 重新生成测试时序和单实验图

处理 `selected_result_folders.md` 中的推荐目录：

```bash
uv run python tools/generate_load_tracking_figures.py --continue_on_error
```

只处理某个实验：

```bash
uv run python tools/generate_load_tracking_figures.py \
  --result_dir results/EXPERIMENT_NAME \
  --output_root target_folder \
  --climate VT \
  --test_month 2 \
  --continue_on_error
```

### 9.3 合并多实验负荷跟踪图

```bash
uv run python tools/combine_load_tracking_figures.py \
  --input_root target_folder \
  --output_root combined_target_figures
```

### 9.4 重画 report2 图

```bash
uv run python report2/build_figures.py
```

该脚本只读取 `experiment_metric_summary/` 中的记录数据，不会重新训练模型。

## 10. 测试与提交前检查

Reward、分组和 SOC 相关核心测试：

```bash
uv run pytest \
  tests/test_ce1_reward.py \
  tests/test_balanced_spectral_grouping.py \
  tests/test_soc_regrouping.py
```

运行全部测试：

```bash
uv run pytest
```

提交前建议：

```bash
git status --short
git diff --check
```

不要提交 `.venv/`、`__pycache__/`、`.pytest_cache/`、大型 W&B cache 或无关临时图片。是否提交 `results/` 应由项目负责人决定，因为 checkpoint 和测试时序可能很大。

## 11. 常见问题

### 11.1 `screen -r NAME` 显示没有 session

先检查：

```bash
screen -ls
```

如果 session 已退出，查看对应的 `experiment_queue_logs/.../screen.log`、`master.log` 或 `.stderr.log`。常见原因是 Python 环境错误、checkpoint 路径不存在、CPU 核数不足或依赖未安装。

### 11.2 脚本找不到正确 Python

不要依赖系统默认 `python`。显式设置：

```bash
PYTHON_EXE="$PWD/.venv/bin/python" bash scripts/SCRIPT_NAME.sh
```

### 11.3 W&B 网络问题导致实验受阻

```bash
export WANDB_MODE=disabled
```

### 11.4 训练目录已经存在

- 先检查 `latest_metrics.json` 是否存在。
- stable Soft Router 脚本在只有 checkpoint 时会自动 `--resume_checkpoint`。
- 其他脚本不一定支持自动恢复，不要直接用 `FORCE=1` 覆盖重要目录。
- 最安全的方式是使用新的 `--save_dir` 和新的 `--wandb_name`。

### 11.5 公平比较最容易遗漏什么

至少核对：

- climate、train/test month；
- feature month 和 feature columns；
- grouping method、K candidates、cluster seed；
- fusion/communication mode；
- reward 五个权重；
- episodes、seed 列表；
- hidden size、learning rates、PPO 参数；
- 是否从 checkpoint 继续训练；
- 是否使用相同测试月份和相同 comfort bounds。

## 12. 给下一位实验人员的推荐工作流

1. 使用 Python 3.11 和 `uv sync --frozen` 建立环境。
2. 运行三个核心 pytest，确认 reward、Balanced Spectral 和 SOC grouping 正常。
3. 先运行 1 个 seed、少量 episodes 的新目录 smoke test。
4. 检查 `run_config.json`、cluster assignments、training curve 和 test metrics。
5. 参数确认后再运行 matched multi-seed。
6. 通过 `summarize_selected_experiment_metrics.py` 汇总，不手工复制大量数字。
7. 只在训练设置一致时计算均值和标准差。
8. 保留每个正式结果的 checkpoint、run config、latest/test metrics 和 cluster/router artifacts。
