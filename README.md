# PhyNDR v0.3.3

> **Current contract (2026-08-15):** the primary output is absolute `partition_utilization [N_P,2]`; no capacity-loss transform is used. See [`TRAINING_NDR720.md`](TRAINING_NDR720.md). The older delta/capacity paragraphs below are retained only as v0.3.3 historical notes.


PhyNDR 是面向 region/layer NDR 选择的物理约束异构图神经网络。模型读取候选 NDR 实施前的 baseline 设计状态以及一个候选 action assignment，预测该 assignment 相对 baseline 的质量变化；它不是 RouteGNN，也不负责执行 routing。

## 固定任务契约

每个有效资源节点 `R(P,L)` 必须选择一个 action：`1W1S`、`1W2S`、`1W3S`、`2W3S`。其中全 `1W1S` 与完全不注册 region NDR 严格等价，不设置 `ndr_enable`。

输出固定为：

- `delta_congestion: [N_P,2]`，列顺序为 `[Δcongestion_H_mean, Δcongestion_V_mean]`；
- `delta_chip: [B,3]`，列顺序为 `[ΔDRC, ΔWNS, ΔTNS]`。

统一定义 `Δmetric = metric_after_NDR - metric_baseline`。PowerMap 可以作为 baseline partition state 的组成部分，但当前版本没有 `ΔPower` head。

## 图与网络

节点类型为 `r=R(P,L)`、`u=U(P)` 和少量显式 `n_critical=N*(k)`。固定关系包括同层 H/V 物理邻接、同 partition 跨层结构边、`R↔U` 归属边、相邻 partition boundary H/V 边，以及 `U↔N*` 逻辑关联边。关键 net 逻辑边不代表物理 route，普通非相邻 net 不建立物理直连。

模型首先编码 supply、baseline layer/partition state、ordinary-net demand、critical-net timing state 与 action。初始容量为：

```text
C0(P,L) = track_number(P,L) × [1 - blockage_rate(P,L)]
explicit_scale = (default_width + default_spacing)
               / (width_ratio × default_width + spacing_ratio × default_spacing)
Ceff = clamp_min(C0 × explicit_scale + gated_learned_residual, 0)
```

`1W1S` 的 learned residual 被严格门控为零。Physics Bridge 按 preferred direction 汇总容量，并计算 `rho=D/(C+eps)`、margin 与 overflow；H 上下文只广播到 H layer，V 同理。NDR 只能经 `Action@R → Ceff@R → R→U` 影响 partition 状态。

Backbone 固定为三个完整 relation-aware HeteroBlock。第三层的最终 `n_critical` 状态和最终 partition 表示分别做全局 attention pooling，拼接后共同进入 DRC、WNS、TNS 三个独立 head。最终 `r3` 按 H/V 分别 attention pooling，再与单一 `u3` 状态形成 partition 表示。

## 输入字段

`r` 节点：

- `x_supply [N_R,8]`：当前默认索引 0/1/2/3 分别是 track、blockage、default width、default spacing；其余列由数据契约确定；
- `x_state [N_R,2]`：该 `(P,L)` 自己计算的 preferred-direction baseline congestion mean/P90，不能从 partition 标量复制到各层；
- `action_id`、`width_ratio`、`spacing_ratio`、`direction`。

`u` 节点：`x_state [N_P,8]`、`x_demand [N_P,6]`、`demand_h`、`demand_v`。`n_critical` 节点：`x [N_NC,8]`，用于 slack、criticality、critical-path membership、fanout/span 等基线信息。连续字段必须仅用训练集统计量标准化，缺失值必须配套 mask；正式列定义仍需在真实数据适配阶段冻结。

## 损失

默认使用 Huber。partition H/V 两列共同构成一个任务，另有 DRC/WNS/TNS 三项，共四个 uncertainty-weighted task：

```text
L = Σ_i λ_i [exp(-s_i) L_i + s_i]
```

配置里的四个 `lambda=1` 是初始先验，不是最终固定权重；`s_i=log(σ_i²)` 可学习。训练优化器必须同时包含模型参数和 loss module 参数。`fixed` 仅用于消融。

## 运行

```powershell
python -m pip install -e .[dev]
$env:DGLBACKEND = "pytorch"
$env:PYTHONPATH = "$PWD\src"
python -m pytest -p no:cacheprovider
python examples\run_synthetic.py
```

Windows 若仅因可选 GraphBolt DLL 与 PyTorch 版本不匹配，可尝试 `PHY_NDR_DISABLE_GRAPHBOLT=1`；正式训练仍应使用 DGL 官方支持的 PyTorch/DGL 组合。

## 能力边界

本版本不重建真实逐层 net route、via 数量/类型/stack，不将 NDR 后 routing/DRC/timing/congestion 作为输入，也不模拟 OpenROAD 标签。真实 region/layer NDR 注册、执行以及 `Δcongestion/ΔDRC/ΔWNS/ΔTNS` 采集由后续自动化数据集工具提供。

