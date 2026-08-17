# PhyNDR v0.3.3：数据与训练 Agent 契约

CURRENT OVERRIDE (2026-08-15): use absolute `partition_utilization [N_P,2]`; do not implement capacity loss. Follow `TRAINING_NDR720.md`. Any older delta-congestion/capacity text below is historical.


实现或适配数据时不得改变以下事实：输入全部来自 NDR 前 baseline；四个 action 对每个有效 `R(P,L)` 四选一；全 `1W1S` 等价于不注册 region NDR；输出为 partition `[H,V]` 两列和 chip `[DRC,WNS,TNS]` 三列；恰有三个完整异构块；最终 `n3` 参与全部三个 chip head；不存在 `ΔPower` head。

样本是“一份 baseline 芯片状态 + 一份候选 region/layer action assignment”。同一 baseline 可以对应多个 assignment。标签必须与稳定索引对应：`r_key=(partition_id,layer_id)`、`u_key=partition_id`、`n_key=stable_net_identifier`，并保存反向映射。

严格张量接口见 `src/phyndr/data/graph_builder.py`。边特征固定存放为每个 canonical etype 的 `data['x']`。每个 `r` 必须且只能有一条 `belongs_to`；无 critical net 时必须保留零节点类型及空 relation，而不是删除 schema。

真实数据适配仍需冻结：各 node/edge tensor 的正式逐列字段、单位与 missing mask；H/V congestion 标签的统计口径和标准化量；DRC/WNS/TNS 的单位、符号与标准化；critical net 选择阈值；PowerMap 表示；boundary demand 提取方法。不得静默补造这些值。

数据验收至少包括：schema 完整、action ratio 一致、无 `ndr_enable`、每个 r 单一归属、输出形状 `[N_P,2]`/`[B,3]`、旧 `[N_P,1]` 标签被拒绝、batch 不跨芯片混合、无 critical net 可运行、forward/backward 无 NaN/Inf、两个 partition 输出通道均能获得梯度。

