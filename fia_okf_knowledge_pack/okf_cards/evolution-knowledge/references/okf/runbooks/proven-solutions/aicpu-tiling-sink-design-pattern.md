---
schema_version: okf.v1
kind: proven_solution
title: "AICPU tiling 下沉设计模式（host 值依赖 → 双算子 + GM 元数据，可迁移方法论）"
strategy_ids: [X47]
Source: "npu_fused_infer_attention_score_v2_sink（fia-tiling-sink，2026-08-26，Ascend910_9362/910_93，CANN 9.1.0；2026-08-27 W1+W2：CANN 9.2.0-beta.2 迁移 + GQA 开放复验）"
Speedup: "C20 1.026x / C19 1.034x（长序列硬门达标）；GQA kernel-bound S1/S2 时间比 1.024/1.029（成本模型地板内持平，严格口径 0.977/0.972）；精度与官方基线 bit 级一致；零 D2H/H2D/aten::item"
Operator type: 任意 host tiling 依赖 host 侧动态标量（seq len 类）的算子（attention 先行实证）
Hardware: Ascend910_9362（910_93）/ CANN 9.1.0（机制与 SoC 无强绑定，移植时复查注册向量）
confidence: pattern
status: active
op_families: [attention, dynamic-shape-tiling]
---

# AICPU tiling 下沉设计模式（通用方法论卡）

> 目的：把 FIA（MLA）一轮完整实证的 tiling 下沉方案抽象成**可迁移模板**，指导后续算子的
> AICPU tiling 下沉优化。步骤级操作见同库
> `fia-class-aicpu-tiling-sink-porting-runbook.md`（9 步移植 runbook）；
> 失败模式见 F1（aclgraph 重放不刷新）/ F2（NFS 构建竞态）/ F3（布局解析缺口）。

## 1. 问题定义与适用判据

**问题特征**：算子的 tiling（分核/分块决策）读取 host 侧动态标量（典型：actual_seq_len 列表），
导致 ① 框架每步 list 落卡（H2D）② tiling 与值耦合、值变必重算 ③ 图模式逐层重绑 hack。

**先算账再动手（本轮实证的成本模型）**：下沉后每 call 固定地板 ≈
AICPU 系统开销 130–317µs（本体 <1µs，全是调度成本）+ 双 aclnn 串行 150–190µs；
主 kernel 与原算子同耗时。**判据：主 kernel ≳500µs 的 kernel-bound 负载净平/微赚；
launch-bound 短负载净亏（实证最差 8.65x）**。不满足先做负载画像，不要盲目下沉。

## 2. 架构模板（七要素）

```
① 双算子形态：<op>_metadata（AICPU 前置）+ <op>（AICore 主体，MIX_AIC_1_2）
② AICPU 接入方式：DNN_VM_AICPU 引擎 + REGISTER_CPU_KERNEL + CUSTAICPUKernel 库（so 独立命名）
③ GM 元数据契约：uint32[1024] = aicMeta[核数][10] + aivMeta[核数][3] + baseMeta[10]
   （AICore 侧提供 GetAIC/AIV/BaseMetaAbsIndex 内联寻址，kernel 不经 tiling buffer 直读 GM）
④ host tiling shape-only 化：只留保底 mBaseSize、最大 workspace、blockDim 恒全核
   ——同 shape 不同值 → tiling 逐字节相同（这是"值依赖已搬走"的可验证判据）
⑤ 动态标量全程 device tensor（INT64[B]），每步原地 copy_，零拷贝零重绑
⑥ python 入口两段式：先 metadata aclnn、后主算子 aclnn，同 stream 天然保序（eager 无需 event）
⑦ 主算子必须携带 meta_data 入参（无 meta 直接 aclnn 失败）
```

## 3. 可迁移设计原则（按重要性排序）

1. **tiling 二分法**：值依赖下沉 AICPU、shape 依赖留 host。host 侧残留任何值读取都意味着下沉不彻底（用 profiler 的 D2H/aten::item=0 做终验）。
2. **元数据放 GM 而非 tiling buffer**：AICore kernel 直读，彻底消除 host 回读路径；代价是 kernel 侧要自建寻址函数（一次性成本）。
3. **AICPU 只做轻决策**：本体计算应 <1µs 量级（分核贪心/代价模型），AICPU 开销全是调度成本——把重计算放 AICPU 是错误方向。
4. **分核算法与原 host tiling 同构**：batch→行→块贪心 + FlashDecode 划分（代价模型 6·⌈M/16⌉+10·⌈S2/64⌉），保证 kernel 行为与基线可 bit 级对齐。
5. **核数等环境参数运行时取**：aic/aiv 核数从 `torch.npu.get_device_properties` 取，勿用 schema 默认值（换 SoC 即错）。
6. **布局语义在 AICPU 侧归一**：所有布局字符串（TND/NTD/TND_NTD…）进入解析辅助函数前归一到统一语义（F3 教训：只匹配字面量 → 隐藏分支落 s1Size=0 崩溃）。

## 4. 移植步骤（摘要）

复制双算子目录 → 全局改名（GE/AICPU/符号/CMake/config/torch schema，AICPU so 独立命名）→
SoC 注册向量补目标芯片（如 `{ASCEND910B, ASCEND910_93}`）→ **【必修】seq 解析归一 + 防御** →
cmake 硬编码路径检查 → 本地盘构建（F2）→ 双条目 ASCEND_CUSTOM_OPP_PATH 部署 →
python 两段式入口 → 五层验证链。逐步细节见 runbook。

## 5. 五层验证链（缺一不可）

| # | 验证 | 判据 | 本轮结果 |
|---|---|---|---|
| V1 | metadata 单测 | meta 关键字段（usedCoreNum/s2Size…）随 device 值变化 | ✅ 同 shape 不同 kvlen → 值随动 |
| V2 | 精度 golden | 与官方基线 **bit 级**一致（含 lse） | ✅ 20/20 |
| V3 | profiler 差分 | D2H=0 / H2D=0 / aten::item=0 | ✅ |
| V4 | 性能门禁 | kernel-bound 用例族 ≥1x | C19 1.034x / C20 1.026x；短负载亏损已归因 |
| V5 | aclgraph 重放动态性探针 | capture → 原地改值 → replay → 与同值 eager **bit 级**比（cos 判定不敏感，禁用） | ❌ FROZEN（F1，平台限制，参考实现同限） |

**V5 是图模式接入的先决门**：两个归档探针（交付目录 `tests/aclgraph_decisive_replay_probe.py`
与 `tests/aclgraph_ref_op_probe.py`）在任何新栈分钟级可复测——CANN/torch_npu 升级后先跑探针
再谈接入。

### 5.1 五层链在「GQA 形态 @ CANN 9.2.0」的完整复验（2026-08-27，W1+W2）

模式可迁移性的第二个算子形态实证——同一下沉包经 entry() 参数化开放 GQA 分支
（rope_head_dim=0 落 `CheckIsMla()=false`、gSize=n1/n2∈{2,4,6,8} 实测），五层链全部重跑：

| 层 | 结果 | 关键数字 |
|---|---|---|
| V1 | ✅ | GQA 形态 numOfFd 16→18→17、s2Size 8192→12288 随 device kvlen 变；mBaseSize=512（MLA 为 256，分支路由字段分化可判） |
| V2 | ✅ | 门禁 C21–C26 vs **9.2 官方** GQA 路径 `torch.equal` 全过（MERE_vs_official=0.0）；MLA C01–C20 在 9.2 复跑 mere_vs_baseline=0.0 无回退 |
| V3 | ✅ | 连续 8 call D2H/H2D/aten::item 全 0 |
| V4 | ✅ | kernel-bound GQA 压测 S1(kv256K) 时间比 1.024 / S2(kv512K) 1.029——成本模型地板带（对标 MLA 硬门 1.018–1.031）；未观察 GQA 分核额外劣化 |
| V5 | ❌ FROZEN | GQA decisive 探针同 F1 结论（固定输入重放 bit=0 ✅ / 更新后重放停捕获值 / 重捕获 live ✅）——F1 是栈级行为，与算子头数形态无关 |

**环境迁移证据（W2）**：9.1 构建交付包在 cann-9.2.0-beta.2 runtime 直接跑通（含最高风险件
AICPU .so 注册 ABI 握手），20/20 bit 级无回退；toolkit `--full` + A3-ops combo 同根目录安装；
torch_npu 2.10.0.post4 免换 wheel。换 runtime 版本不需重建的结论 + 决策树见 plans/08 §4.5。

**二进制 tiling key 包络边界（移植必查项）**：GQA 实测命中 {TND/NTD_TND × PA × D64/D128 ×
fp16/bf16 × gS{2,4,6,8}}；**D192(rope=0) 与 TND_NTD 两支 tiling key 未编进二进制**
（9.2 官方支持该形态）——开门禁前先按 08 §3.3 做 key 包络探测，缺失形态如实移出并记录
（复现命令归档 `results/round_3/r3_w1_gqa_boundary.json`），覆盖须重建补 key。

## 6. 已知边界（引用同库失败模式）

| 边界 | 影响 | 详见 |
|---|---|---|
| aclgraph 重放不刷新（AICPU 定制节点不重执行 + 主算子烘焙分核） | 动态 seq 图模式输出错误（正确性问题，非性能）；固定输入图模式完全正确且快 | F1 |
| NFS 构建竞态 | 网络盘构建随机失败（空 .mk → 缺 .o → ld.lld） | F2 |
| 布局字面量解析缺口 | 组合布局（TND_NTD）落入未设分支崩溃 | F3 |
| 固定开销地板 | launch-bound 负载净亏；未穷尽：AICPU/AICore 双 stream 重叠、meta buffer 常驻复用 | 本卡 §1 |

## 7. 实证锚点（供横向对比）

- 环境：Ascend910_9362（910_93，AIC20/AIV40）/ CANN 9.1.0、cann-9.2.0-beta.2 / torch_npu 2.10.0.post4
- 精度：20/20 用例 vs fp32 golden MERE≈0.0015（阈 0.0078）；vs 官方基线 bit 级一致（9.1 与 9.2 双环境均实测 0.0）；GQA C21–C26 对 9.2 官方 GQA 路径逐位一致
- 性能：C11（现网 8×61.8K）1.074x；C12–C18 1.06–1.56x；C19/C20 达标；C01–C10 1.42–8.65x；GQA S1/S2 1.024/1.029
- 图模式：捕获/固定输入重放 ✅（218–348µs vs eager 860µs）；动态值重放 ❌（F1，9.1 与 9.2 同限，MLA/GQA 形态同限）
- 关联策略：X47（双 aclnn 形态策略卡）；与 X46（DS 静态 bound tiling）互补——X46 治
  host 回读、本模式治 tiling 值耦合，二者不可混用同一算子层。
