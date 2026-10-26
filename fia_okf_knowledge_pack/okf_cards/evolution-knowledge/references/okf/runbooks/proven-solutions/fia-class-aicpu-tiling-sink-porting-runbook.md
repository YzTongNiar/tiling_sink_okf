---
schema_version: okf.v1
kind: proven_solution
title: "FIA 类算子 AICPU tiling 下沉移植 runbook（omni-ops 双算子形态）"
strategy_ids: [X47]
Source: "npu_fused_infer_attention_score_v2_sink（fia-tiling-sink 轮 round 1-3，2026-08-26，Ascend910_9362/910_93）"
Speedup: "C20 1.026x / C19 1.034x（长序列硬门达标）；C11 1.074x；精度 20/20 与官方基线 bit 级一致"
Operator type: attention（MLA/paged/rope，可推广至任意 host-tiling 依赖 host 值的动态算子）
Hardware: Ascend910_9362（910_93，AIC20/AIV40）/ CANN 9.1.0
---

# FIA 类算子 AICPU tiling 下沉移植 runbook

**问题**：算子 tiling 依赖 host 侧 seq len 值 → 框架每步 list 落卡（H2D）+ tiling 无法与值
解耦 + 图模式逐层重绑。**方案**：复制 omni-ops
`ai_infra_fused_infer_attention_sink(+_metadata)` 双算子形态（策略 X47），按下述步骤移植。

## 步骤

1. **复制双算子目录 + 构建框架**到输出工作区（源仓只读）：
   主算子（op_host/op_kernel/op_api/def/tiling）+ metadata 算子（op_kernel_aicpu/op_api/
   infershape/config.ini）+ 依赖的 common/ 与 cmake 框架。
2. **全局改名**：GE/AICPU 算子名、AICore kernel 入口符号、REGISTER_CPU_KERNEL、
   REGISTER_TILING_DATA_CLASS、CMake OP_NAME、config.ini、torch schema；AICPU so 独立命名
   （避免与源仓包同装冲突）。
3. **SoC 注册**：tiling 模板注册向量补目标 SoC（如 `{ASCEND910B, ASCEND910_93}`）。
4. **【必修】seq 解析归一**（失败模式 F3）：AICPU 侧把 TND_NTD/NTD_TND 归一到 TND 累积和
   语义（GetBatchSize/GetS1Size/GetActualSeqInfo/CheckFeature*/CheckActualSeqLens*），
   CalcS2Range 加 s1Size==0 防御；修复后回归 TND 路径 bit 不变。
5. **cmake 硬编码路径检查**：框架内 tiling_sink.cmake 等若按原算子名拼路径，需改指新算子。
6. **本地盘构建**（失败模式 F2）：`cp -a` 源树到 /tmp，`bash build.sh -n "<main>;<meta>"
   -c ascend910_93`，产物（install/custom + output/*.run）拷回输出目录。
7. **部署**：`export ASCEND_CUSTOM_OPP_PATH=<pkg>/packages:<pkg>/packages/vendors/<vendor>`
   （双条目缺一不可）；torch 扩展 setup.py bdist_wheel + pip 安装。
8. **python 入口两段式**：entry() 内先 metadata（aic/aiv 核数从
   get_device_properties 取，勿用 schema 默认值）再主算子（必须带 meta_data，无 meta 直接
   aclnn 失败）；seq_len 全程 device INT64[B]。
9. **验证链**：metadata 单测（meta 关键字随 kvlen 变化）→ 最小 case golden → 全量门禁 →
   profiler 差分（D2H/H2D/aten::item=0）→ 图模式探针（按失败模式 F1 的重放动态性探针）。

## 关键量化（决定该不该用此形态）

每 call 固定成本 ≈ AICPU 系统开销 130-317µs + 双 aclnn 串行 150-190µs；
主 kernel 与原算子同耗时。**主 kernel ≳500µs 的 kernel-bound 负载净平/微赚，
launch-bound 短负载净亏。**
