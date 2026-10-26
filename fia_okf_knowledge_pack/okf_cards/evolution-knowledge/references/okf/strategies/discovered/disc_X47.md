---
schema_version: okf.v1
kind: discovered_strategy
id: X47
origin: discovered
discovered_round: 3
discovered_from: "npu_fused_infer_attention_score_v2_sink（AICPU tiling 下沉轮，fia-tiling-sink，2026-08-26）round 1-3：复制 omni-ops FIA_Sink 双算子移植 + TND_NTD 修复 + 固定开销定量归因"
base_speedup: "kernel-bound 长序列净平/微赚：C20(96头×512K) 1.026x、C19(96头×256K) 1.034x 达 1x 硬门；C11(现网 8×61.8K) 1.074x 差 25µs；launch-bound 短用例 0.12-0.7x（AICPU 系统开销 130-317µs + 双 aclnn 串行 150-190µs 固定成本主导）。精度 20/20 与官方基线 bit 级一致"
confidence: single_op
status: active
op_families: [attention, flash_attention, mla, decode, varlen]
bottlenecks: [host_launch_overhead, d2h_sync, tiling_value_dependency, graph_param_rebind]
tags: [aicpu-tiling-sink, metadata-pre-op, device-seqlen-tensor, zero-d2h, dual-aclnn, shape-only-tiling, omni-ops-porting, 910_93]
strategy_ids: [X47]
---

# X47: AICPU 前置 metadata 算子 tiling 下沉（双 aclnn 形态，eager 零 D2H）

## 核心思路

动态 seq-len 算子的分核计算（任务切分/负载均衡/FlashDecode 划分）从 host tiling 迁移到
**AICPU 定制 CpuKernel**：前置 metadata 算子在 device 读 seq_len 张量（INT64[B]，全程
device 传参）跑与 host 同构的 SplitCore 算法，把每核任务区间（bN2End/gS1End/s2End +
FD 表 + base 参数）写入 **GM 上的 uint32[1024] 元数据张量**；主算子 host tiling 变
shape-only（mBaseSize 取保底值仅算最大 workspace、blockDim 恒全核、不读任何数据值），
AICore kernel（MIX_AIC_1_2）启动后按索引内联函数直接读 GM 元数据驱动调度。

与 X46（ge::Session 图执行管线形态）互补：X47 是**算子级实现**，eager 即可用、不依赖 GE
图栈；代价是每 call 多一段 AICPU 执行 + 一段 aclnn 发射。

## 收益与代价（定量，910_9362 / CANN 9.1.0 实测）

- 收益：seq_len 相关 D2H/H2D/aten::item 归零（profiler 实证）；host tiling 与数据值解耦
  （同 shape 不同值 → tiling 输出逐字节相同）；精度与官方算子 bit 级一致（数学主体未动）。
- 代价：**AICPU 执行系统级固定开销 130-317µs/次**（trace 实测 kernel 时长；本体计算 <1µs，
  纯 DNN_VM_AICPU 调度开销）+ **双 aclnn 串行发射 ~150-190µs**（metadata GWS 55µs +
  主算子 GWS 130µs，其中 tiling 仅 46µs）。
- 净效果判据：**主 kernel 时间 ≳ 500µs 的 kernel-bound 负载净平/微赚；launch-bound 短负载净亏**。

## 适用场景

- seq len 每请求/每步变化、且负载偏 kernel-bound 的推理算子（长上下文 decode / 大头数 MLA）
- 框架侧被 host list 落卡 + host tiling 值依赖 + 图重绑（graph_task_update）困扰的调用链
- 需要图模式下"参数只更新 device 内存"语义的接入（前提：目标栈支持 AICPU 节点重放，
  见失败模式 F1-aclgraph-replay-aicpu-not-reexecuted）

## 实现要点

1. 双算子结构：`<op>_metadata`（op_kernel_aicpu/，REGISTER_CPU_KERNEL，json 声明
   engine=DNN_VM_AICPU / opKernelLib=CUSTAICPUKernel / kernelSo=独立命名 so）+
   主算子（def 中 metadata 为 OPTIONAL 输入，kernel 形参 `__gm__ uint8_t *metadata`）。
2. 元数据契约共享头（AICPU 生产者 ↔ AICore 消费者）：aicMetadata[AIC核数][10] +
   aivMetadata[AIV核数][3] + baseMetadata[10]，≤1024×u32；AICore 侧用
   GetAIC/AIV/BaseMetaAbsIndex 内联寻址。
3. aic/aiv 核数由调用方 attr 传入（torch 层从 get_device_properties 取，勿用 schema 默认值）。
4. 主算子必须带 meta_data 两段式调用（无 meta 直接 aclnn 失败，实测）。
5. 移植必改：seq 解析辅助函数对 TND_NTD/NTD_TND 的归一（见
   F3-aicpu-tnd-ntd-seq-parse-gap）、SoC 注册补 910_93、全局改名 + AICPU so 独立命名。
6. 构建必须本地盘（见 F2-nfs-opc-build-race）；部署双条目 ASCEND_CUSTOM_OPP_PATH
   （含 vendors 根 + vendor 目录本身）。

## 已证伪/待挖方向

- 官方 PTA 执行器缓存（PTAGetExecCache）对自定义算子恒 MISS，不能复用；
  绑定层自建 executor 缓存：AICPU 重发射 aclnn 报错、AICore 重发射 runtime 崩溃 → 放弃。
- 未穷尽：AICPU/AICore 双 stream 重叠发射、metadata buffer 常驻复用（省每 call zeros 清零）。

## 与 X46 关系

同一目标（seq-len 算子零 D2H）的两条形态：X46 = GE 图执行管线 device tiling（图形态收益大，
依赖 ge::Session）；X47 = AICPU 算子形态（eager 可用、侵入小，固定开销高）。按部署形态选择。
