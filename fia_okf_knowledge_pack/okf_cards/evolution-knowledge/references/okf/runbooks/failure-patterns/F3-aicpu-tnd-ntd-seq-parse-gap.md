---
schema_version: okf.v1
kind: failure_pattern
key: aicpu-tnd-ntd-seq-parse-gap
phenomenon: "AICPU 定制算子读 device seq_len 计算分核时，input_layout='TND_NTD' 直接执行崩溃（device aicpu exception errcode 0x2a，无任何 KERNEL_LOG 输出）；同参数 input_layout='TND' 完全正常——layout 字面值匹配缺口导致 s1Size 等关键量未被赋值"
severity: high
failed_combination: "omni-ops ai_infra_fused_infer_attention_sink_metadata（AICPU CpuKernel）+ input_layout=TND_NTD/NTD_TND + MLA 分支（rope=64, D=512, kv_heads=1）；host 侧 FiaInfoParser 同布局解析正常，形成 host/AICPU 行为不一致"
operator_type: attention / aicpu-custom-op
occurrences: 1
created_at: 2026-08-26
updated_at: 2026-08-26
---

# F3: AICPU 内核 seq 解析只匹配字面 "TND"/"NTD"，TND_NTD 落入未设值路径崩溃

## 根因（二分实证）

`GetBatchSize / GetS1Size / GetActualSeqInfo / CheckFeatureActualSeqLensQ(Kv) /
CheckActualSeqLensQ(Kv)` 等辅助函数一律以 `inputLayout_ == "TND" || == "NTD"` 判断是否
按累积和语义解析 qlen；`"TND_NTD"`（以及 NTD_TND）不匹配 → s1Size 保持 0、isAccumQSeq
保持 false 等，后续 SplitCore 计算在未设值状态触发了致命路径（整除/越界类，AICPU 侧无日志，
host 侧同布局由 FiaInfoParser 正确导出，故仅 AICPU 路径暴露）。

## 规避/修复（已在 npu_fused_infer_attention_score_v2_sink 中落地并回归）

1. 把 `TND_NTD/NTD_TND` 归一到与 `TND` 相同的解析语义：bSize 取 qlen 长度、
   s1Size 取逐 req 差分最大值、isAccumQSeq=true、维度校验包含之；
2. `CalcS2Range` 增加 s1Size==0 防御跳过；
3. 修复后必须回归原 TND 路径 bit 级不变（防行为漂移）。

## 通用教训

移植"host tiling ↔ AICPU tiling 双胞胎"实现时，**两侧 layout/分支判定集合必须逐一 diff**——
host 侧由 parser 兜底的派生量（s1Size/bSize/accum 标志），AICPU 侧往往只在部分 layout 字面值
下自行推导，新增 layout 支持时极易漏改 AICPU 侧。
