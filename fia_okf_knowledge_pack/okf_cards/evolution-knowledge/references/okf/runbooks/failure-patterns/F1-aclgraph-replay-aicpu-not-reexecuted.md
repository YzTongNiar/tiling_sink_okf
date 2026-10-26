---
schema_version: okf.v1
kind: failure_pattern
key: aclgraph-replay-aicpu-not-reexecuted
phenomenon: "aclgraph 捕获含 DNN_VM_AICPU 定制算子的图：捕获成功、固定输入重放正确；但重放期间原地更新 seq_len/meta buffer 后输出不随新值变化（分核方案固化），动态 seq 更新重放产生错误输出"
severity: critical
failed_combination: "cann-9.1.0 + torch_npu 2.10.0.post4 @ Ascend910_9362(910_93)；cann-9.2.0-beta.2 + torch_npu 2.10.0.post4 @ Ascend910_9362(910_93)；aclgraph(torch.npu.NPUGraph) 捕获 + DNN_VM_AICPU/CUSTAICPUKernel 定制节点 + 依赖 GM 元数据的 AICore 主算子；双流(aicpu_stream/event/aicore_stream)与单流两种写法同限"
operator_type: attention / aicpu-custom-op / tiling-sink
occurrences: 3
created_at: 2026-08-26
updated_at: 2026-08-27
---

# F1: aclgraph 重放不重新执行 AICPU 定制节点 + 主算子烘焙分核方案

## 现象（三连实验定性）

1. **AICPU 节点重放不重执行**：图含 metadata AICPU 算子，capture 后多次 replay，trace 中
   AICPU kernel 不再执行 → metadata 停留在捕获时刻值。
2. **主算子捕获形态烘焙分核方案**：绕过 AICPU 直接污染 GM meta buffer 后 replay，输出不变
   ——分核真值已固化为捕获时参数（尽管主 AICore kernel 在 replay 中真执行且读 live 输入/
   输出 GM，污染输出/kpool 实验证实）。
3. **混合形态 workaround 无效**：图外 eager 重算 metadata → 拷入静态 buffer → replay 主算子
   图，输出仍固化（原因同 2）。

> **独立复核（2026-08-26，协调者，决定性四元组实验，交付算子 TND_NTD + SinkGraph）**：
> eager(K1) vs eager(K2) max_abs=0.0781（有分辨力）→ 固定输入重放 vs eager(K1) **bit=0** →
> kvlen 原地更新后重放 vs eager(K2)=0.0781、vs eager(K1) **bit=0（FROZEN 实锤）** →
> K2 重捕获+重放 vs eager(K2) **bit=0（捕获用 live 值，图机制本身正确，仅重放不刷新）**。
> 与"支持 aclgraph 入图"的文档声明不矛盾：该声明的实测语义 = 可捕获 + 固定输入重放正确。
>
> **未修改参考算子直接实证（2026-08-26，omni-ops 原版双算子，TND 布局，同栈）**：
> 同一探针（tests/aclgraph_ref_op_probe.py 归档于交付目录）：sanity 0.0781 → 固定输入重放
> bit=0 ✅ → **kvlen 原地更新后重放 vs eager(K1) bit=0（FROZEN）**——参考实现同样固化，
> 本结论由"机制外推"升级为"直接实验证实"。探针：交付算子用
> `tests/aclgraph_decisive_replay_probe.py`。
>
> **CANN 9.2.0-beta.2 解锁复测（2026-08-27，fia-tiling-sink P1，decisive 探针，bit 级判定）**：
> 9.1 构建交付包在 9.2 runtime（/home/t00886357/ascend-9.2.0 + deploy_env_92.sh 双条目部署，
> torch_npu 2.10.0.post4 未换 wheel）直接复测：sanity 分辨力 0.0781 充分 → 固定输入重放
> bit=0 ✅ → kvlen 原地更新后重放 vs eager(K1) **bit=0、vs eager(K2)=0.078125（FROZEN 不变）**
> → K2 重捕获+重放 bit=0。**failed_combination 增加 cann-9.2.0 组合**。
> 附注：ref 探针在 9.2 不可运行（其 omni-ops 包为 9.1 构建，host libopapi 符号解析失败，
> 尽力而为两次尝试即止）；FROZEN 的平台级定性由 9.1 双实现对照 + 本轮 9.2 decisive 共同支撑。
> 证据：npu_fused_infer_attention_score_v2_sink/results/round_3/r3_p1_env_migration.json。

## 后果

- 动态 seq-len（每步 kvlen 递增的 decode/MTP）场景：重放输出**错误**（按旧 seq 范围计算），
  不是单纯性能损失——正确性问题。
- 固定输入（shape 与值均不变）捕获/重放完全正确；replay 计时仅含 AICore 主 kernel，远快于
  eager 全链路（C11 形态 218-348µs vs 860-895µs）。

## 影响范围

平台机制而非实现缺陷：omni-ops 参考实现（ai_infra_fused_infer_attention_sink 双算子）同栈
同限（其官方示例即固定输入 replay）。任何依赖"AICPU 定制节点在图重放中重执行"或"重放时 GM
元数据更新生效"的设计在本栈都会踩中。

## 规避/检判

1. 接入前必跑**重放动态性探针**：capture → 原地 copy_ 更新 kvlen → replay → 校验输出与
   同值 eager bit 级一致（cos 判定不敏感，必须 bit/大幅值差判定）。
2. 当前栈可用边界：seq 值固定的图（或按"值桶"重捕获，代价高）。
3. 解锁条件：CANN/torch_npu 版本支持 AICPU 定制节点重放（或 tiling-sink 算子重放刷新）；
   torch_npu 生态已有 graph 参数更新机制方向（torchair npugraph_ex 的 updated_param_keys
   处理 seq-len 类参数），升级后优先复测。（2026-08-27 已复测 cann-9.2.0-beta.2：
   **仍未解锁**；W5 接入按 eager 混合路由评估或继续等平台。）
4. 证据工件：fia-tiling-sink npu_fused_infer_attention_score_v2_sink/results/round_2/r2_evidence.json
   （三连实验记录，可直接复用的探针方法与判定口径）。
