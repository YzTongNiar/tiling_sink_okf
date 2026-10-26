---
id: X46
origin: discovered
discovered_round: 6
discovered_from: "npu_fused_infer_attention_score_v2 (MLA) evolution rounds 2-6 (P0 torchair-GE 机制证明 → r4 eager 下沉双变体证伪 → r6 aclgraph 纯 C++ ge::Session 收口)"
base_speedup: "F 形态 vs 官方 eager 归档基线全量 11 case ratio 0.332-0.941 全过门（C11 硬门 0.941，speedup 1.063；小 case 2-3x）；eager 极限对照 1.0079（C11 0.992 勉强过门，小 case 仍 1.0-1.5 超线）"
confidence: single_op
status: active
op_families: [attention, flash_attention, mla, gqa, decode]
bottlenecks: [mte2_stall, no_overlap, scalar_loading]
tags: [aclgraph, ge-session, device-tiling, tiling-sink, dynamic-shape, seq-len-tensor, zero-d2h, graph-cache, precision-mode]
strategy_ids: [X46]
---

# X46: aclgraph 图形态设备侧 tiling 真下沉（动态 seq-len 算子的零 D2H 形态）

## 核心思路

动态 shape 算子（seq len 每请求变化）在 eager aclnn 通道下，host 侧 tiling 必须拿到
seq len 真值 → 每请求一次 pinned D2H 同步（30-100µs，小 case 占比 50%+）。把 seq len
从 aclIntArray（host 值）改为 **DEVICE INT64[B] Tensor 图输入**，经 aclgraph（ge::Session
纯 C++：GEInitialize + AddGraph + CompileGraph + LoadGraph +
ExecuteGraphWithStreamAsync）构图执行，官方 tiling parser 的 isMaxWorkspace sink 分支被
激活，tiling 在 device 侧重跑 → 客户端零 D2H、零 H2D，host submit（38-67µs）与 device
执行完全重叠（per-iter ≈ max(host, device)）。

收益双份：消除 D2H 同步等待 + 小 case 的 host 开销占比归零。本例 C11 935.8µs vs 基线
994.7（ratio 0.941），小 case ratio 0.33-0.70（eager 通道同 case 1.0-1.5）。

## 适用场景

- 动态 shape（per-request 变长度）算子：MLA/GQA/FA decode、varlen attention、动态 batch
- eager 通道 D2H 同步是可测瓶颈（pinned D2H e2e ~30µs+ 或 bench 小 case ratio>1）
- 目标算子已有官方 opp 二进制（零内核改动，纯客户端/构图层）
- 关键判据：算子 tiling parser 有 isMaxWorkspace/sink 分支（GE 执行图 dump 出现
  CacheableTiling/FindTilingFunc/TilingParse/SinkNodeBin 节点即激活）

## 实现要点（判死陷阱全在细节）

1. **DYNAMIC 输入连接**：proto 表没有动态名时，`op.DynamicInputRegisterByIndex("key",1,1)` /
   `("value",1,2)`（graph/operator.h:517 public）是唯一可行注册方式，随后
   `SetInput((uint32_t)1, dk, 0)` 按索引连接。
2. **精度模式必须显式关 cast**：多 dtype 集合输入（集合首项 FP16）在 GE 默认精度模式下被
   整体解析为 FP16 + 隐藏 5+1 向量核 cast 管线（本例 C11 劣化 3 倍，且精度门照过！必须
   msprof 验 task 形态）。解 = session option `ge.exec.precision_mode=must_keep_origin_dtype`
   （precision_mode_v2 同值会被 GEInitialize 拒绝）。
3. **环境净化**：unset ATB_*/ASDOPS_*；pip libtriton 的 pybind ctor GIL 断言用
   LD_PRELOAD no_triton hook（空 DSO 拦截）绕过；proto 经 dlopen libopsproto.so
   RTLD_NOW|RTLD_GLOBAL；链接 -lascendcl -lge_runner -lgraph -lgraph_base -lge_common。
4. **per-shape graph cache**：每 shape 一张图（AddGraph 一次），per-request 只
   ExecuteGraphWithStreamAsync；首次 JIT ~10s（kernel_meta 缓存后消失）。
5. **msprof 盲区**：msprof 对 GE 图形态不产 op_summary（device sqlite 空），证据从
   ge_info.db + api_statistic.csv 导出；脚本化时 app/datadir 必须绝对路径。
6. **gert::Tensor 非 nothrow-move**：vector 增长退化拷贝会编译失败，须定长 vector +
   下标移动赋值。

## 真下沉活性验证方法（可复用的判定协议）

- 进程内 device 侧编辑 seq len 值 → 同图重执行 → 输出哈希变化且零图重建（真消费）
- 超容量值（tiling 块数超 bt 上限）被 Execute 拒绝：host 无 D2H 无从知晓值 → 检查必在
  device tiling（sink 活性的最直接证明）
- 跨进程输出与官方 eager bit-exact（语义等价）
- 对照组：eager 通道同编辑必须 D2H 才生效（证明差异来自形态而非算子）

## 反面证据（为什么必须离开 eager）

eager aclnn 通道结构性做不到（双变体独立证明）：(a) host tiling 需 seq len 真值 →
必须 D2H；(b) executor 的 kernel 参数装配由输入形式驱动——device tensor 形式的 seq len
产生与官方 .o 编译期预期不符的参数布局（TilingDataSize/hostInputInfoNum/argsSize 全部
错位）；(c) 官方 isMaxWorkspace sink 分支在 eager 下是编译期 dummy（DoOpTiling 跳过
FillTiling）——真 sink 只存在于 GE 管线。eager 极限 = 客户端 launch 开销优化
（IntArray 移到 D2H sync 之后 + shadow window + stream 句柄缓存，本例 1.0079）。

## 来源

自动发现于 npu_fused_infer_attention_score_v2 进化 round 2-6（P0 torchair-GE 0.3546 证机制
→ round_4 证伪 eager 下沉 → round_6 aclgraph 纯 C++ 收口 11/11 + 真下沉五重证据），
C11 speedup 1.063，全 case bit-exact。
