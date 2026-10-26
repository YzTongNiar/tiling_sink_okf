# FIA tiling 下沉 OKF 知识卡片迁移包（2026-08-28）

用途：在目标机继续 vllm-ascend 接入调试与后续进化工作时，携带本项目 FIA AICPU tiling 下沉
的全部知识沉淀。
来源：源机 cann-skills-private 仓 `plugins-community/ops-perf-evolution/skills/`
（round_3 收口态）。卡片正文一律中文（frontmatter 结构化字段除外，遵循知识沉淀约定）。

## 卡片清单（okf_cards/ 下，保留插件树内相对路径）

| 相对路径 | 卡片 | 一句话 |
|---|---|---|
| evolution-knowledge/.../okf/runbooks/proven-solutions/aicpu-tiling-sink-design-pattern.md | **设计模式方法论** | host 值依赖 → 双算子 + GM 元数据：七要素架构 / 六条设计原则 / 五层验证链 V1–V5 / 成本判据；§5.1 含 GQA@9.2 复验 |
| evolution-knowledge/.../okf/runbooks/proven-solutions/fia-class-aicpu-tiling-sink-porting-runbook.md | **移植 runbook** | FIA 类算子按 omni 双算子形态下沉的执行级操作手册（X47 的 runbook 版） |
| evolution-knowledge/.../okf/runbooks/failure-patterns/F1-aclgraph-replay-aicpu-not-reexecuted.md | **F1** | aclgraph 回放不重执行 AICPU 定制节点 + 主算子烘焙分核（9.1/9.2、MLA/GQA 四组合同限）→ **eager-only 铁律依据**；私有仓 docstring 相反说法以本卡探针实测为准 |
| evolution-knowledge/.../okf/runbooks/failure-patterns/F2-nfs-opc-build-race.md | **F2** | 网络盘/NFS 上 .opc/.o 构建竞态 → 本地盘构建纪律（红线 R5）依据 |
| evolution-knowledge/.../okf/runbooks/failure-patterns/F3-aicpu-tnd-ntd-seq-parse-gap.md | **F3** | AICPU 侧 `"TND"` 字面量解析崩溃（0x2a）→ metadata 的 input_layout_kv 勿传 TND（BnBsH/PA/BSH 等价可用） |
| evolution-knowledge/.../okf/strategies/discovered/disc_X47.md | **X47（knowledge 侧）** | 双算子形态策略卡（multi_op_form；tiling-key 包络检查点） |
| evolution-strategies/.../discovered/disc_X46.md | **X46** | aclgraph 图形态设备侧 tiling 真下沉（MLA 谱系；图模式后续跟进的背景知识） |
| evolution-strategies/.../discovered/disc_X47.md | **X47（strategies 侧副本）** | 与 knowledge 侧同源；两处都随包携带，保持索引路径不失配 |

## 用法 A：目标机装有 ops-perf-evolution 插件

1. 把 `okf_cards/` 下内容按**相同相对路径**合并进目标机的
   `plugins-community/ops-perf-evolution/skills/` 树；
2. 索引合并：`index_snapshots/` 是源机当前态快照，**先 diff 后合并、勿整文件覆盖**目标机自己的
   条目——关键注册点：
   - `evolution_knowledge/INDEX.json`：X47 注册项
     （source_key=`evolution-knowledge#okf/strategies/discovered/X47`，格式见同目录 SOURCE_KEY.md）；
   - `a3_proven_solutions/INDEX.md` L739 附近：X47 条目交叉引用 runbooks 两卡与 F1/F2/F3；
   - `evolution_strategies/INDEX.json` 与 `strategy-index.md`：X46、X47 行；
3. 合并后跑该 skill 的索引重建脚本（如有）验证召回。

## 用法 B：目标机无插件（纯参考阅读）

卡片自包含，直接按上表阅读/让 agent 把本目录当参考资料读入即可。
接入期最常被引用的三张：**设计模式方法论（五层验证链）**、**F1（为何 eager-only）**、
**移植 runbook（双算子形态与 GM meta 契约）**。

## 回迁口径

目标机上的新增沉淀先在目标机登记；回源机时以源仓为权威做合并，避免双头写。
配套件：部署/接入套件 `fia_sink_deploy_kit.tar.gz`（算子预编译包 + 手册 docs/10、13 + 模板）。
