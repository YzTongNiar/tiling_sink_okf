---
schema_version: okf.v1
kind: failure_pattern
key: nfs-opc-build-race
phenomenon: "AscendC 算子在网络文件系统上构建随机失败：opc 内层 make 报 'No targets. Stop.' 或结果 json 为空（JSONDecodeError: Expecting value），进而 mix_aic/mix_aiv_0.o 缺失、FATBIN 阶段 ld.lld cannot open；失败目标每次不同，降低并行度不能根治"
severity: high
failed_combination: "/mnt/share 网络盘（NFS 类，服务器时钟偏移 0.4-1.5s）+ opc/TILINGKEY_PAR_COMPILE 多内核变体并行构建 + omni-ops/ops-transformer 构建框架（cann-9.1.0 bisheng）"
operator_type: any-ascendc-op-build
occurrences: 2
created_at: 2026-08-26
updated_at: 2026-08-26
---

# F2: 网络盘上 opc 构建的 .mk 读取竞态（空 makefile → 缺 .o → FATBIN 失败）

## 根因

opc 编译每个内核变体时先生成 pid 后缀的 `.mk` 再立即 `make -f` 执行；网络盘属性缓存/时钟
偏移下，make 会读到**尚未可见/为空**的刚生成文件 → `make: *** No targets. Stop.` →
`<kernel>_mix_aic_0.o`/`_mix_aiv_0.o` 不产出 → FATBIN 链接报
`ld.lld: cannot open ... .o: No such file or directory`，上层表现为
`read json file failed (JSONDecodeError)`。并发越高命中概率越大，但单进程也可能命中。

## 规避（已验证一次成功率 100%）

**源码树 cp 到本地盘（如 /tmp tmpfs）构建，产物拷回共享目录**：

```bash
mkdir -p /tmp/build_src && cp -a <repo>/{CMakeLists.txt,build.sh,cmake,scripts,src,...} /tmp/build_src/
cd /tmp/build_src && source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -n "<op1>;<op2>" -c "ascend910_93"
# 成功后 cp -a 产物回 output_dir（install/custom、output/*.run）
```

注意：本机无 rsync；增量重试不可靠（build.sh clean 会整树重建），直接本地盘全量最稳。

## 快速判别特征

- 日志同时出现 `Clock skew detected` / `modification time N.Ns in the future` 警告；
- 失败 target 编号在两次全量构建间漂移（非确定）；手动单跑同 target 可复现或不复现不定。
