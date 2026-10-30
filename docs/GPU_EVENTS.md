# 可选 SGLang GPU event 插件

## 使用前必须知道

- 默认未启用。已有服务只要没有加载 hook，就只能用本工具的 SSE/metrics 功能，不能事后靠客户端增加 GPU event。
- 仅支持匹配适配契约的 SGLang + PyTorch CUDA/HIP；GPU event 功能**不是**任意框架通用接口。
- 不要求修改 SGLang 仓库，但在进程内包装 `Scheduler.run_batch`。没有替换算法、模型权重、collective 或调度决策。
- 调用 `Event.record`，后续轮次用 `Event.query` 查询完成状态，再读取 `elapsed_time`；不调用 `synchronize`。
- 只支持单 pipeline stage。多机/跨 PP stage 的统一时间线不在本版本范围内。
- 不使用 ROCm SDK 注入，不创建 signal handler，不自动停止服务。

## 1. 服务侧准备

在真正启动服务的 Python 环境中安装本仓库，或者把其 `src` 加入 `PYTHONPATH`。下面是环境准备，不替换你的服务启动参数：

```bash
REPO=/path/to/llm-decode-bench
export LDB_EVENT_DIR=/path/to/telemetry/decode-bench-001
mkdir -p "$LDB_EVENT_DIR"
chmod 700 "$LDB_EVENT_DIR"
export LDB_EVENT_LEADER_TP_RANKS=0,1,2,3,4,5,6,7
export PYTHONPATH="$REPO/hooks:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# 现在执行原本的 SGLang 启动命令/脚本。
# 如果原启动脚本完全覆盖 PYTHONPATH，需要把上述路径合入最终值。
```

该目录应只允许自己的进程访问。控制文件会触发测量，不应暴露给其他租户或公开共享。客户端和服务需要看见同一目录内容，容器 mount 可以不同路径但必须指向同一宿主目录。

**leader 设置：** 每个 attention DP rank 只选择一个 TP worker。TP8/DP8 常见布局为 `0,1,2,3,4,5,6,7`；纯 TP8/DP1 用 `0`。其他拓扑请核实实际 rank 映射，不按 GPU 个数猜。默认只选择 TP rank 0，漏选会导致 capture 不完整并报错，不会把少数 rank 当作全机。

**已有 sitecustomize：** Python 只导入找到的第一个 `sitecustomize.py`。如果环境原本就有，不要覆盖其初始化逻辑；在已有入口中显式调用：

```python
from llm_decode_bench.sglang_hook import install_from_env
install_from_env()
```

hook 通过 PYTHONPATH 传给 spawned Python workers。运行 `python -S`、启动脚本清除环境、框架更换 scheduler module 路径，都可能导致未安装；检查 `status-<pid>.json`，不要只看客户端是否连通。

## 2. 完整客户端例子

```bash
llm-decode-bench run \
  --base-url http://127.0.0.1:8000 \
  --backend sglang --model your-served-model \
  --dataset data/mixed.jsonl \
  --dp-size 8 --batch-size 32 --route-dp \
  --output-tokens 8192 --temperature 0 --ignore-eos --warmup \
  --steady-mode metrics \
  --metrics-url http://127.0.0.1:8000/metrics \
  --event-dir /path/to/telemetry/decode-bench-001 \
  --event-steps 128 --event-timeout 600 \
  --output results/mixed-bs32-event
```

客户端等自己的请求均开始输出、metrics 达标且持续 guard 时间，再原子写入 `control.json`。每个被选中的 scheduler 只测符合以下条件的 batch：

- `batch.forward_mode.is_decode()` 为真；
- 请求数等于目标 BS；
- 所有 rid 都属于本次正式测试的命名空间；
- 控制文件未过期，尚未收集满指定轮数。

收到各 rank 完成文件后复制 event JSONL 到结果目录。分析再检查每 rank 轮数、BS、rid 集合稳定、全局 rid 覆盖，以及显式路由时每个 rank 的请求分配。头尾各去掉 2 轮再统计。

SSE 分析排除 capture 和 guard 区间。输出长度需要足够长，让 capture 后还有干净 SSE 窗口；否则会得到 GPU 数据但整个 run 的 SSE 验收不通过。

## 3. 适配契约与故障处理

当前 adapter 查找 `sglang.srt.managers.scheduler.Scheduler.run_batch`，读取：

```text
scheduler.ps.tp_rank / scheduler.tp_rank
scheduler.ps.attn_dp_rank / scheduler.attn_dp_rank
scheduler.forward_stream
batch.forward_mode.is_decode()
batch.reqs: rid, origin_input_ids, output_ids
```

这些是框架内部实现，版本可能不同。缺失字段或事件 API 出错时，记录 `disabled`/`unsupported` 状态并停止测量；原 `run_batch` 仍执行，其异常仍传播。请先审查状态文件与版本，不能将 incomplete capture 当成 0 ms。

控制目录中的 `client.lock` 防止多个客户端相互覆盖 capture。正常结束会释放锁；客户端被 SIGKILL 时可能遗留锁。**先确认没有该 capture 客户端和有效 control 仍在运行**，再人工清理对应专用目录里的锁。不要清理其他实验目录。

event 本身和 Python/文件 I/O 有少量开销。它用于测量而不是无开销旁观；正式优化结论建议另跑不启用 event 的 SSE-only A/B。GPU hook 的当前 CPU 测试不构成真实 CUDA/HIP 验收。
