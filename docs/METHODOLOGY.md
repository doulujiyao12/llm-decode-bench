# 测量方法、计算口径和边界

## 三种不同的时间

1. **SSE progress gap**：同一个请求，相邻两个包含生成进度的 SSE 消息在客户端的到达间隔。
2. **GPU event interval**：服务端同一 forward stream 上包围一次 scheduler `run_batch` 的两个事件之间的时间。
3. **每输出 token 的 TPOT**：需要知道时间区间对应的真实 token 数。不能假设一条 SSE 等于一个 token，尤其启用 MTP 时。

为了避免 Markdown 数学渲染差异，本仓库使用普通文本公式：

```text
global_requests = dp_size × batch_size
sse_gap_ms(i) = 1000 × (receive_time(i) - receive_time(i-1))
window_p50 = median(per_request_median_gap)
reported_sse_p50 = median(valid_window_p50)
saved_ms = baseline_ms - candidate_ms
reduction_percent = 100 × saved_ms / baseline_ms
```

本工具只报告第一项和可选第二项，不自行除以 3 或其他接受长度。

native SGLang 的 progress 依据累计 `completion_tokens` 增加，不依据累计 text 是否重复出现；OpenAI 接口依据非空生成内容、reasoning 或 tool-call delta。role-only、usage-only、终止消息不作为生成事件。

即使 native 一个事件增加 4 tokens，也不能无条件推导 MTP 接受长度为 4：服务器可能合并多个 step 后发送。Chat 的 content 与 reasoning 分开发送也可能增加事件数量。

## 为什么采用固定请求集合

连续补请求会混入新 prefill、KV admission/transfer、prefix 匹配和 batch 整理。固定 cohort 不补请求，便于观察同一组长短上下文的 Decode。在最后一个请求开始输出之前、以及第一个请求停止输出之后，均不计入共同窗口。

它不是线上 QPS 回放。线上吞吐、TTFT、负载均衡、缓存命中分布与到达率，需要另一个开放式压测工具；不要将固定 cohort 的低延迟直接乘并发作为线上 SLA。

## 每个窗口如何验收

- 所有正式请求成功；失败请求不会从分母中删除。
- 每个请求已产生至少 `settle_events` 个进度事件。
- 使用 `perf_counter` 单调时钟，避免客户端墙钟跳变影响 gap。
- 去掉共同窗口前后的 guard 时间，每个完整窗口内各请求 gap 数量达标。
- metrics 模式要求窗口前后有覆盖样本、间隔和 scrape 延迟不超上限；失败/缺失样本不当作 0。
- 所有 rank running 恰好等于目标值；配置声明的所有等待/PD 队列必须存在且为 0。
- GPU event capture 区间及 guard 从“干净 SSE 窗口”中排除。

metrics 是轮询证据，不是连续真值。客户端事件也可能被网络缓冲合并，因此不保证严格还原 GPU 执行序列。需要精确解释 step 内部算子时，仍需框架 profiler、Nsight/rocprofiler 或其他对应工具。

## A/B 可比性

自动检查：数据 hash/顺序、BS、DP、路由、输出上限、温度、ignore EOS、预热开关、窗口规则、metrics 映射、extra body hash、event 轮数。

必须人工记录：

- 同一模型权重和 tokenizer revision、量化格式、KV dtype、MTP steps/topk/draft tokens。
- SGLang/vLLM/PyTorch/通信和算子库版本、服务启动参数、graph bucket。
- GPU 型号、卡数、时钟、功耗、温度、其他任务。
- 实际输入长度分布、prefix 复用方式、输入是否代表线上业务。
- A/B/A 重复实验；关注噪声范围，不能只选最快一轮。

相同随机种子不等于相同输入：生成器版本、tokenizer、词表或长度选项变化都会影响结果。以 manifest hash 为准。跨硬件但同模型可以对照，跨模型通常只能做描述性展示，不能叫同工作负载优化。

## GPU event 与 trace

event 测的是流上的时间区间，不是所有 kernel duration 的和。区间可包含排队、依赖等待、通信和执行空洞；其他 stream 只有通过依赖被 forward stream 等待时才体现。不能把它叫全 GPU busy time。

GPU gap 也可能包含其他 GPU 工作或 CPU 调度等待，不是“所有 SM 完全空闲”的证明。跨 rank 中位数适合展示，瓶颈定位还应查看每个 rank 和慢 rank，不只看聚合值。

参考官方资料：[PyTorch CUDA/HIP 异步语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)、[SGLang Benchmark/Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)。
