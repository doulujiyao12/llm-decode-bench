# 接口、metrics 与新框架适配

## HTTP 后端

| backend | URL 后缀 | 输入 | 进度判定 |
| --- | --- | --- | --- |
| `sglang` | `/generate` | `input_ids` 或 `text` | `meta_info.completion_tokens` 增加 |
| `openai` | `/v1/completions` | `prompt` 文本或 token IDs（需服务支持） | choice.text 非空 |
| `openai-chat` | `/v1/chat/completions` | 单 user text message | content/reasoning/tool_calls delta 非空 |

仅支持 `n=1`。`--extra-body` 接受 JSON 文件，但不允许覆盖工具拥有的输入、stream、model、长度、采样基础参数等字段，避免配置看似一致而实际请求不同。确有服务扩展可以放在不冲突字段中，例如：

```json
{
  "stream_options": {"include_usage": true}
}
```

不是所有实现都支持相同扩展。HTTP 4xx/5xx 会记录状态并使测试无效，不会自动重试成另一套参数。服务错误正文不写入结果，以免泄漏提示词。必要时单独用自己的诊断客户端查看服务日志。

参考 [vLLM OpenAI-compatible API](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)。本工具提供的是协议适配，不声明所有模型和服务器版本都已实测。

## 自定义 metrics

`--metrics-map examples/metrics-vllm-global.json` 示意单个 engine 的全局 gauge，必须用 `--dp-size 1`。多 engine 实际标签名请用自己 `/metrics` 输出验证，不能将多个进程 gauge 相加当作单 rank BS。

```json
{
  "rank_label": null,
  "labels": {},
  "metrics": {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting"
  }
}
```

多个模型/worker 产生同名指标时，在 `labels` 中明确过滤实际存在的标签。例如 `{"model_name":"your-model"}`；TP replica 仍重复时还需选择对应 leader 标签。解析到同 rank 同字段重复会报错，不会静默覆盖或求和。

SGLang 默认映射：

```json
{
  "rank_label": "dp_rank",
  "labels": {},
  "metrics": {
    "running": "sglang:num_running_reqs",
    "waiting": "sglang:num_queue_reqs"
  }
}
```

PD 还需加入：

```json
{
  "prealloc": "sglang:num_decode_prealloc_queue_reqs",
  "transfer": "sglang:num_decode_transfer_queue_reqs"
}
```

上面的两项应合入同一个 `metrics` 字典，不是单独传给 CLI。完整预设可用 `--metrics-preset sglang-pd`。声明的指标必须全部存在；如果服务版本根本不暴露某项，不能从缺失推导它为 0。

## PD 分离

请求发给原有 router，读取 D 的 metrics。最通用方式是 OpenAI-compatible 接口，不启用 `--route-dp` 或 GPU event；分离 D 通常不能处理缺少 bootstrap 元数据的普通裸请求。

如果 router 透传 native `rid`/`routed_dp_rank` 且已有协议正确连接 P/D，可以试 native 路径，但需要实际验证路由结果。当前仓库不包含任何厂商特定 PD 握手、RDMA 或 FlexKV 初始化代码。

若 D metrics 的 prealloc/transfer 一直不为 0，这个窗口不满足“隔离 Decode”的条件；线上连续到达负载应该使用完整服务基准，而不是放宽规则假装已隔离。

## 扩展代码入口

- `runner.make_body`：构造对应协议请求。
- `sse.Progress`：把完整 SSE frame 解释成生成进度；严禁把字节 chunk 当作 token。
- `metrics.py`：只处理显式定义的 readiness gauge。
- `sglang_hook.py`：框架内部适配；不把此 hook 直接套到 vLLM 或其他后端。
- `tests/test_http.py`：为新协议加入本地模拟测试，覆盖失败和截断场景，再做真实服务验收。
