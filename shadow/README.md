# 真实影子流量文本样本

源数据所有者已于 **2026-11-01 授权公开发布**本数据包。两份文件随 Git 同步，合计约 **10.58 MB（10.09 MiB）**，不需要 Git LFS 或额外下载。它们保存 UTF-8 文本，不包含 tokenizer、模型权重、原始 HTTP 认证或服务器配置。

| 文件 | 内容 | 数量 | 参考 HY4 内容长度 |
| --- | --- | ---: | --- |
| `basic.jsonl.gz` | 规则脱敏后的真实内容连续片段 | 512 个不同片段，来自 39 条源请求 | 约 2K / 4K / 8K / 16K，每档 128 个；均值约 7.5K |
| `long-context.jsonl.gz` | 40 条源请求的完整渲染文本，移除模型控制 token 并规则脱敏 | 40 条不同文本 | 565～280,685 token，均值约 116,661 |
| `manifest.json` | 来源、构造方式、处理次数、文件和内容 SHA256、大小、参考长度 | — | — |

**512 个片段不是 512 条独立线上请求。** 原始冻结工作负载只有 40 条源请求，采集于 2026-09-09。39 条足够长的源请求用于基础集；最短的一条仅保留在长上下文集。片段之间可能相关、重叠或来自相同对话，因此不能代表完整线上流量分布。

## 如何构造

1. 对冻结数据的每条 `prefix_ids + tail_ids` 使用源服务器已有 HY4 tokenizer 解码；该 tokenizer 不随数据包发布。
2. 先处理完整文本：移除模型控制 token；替换匹配到的凭据、密钥块、Bearer、邮件地址、URL、IP、电话号码、UUID、长十六进制标识和私有目录路径。
3. 将处理后的完整文本保存到长上下文集。它是渲染后的 prompt 文本，包含对话/工具描述的文字；不是原始 `messages/tools` HTTP 请求结构。
4. 对处理后的文本重新编码，以固定种子 `20261101` 抽取 2,048 / 4,096 / 8,192 / 16,384 token 的连续窗口，解码、去重后各保留 128 个片段。片段是裁剪内容，不承诺句子、代码块或对话边界完整。
5. 没有追加 nonce、合成段落、重复填充或拼接多个源请求来制造样本。每行保留 `source_id`、片段区间、参考长度和文本 SHA256；原始源请求正文、服务器地址和 tokenizer 文件不发布。

采用 gzip `mtime=0` 保存。`manifest.json` 的 SHA256 可以校验 Git 同步得到的文件是否一致；参考 token 数包含文本本身，不含目标模型新加的 chat template。使用其他 tokenizer 时，实际长度必须由客户端重新计算。

## 单文件测试入口

复制 `decode_sse_bench.py` 和整个 `shadow/` 目录即可。无需解压，模式 4 默认选择基础集；显式传文件可以避免把基础集和长上下文集混在一起。

```bash
python3 -m pip install aiohttp transformers

# 先以小规模确认接口、容量和目标 tokenizer 的长度；不发送生成请求
python3 decode_sse_bench.py \
  --base-url http://127.0.0.1:31000 \
  --backend sglang --model your-served-model \
  --tokenizer /path/to/matching-tokenizer \
  --mode 4 --shadow-dir shadow/basic.jsonl.gz \
  --dp-size 1 --batch-sizes 4 --input-lengths-k 6 \
  --min-length-ratio 0.1 --max-length-ratio 4 \
  --shadow-mean-tolerance 0.05 \
  --max-total-kvcache-length-k 3500 \
  --plan-only
```

**`--dp-size` 必须与实际服务的 attention DP 数一致。** 上例假设服务 DP=1；DP8 服务改为 `--dp-size 8`。手动 KV token 容量也应使用实际服务的值；如果能自动探测，就去掉手动容量参数。

确认计划后去掉 `--plan-only`，按需要加上 `--output-tokens 2048 --ignore-eos --measure-seconds 10 --plot` 执行。`--ignore-eos` 只适用于支持它的服务。要验证真实每-rank running BS，加 `--steady-check required` 并配置现有 metrics。

长上下文集改为：

```bash
--shadow-dir shadow/long-context.jsonl.gz
```

40 条文本不足以填满全机 256 个互不重复的请求，且不同目标均值下符合长度范围的样本可能更少。脚本会报错，不会循环复用补到目标并发。大并发、64K/100K 等指定均值测试，应另准备数量和长度足够的真实数据，或使用模式 1～3 的明确标注的合成输入。

基础集的 512 条只保证总样本数；指定均值、长度上下限、目标模型 tokenizer 和每-rank 分组约束仍可能使可选数量不足。脚本打印重新分词后的长度统计，按这些统计设置均值和范围，实际每-rank 均值保存在结果中。

## 原有包入口

原有 `llm-decode-bench run/sweep` 同样可直接读取压缩文件：

```bash
llm-decode-bench run \
  --base-url http://127.0.0.1:31000 \
  --backend openai-chat --model your-served-model \
  --dataset shadow/basic.jsonl.gz \
  --dp-size 1 --batch-size 4 \
  --output-tokens 2048 --ignore-eos --warmup \
  --output results/real-text-smoke
```

包入口按文件顺序取样，单文件入口模式 4 会按目标均值选样；两者的请求集合和选样方式不同，应分别记录。该数据包用于输入、缓存和性能流程复现；不提供答案标签、模型质量分数或线上 MTP 接受率结论。
