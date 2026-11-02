# llm-decode-bench

**可审计的混合长度 LLM Decode 测试工具。** 面向已启动的推理服务，默认只测 SSE；可选读取 metrics 验证每个 DP rank 的真实 running BS，也提供独立、显式启用的 SGLang GPU event 插件。

Portable fixed-cohort, mixed-length decode benchmarks. SSE-only by default; optional sampled per-rank readiness and SGLang CUDA/HIP events. No model-specific kernels, private traffic, service restart automation, or GPU requirement on the client.

> **SSE gap ≠ token TPOT ≠ GPU step。** 本工具不按 MTP 接受长度折算时间，也不把 TCP 分包数当作 token 数。精度/接受率/线上吞吐需要单独评测。

## 只分享一个 Python 文件：`decode_sse_bench.py`

**只复制仓库根目录的 `decode_sse_bench.py` 即可运行，不依赖本仓库其他模块。** 不修改服务启动参数、不安装插件、不抓 GPU trace、不清空服务缓存。认证读取 `OPENAI_API_KEY`，客户端不需要 GPU。

```bash
python3 -m pip install aiohttp transformers
python3 decode_sse_bench.py --help
# 可选：内置单测 + 本机 HTTP 模拟测试，不使用模型/GPU
python3 decode_sse_bench.py --self-test
```

### 五种输入模式

| `--mode` | 长度与内容 | 可复现性 |
| --- | --- | --- |
| `0` | 同一测试点的所有请求使用**完全相同的输入**，主动共享完整前缀，类似旧摸高脚本 | 固定公共前缀；不同 BS、rank 使用同一序列，较短输入是该序列的切片 |
| `1` | 每个请求同长度，独立随机 token 前缀 | 固定输入和生成种子 |
| `2` | 每个 rank 长短混合，整数 token 的平均值严格等于指定值，独立前缀 | 固定长度分配、输入和生成种子 |
| `3` | 同样严格保持每 rank 平均值，每次重新随机分配长度和输入 | 记录有效 `run_seed`，可用 `--replay-seed` 重放 |
| `4` | 从真实 JSON/JSONL 影子数据中选取不同的完整文本/对话 | 每次随机选取，记录种子和输入哈希；**不截断、不补文本、不重复凑并发** |

`--input-lengths-k` 在模式 2～4 表示**每个 rank 的平均输入长度**，K=1024。模式 2/3 默认长度范围为平均值的 0.25～1.75 倍，可用 `--min-length-ratio` / `--max-length-ratio` 修改。BS=1 时不可能形成长短混合。

**模式 0 是显式的共享前缀摸高模式，不需要加 `--allow-shared-prefix`。** 例如 `--mode 0 --batch-sizes 16 32 64 --input-lengths-k 4 64 100 150 190 --seed 60`：同一输入长度下，所有请求、所有 BS、所有 rank 的输入 token 都完全相同；100K 输入也是同一条 190K 序列的前 100K。固定 tokenizer 和输入种子可复现这条序列，改变 `--generation-seed` 不会改变输入。各请求仍独立生成输出，生成种子为基准值加请求编号，不能假设输出也共享。

模式 0 不复制旧脚本缺少验证的行为：仍按当前测试点的实际输入长度预热，再逐条检查缓存命中，保留逐 rank 稳态检查和所有并发请求。不会仅预热最长输入，就假定所有较短截断位置也已命中。相同输入不代表不同 GPU 自动共用显存；容量始终按每个 DP rank 分别计算。结果的 `prefix_workload` 标记为 `fully_shared_synthetic`，不要与 mode 1 的独立前缀结果混淆。

模式 4 的自然长度不一定能凑出精确均值：默认容差 1%，`--shadow-mean-tolerance` 可调整；找不到符合要求的样本集合就失败，不会悄悄改内容。真实输入文件放在自选目录：

```json
{"text":"完整的真实请求文本"}
{"messages":[{"role":"user","content":"完整的真实问题"}]}
```

也支持标准 ShareGPT `conversations`。需要足够多的不同请求；暂不接受图像/工具调用数据，避免丢字段改变语义。不会把原始输入或生成内容保存到结果，更不会自动上传影子数据。

仓库现在附带已获源数据所有者公开授权的 [真实影子文本样本](shadow/README.md)：`shadow/basic.jsonl.gz` 有 **512 个脱敏真实内容片段（来自 39 条源请求）**，`shadow/long-context.jsonl.gz` 有 **40 条完整渲染文本**；合计约 **10.58 MB**。512 个片段不是 512 条独立生产请求。模式 4 不传 `--shadow-dir` 时默认读取基础集；也可以指定一个 `.jsonl.gz` 文件，无需解压。基础集约 2K～16K，不能拿来冒充 100K 上下文数据。具体命令、样本来源、长文本数量限制和校验值见数据说明。

### 直接测试已启动的 SGLang

```bash
python3 decode_sse_bench.py \
  --base-url http://127.0.0.1:31000 \
  --backend sglang --model hy4 \
  --tokenizer /data/models/hy4 \
  --dp-size 8 --batch-sizes 16 32 \
  --mode 2 --input-lengths-k 32 64 100 \
  --seed 60 --generation-seed 60 \
  --output-tokens 2048 --ignore-eos \
  --measure-seconds 10 --plot
```

`--ignore-eos` 仅用于已支持该参数的服务，不支持就去掉；输出过早结束时本组失败，不会拿剩余请求冒充原 BS。换模式只需改 `--mode`；模式 4 另加 `--shadow-dir /path/to/traffic`。种子固定的是请求内容和采样参数，**不保证跨后端/批处理顺序的输出逐字一致**。

脚本会依次：构造完整请求集合 → 容量预检 → 每个输入预热 1 token → 再探测全部输入的缓存命中 → 同时发送正式请求 → 所有流开始持续输出后等待 3 秒 → 采集 10 秒公共窗口。默认继续读完输出，校验最终 usage；并不补充新请求。

### 容量不足：默认测到当前 BS 的上限，可显式共享前缀

默认自动读取 SGLang `/server_info` 的**每个 attention DP rank 的 KV token 池容量**；不根据空闲显存推算。多个 scheduler 上报不同容量时取最小值。接口不提供可靠容量时，必须手动指定：

```bash
# 与旧摸高参数一致：这是每卡/每 attention-DP rank 的 token 容量，不是全机总和
--max-total-kvcache-length-k 3500
# 也可以给精确 token 数（两种手动参数不能同时使用）
--kv-capacity-tokens 3584000
```

默认使用容量的 95%，剩余作为安全余量，可用 `--capacity-fraction` 修改。预算包括按 KV page 对齐的输入和 **BS × 输出 token 上限**，不忽略 decoding 期间的 KV 增长。手动值高于可探测的服务上限时取较低值。`--page-size` 可以覆盖服务上报的 KV page token 数；没有上报时默认保守按 256 计。

例如独立输入、每卡 BS32、100K 上下文、2048 输出，输入+输出约需 **3264K token/卡**，再和安全预算比较；BS64 约需 **6528K/卡**。这和所有请求共用一条 100K 前缀的显存需求不同。

**不加 `--allow-shared-prefix` 时，模式 1～3 会按每个 BS 自动计算输入长度上限。** 预算包含安全余量、输出增长和 KV 页对齐；已知服务上下文上限时也会扣除输出。脚本将输入列表按长度排序并去重，测试范围内的点；当列表中有点超限时，自动补测一个边界点（如果列表里尚未包含），随后结束当前 BS，继续下一个 BS。如果指定的最大输入本来就在范围内，则只测原列表，不会为了摸高擅自扩大输入。

例如每卡容量 3500K、默认 95% 预算、BS64、输出上限 8192 token、KV page 256 token：

```text
可用预算：3325K
独立同长输入上限：floor_to_page(3325K / 64 - 8K) = 43.75K
输入列表：0.5 1 2 4 8 16 32 64 100 128 160 190 200
实际测试：0.5 1 2 4 8 16 32 43.75
随后进入下一个 BS；64K～200K 不会发送生成请求，也不构造共享前缀。
```

模式 1 的 KV 边界是同长输入的精确页对齐边界。模式 2／3 为每个请求预留最坏的末页浪费，并按最长输入比例检查上下文，因此自动补入的是**保守的平均长度边界**，不同 rank 仍保持相同目标均值。模式 4 不裁剪或补写真实输入；若实际选出的真实请求组超出 KV 预算，就记录 `capacity_limited`、结束当前 BS，并继续下一个 BS。

模式 0 同样自动测到容量/上下文边界，但**不按 `BS × 输入长度` 计算**。按每卡一份共享输入，加上每条请求的私有尾页和输出预留：

```text
模式0每卡预算 = floor((输入token数 - 1) / page) × page
                + BS × (page + ceil(输出token数 / page) × page)
```

即使输入恰好页对齐，也预留每条请求重算/复制最后一页的空间。不假设生成结果相同；不忽略 BS 增大时的输出容量。这个预算是 Full KV 规划，不保证 SWA 等其他池一定够用，运行时缓存校验仍必须通过。加 `--allow-shared-prefix` 不改变 mode 0 的逻辑。

某个 BS 连输出预留和最小输入都放不下时，会记录为 `capacity_limited` 并继续其他 BS；如果整个矩阵没有任何可运行组，返回非零退出码。网络失败、缓存未命中、请求提前结束、稳态超时、无效 tokenizer 等运行错误仍会中止测试，不能当作容量边界吞掉。

如果希望保持列表中指定的长输入，即使独立前缀装不下也继续尝试，可显式加：

```bash
--allow-shared-prefix
```

加了该选项就保留原始输入列表，不执行上述自动长度截断。只有当前组的独立前缀放不下时才启用共享：同一个 rank 的请求共享部分前缀，仍保留独立后缀（默认至少 128 token，`--min-unique-suffix-tokens` 可调），长度和均值保持不变。混合长度采用嵌套共同前缀，使短请求和长请求都能分享已有部分。尽可能少共享到满足预算；即使最大共享仍装不下，也会退出。

**共享前缀会改变显存需求、访问局部性和工作负载，不能把它与独立前缀结果当作同一个基准。** 结果明确记录 `prefix_workload` 和每个请求的共享长度。模式 4 只利用真实内容自然具有的相同前缀，即便加了该参数，也不会人为改写真实请求来省显存。

想先看容量计划、不发送任何生成请求，加 `--plan-only`。容量预估不是运行时 OOM 保证：其他流量、临时激活、路由、驱逐和不同后端实现仍可能导致失败。本脚本不是 HiSparse 热 KV/CPU offload 容量测量工具。

### 什么条件才算结果可信

- 缓存：默认要求探测和正式请求都报告 `cached_tokens`，至少命中 98%，且未命中尾部不超过 256 token，两项同时满足。允许少量尾页重算，不宣称 100% 命中；可用 `--min-cache-hit-ratio` 和 `--cache-tail-tokens` 收紧。已知未命中直接失败；接口缺少计数时，仅显式 `--allow-unverified-cache` 才允许继续，并标为 `unverified`。
- 每卡 BS：SGLang native 默认用 `routed_dp_rank` 固定请求归属；如果服务没有该扩展，可加 `--no-route-dp`，但不能再保证每 rank 的长度分布和 cache affinity。通用 OpenAI 接口同理。
- 稳态：默认尝试读取 metrics，要求所有 rank 的 running 都等于指定 BS、等待队列为 0；缺少指标会明确降级为 `client_only_unverified`。**要强制每卡 BS 验证，加 `--steady-check required`**。可通过 `--metrics-url`、`--running-metric`、`--waiting-metric`、`--rank-label`、`--metric-label` 适配现有服务；全局 running 指标不能冒充多个 rank 的指标。
- PD：从已部署的完整 router/P-D 入口发请求，不能直接把普通请求发给仅 D 的端口来代替 bootstrap/KV 传输。metrics 需要指向 D；`--pd-metrics` 额外要求 prealloc/transfer 队列清空。路由器未透传 DP pinning 时需关闭 pinning；跨机器/实例前缀不会因客户端预热自动共享。
- SSE：统计完整生成进度事件之间的间隔，忽略 keepalive、纯 role、usage 和 DONE；不是 TCP 包间隔。MTP 或 `stream_interval` 可能使一次事件含多个 token，也可能把一步拆为多次发送；**不保证一次 SSE 就是一次 GPU step**。

OpenAI-compatible completions 可选 `--backend openai`，需要支持 token-ID `prompt`。仅有 chat 接口的服务使用 `--backend openai-chat --mode 4`，本地 tokenizer/chat template 必须与服务一致；如果服务报告的 prompt token 数不一致就失败，不用字符数冒充 token 数。脚本不保证任意厂商接口都能验证缓存/每卡 BS。

### 输出文件

每次生成一个 `results/sse_日期时间_随机标识/`：`config.json`、`sweep_plan.json`、`results.json`、`summary.csv`；加 `--plot` 生成 `sse_gap.svg`。图为每请求 SSE gap 中位数的中位数；CSV 同时包含均值和合并间隔的 P99。不除以 MTP 接受长度，也不计算模型精度。

**边测边保存、边测边画图**：每完成并校验一个 `BS × 输入长度` 测试点，就更新 JSON、CSV；加 `--plot` 时同时更新同一个 `sse_gap.svg`，不必等整套矩阵结束。第一个有效点完成后即可打开 SVG，后续点完成后刷新查看；SVG 文件本身不自动刷新。保存发生在稳态计时之外，不把写盘或绘图耗时混入 SSE 间隔。

开始下一点的预热前也会保存快照，当前点标为 `running`；失败为 `failed`，正常处理的取消/中断为 `interrupted`。图中只包含 `ok` 点，未完成的点不会当作有效结果。正常结束、异常和可处理的中断仍会再保存一次。文件通过同目录临时文件原子替换，避免写到一半截断上一版；原子性针对单个文件，不是四个文件的整体事务。`kill -9` 或断电不能触发最终保存，但之前已保存的完整文件仍可保留；不能保证当前点或最新一轮各文件同时更新。

`config.json` 保留原命令；`sweep_plan.json` 逐 BS 记录理论边界、实际测试点、自动添加点和省略的原始点。`results.json` 和 CSV 的 `mean_input_tokens_requested` 表示该组**实际采用的目标长度**，`input_length_origin` 区分原列表点和自动边界，`theoretical_input_limit_tokens` 给出当前 BS 的上限。不会把 43.75K 的结果标成原命令里的 64K 或 200K。

默认排空输出以拿到最终缓存 usage，采样结束后可能还要等一段时间。`--stop-after-measure` 只关闭本脚本的 HTTP 流，不清缓存、不全局 abort；不同服务可能继续后台生成，OpenAI 最终 usage 也可能缺失，因此通常不建议在连续矩阵中使用。

## 1. 能做什么

- 同长输入、长短混合输入、共享长前缀，使用同一套测试流程。
- 固定请求集合：一次发出 `dp_size × batch_size` 个请求，不补新请求；只分析所有请求共同持续输出的窗口。
- 单组或 BS 矩阵测试，例如每个 DP rank 的 BS 8、16、17、18、19、32、64。
- 默认 SSE-only：无需修改服务、安装 GPU profiler 或重启模型。
- 严格 metrics 模式：检查所有 DP rank 的 running BS、等待队列；PD 场景额外检查 prealloc/transfer。
- 可选 GPU event：测量 `Scheduler.run_batch` 的 forward-stream 时间区间；同时记录两轮之间的 stream gap。
- 可复算 JSON/JSONL、输入 SHA256 清单、A/B 可比性检查、SVG 图。

**不做什么：** 不加载模型，不改算子，不停止别人的进程，不清缓存，不自动重启服务，不管理容器，不实现 PD 的 bootstrap/KV 传输，不把合成输入的回答当作模型质量验证。

## 2. 安装

Python 3.10+；客户端不需要 GPU、PyTorch、ROCm、CUDA、SGLang。

```bash
git clone https://github.com/doulujiyao12/llm-decode-bench.git
cd llm-decode-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
llm-decode-bench --help
```

只有生成严格长度的合成 token 输入时才需要 tokenizer 依赖：

```bash
python -m pip install -e '.[tokenizer]'
```

认证通过环境变量传递，不保存到结果：

```bash
export OPENAI_API_KEY='你的服务 API key'
```

也可以用 `--api-key-env MY_API_KEY` 指定已有变量名。不把 key 写进 URL 或提交到 Git。客户端默认不读取系统 HTTP 代理；跨机器请使用实际可达地址。

## 3. 先验证一个已经运行的服务（不用 GPU event）

OpenAI-compatible **chat** 接口：

```bash
llm-decode-bench run \
  --base-url http://127.0.0.1:8000 \
  --backend openai-chat \
  --model your-served-model \
  --dataset examples/prompts.jsonl \
  --dp-size 1 --batch-size 4 \
  --output-tokens 2048 \
  --output results/chat-smoke
```

`--base-url` 不带 `/v1`；工具自动选择 `/v1/chat/completions`、`/v1/completions` 或 `/generate`。如果服务用路径前缀，例如 `/inference`，可以在 base-url 中保留此前缀。

这个示例的 4 条短文本只用于接口检查，模型可能很快 EOS，导致没有完整稳态窗口。工具会返回非零退出码并说明原因，不会硬凑一个结果。需要长时间 Decode 时，增加输出上限，或在服务确实支持时显式增加 `--ignore-eos`。

默认 `--steady-mode client` 只确认客户端所有请求持续收到生成进度，结果标记为 **`client_only_unverified`**。这不证明每卡真实 BS，也不证明期间完全没有 prefill；要验证这些条件请使用下一节。

## 4. 构造同长/混合长度数据

### 4.1 严格 token 数的合成数据

使用与服务权重**完全匹配**的 tokenizer，排除其 special token IDs。不会随机猜测词表范围，不会启用 `trust_remote_code`。

```bash
# 512 条独立输入，长度从 8K/32K/64K/100K 中等概率抽取
llm-decode-bench generate \
  --tokenizer /path/to/model-or-tokenizer \
  --lengths 8192 32768 65536 102400 \
  --count 512 --seed 60 \
  --output data/mixed.jsonl

# 同长对照；必须使用另一个结果标签，不能宣称与 mixed 是完全相同工作负载
llm-decode-bench generate \
  --tokenizer /path/to/model-or-tokenizer \
  --lengths 98304 --count 512 --seed 60 \
  --output data/uniform.jsonl
```

长度选择独立于 DP rank，避免按 `i % 长度种类数` 分配导致某些 rank 全是长输入。较小 BS 读取相同数据集的前 N 行，构成嵌套请求集合；每种长度的比例是抽样结果，不保证每个小 batch 恰好均分。

合成 token 序列适合性能诊断，不是自然语言质量或 MTP 接受率数据。它们不会代表线上接受长度。跨模型比较必须重新使用各自 tokenizer 生成，并明确数据不同。

### 4.2 共享长前缀

```bash
llm-decode-bench generate \
  --tokenizer /path/to/model-or-tokenizer \
  --lengths 102400 --count 256 --seed 60 \
  --shared-prefix-tokens 100352 --prefix-families 1 \
  --output data/shared-100k.jsonl
```

共享前缀必须小于最短输入，剩余后缀独立生成。`--prefix-families 8` 可产生 8 个前缀家族。设置共享前缀**不等于**保证物理 KV 共享或 cache 命中：是否启用 prefix cache、路由落点、容量、驱逐、HiSparse/PD 限制都由服务决定。共享前缀也不能完整代替独立真实流量。

### 4.3 自己的真实数据

JSONL 每行二选一：

```json
{"text":"Explain how a binary search works."}
```

或者：

```json
{"input_ids":[101,102,103,104]}
```

- token IDs 只适用于 native SGLang 和支持 token-ID prompt 的 completions 接口；chat 模式拒绝 token IDs，避免 decode→encode 改变长度。
- text 输入实际 token 数由服务 tokenizer/chat template 决定；工具不会用字符数冒充 token 数。
- 至少准备 `最大 BS × DP size` 行。数据不足直接报错，不会自动重复到目标并发。
- `data/`、`results/` 被 `.gitignore` 排除。不要把原始业务请求、模型权重或用户对话提交到本仓库。

## 5. 严格检查 DP BS 的 SSE 测试

下例针对 native SGLang；需服务暴露所有 scheduler 的 metrics，且支持 `routed_dp_rank`。

```bash
llm-decode-bench run \
  --base-url http://127.0.0.1:8000 \
  --backend sglang --model your-served-model \
  --dataset data/mixed.jsonl \
  --dp-size 8 --batch-size 32 --route-dp \
  --output-tokens 4096 --temperature 0 --ignore-eos \
  --warmup --warmup-concurrency 8 \
  --steady-mode metrics \
  --metrics-url http://127.0.0.1:8000/metrics \
  --metrics-preset sglang \
  --output results/baseline-mixed-bs32
```

- 此处全局请求数是 `8 × 32 = 256`，不是 TP size 乘以 BS。
- `--route-dp` 按 `请求索引 % DP size` 路由，同时要求响应的 `meta_info.dp_rank` 与目标一致；不支持该扩展的版本会失败，而不会悄悄取消路由。
- `--warmup` 先对相同输入逐批请求 1 个输出 token；随后才启动正式 cohort。预热只是尝试减少正式 prefill，不能保证缓存未被驱逐。
- `--ignore-eos` 是性能压测选项；强制 EOS 之后继续生成可能没有语义，不应据此评价精度。
- metrics 缺 rank、缺字段、重复序列、采样过稀或队列不为空，该窗口不计入结果。

不支持显式 DP 路由时去掉 `--route-dp`，仍可用 metrics 检查服务是否实际均衡。若不能达到每个 rank 的目标 BS，结果会无有效窗口，不会将全局并发平均分配后假设已经满载。

### BS 矩阵

```bash
llm-decode-bench sweep \
  --base-url http://127.0.0.1:8000 \
  --backend sglang --model your-served-model \
  --dataset data/mixed.jsonl \
  --dp-size 8 --batch-sizes 8 16 17 18 19 32 64 --repeats 2 \
  --route-dp --output-tokens 4096 --ignore-eos --warmup \
  --steady-mode metrics --metrics-url http://127.0.0.1:8000/metrics \
  --output results/mixed-matrix

llm-decode-bench plot results/mixed-matrix/bs*-r1 \
  --output results/mixed-matrix/summary.svg
```

按顺序测试，不同时叠加多个 BS。发生请求失败即停止后续矩阵，避免继续给异常服务施压。每个 case 输出目录必须不存在；重复测量请用新目录。

## 6. 结果怎么读

```text
results/baseline-mixed-bs32/
├── config.json          # 客户端参数、版本；不存 key 或 extra-body 原文
├── manifest.json        # 输入 hash、token 长度（已知时）、目标 rank
├── requests.jsonl       # 请求成败、进度时间点、可用的 token 计数；不存输入/生成文本
├── metrics.jsonl        # 仅保存 readiness 字段，不存全部原始 metrics
├── capture.jsonl        # 可选 GPU event 的 arm/done/failed 时间
├── analysis.json        # 每个候选窗口、拒绝原因、汇总
├── gpu_analysis.json    # 仅启用 GPU event 时存在
└── ldb_…-events.jsonl   # 仅启用 GPU event 时存在
```

默认等每个请求收到至少 5 次有效生成事件，前后各剔除 3 秒，再对所有请求的共同区间按 5 秒分窗。每个请求在一个窗口内至少有 20 个 gap 才接受该窗口。严格模式还要求每个窗口有新鲜、成功、首尾覆盖的 metrics，且所有 rank 达标。

| 字段 | 定义 |
| --- | --- |
| `sse_gap_p50_ms` | 每请求 gap 中位数 → 同窗口跨请求中位数 → 跨有效窗口中位数 |
| `sse_gap_p95_ms` | 每请求 gap P95 → 同窗口跨请求中位数 → 跨有效窗口中位数；不是所有样本合并的 P95 |
| `mean_tokens_per_progress_event` | native SGLang 报告 token 增量时的平均值；不保证等于真实 MTP accept length |
| `gpu_step_p50_ms` | 可选插件：每 rank 去掉头尾 2 次后的 GPU event 区间中位数，再跨 rank 取中位数 |
| `gpu_gap_before_p50_ms` | 同一 forward stream 上前一次 end event 到下一次 begin event 的间隔；不是“GPU 完全空闲”证明 |

**所有时间均不除以 MTP 接受长度。** SSE 可能受网络聚合、输出缓冲、CPU detokenization、发送策略影响。只有确认一次进度 SSE 对应一轮 Decode 时，才可把它作为 step 的代理；需要直接 GPU 时间时启用下一节插件。

```bash
llm-decode-bench analyze results/baseline-mixed-bs32
llm-decode-bench compare \
  results/baseline-mixed-bs32 results/candidate-mixed-bs32 \
  --output results/compare-bs32.json
llm-decode-bench plot \
  results/baseline-mixed-bs32 results/candidate-mixed-bs32 \
  --output results/compare-bs32.svg
```

A/B 会检查输入 manifest 和主要测量参数；不一致就拒绝计算“优化收益”。仍需人工确认模型权重、量化、KV dtype、MTP 参数、graph bucket、服务版本、GPU 时钟/功耗及竞争任务一致。`--label candidate` 只记标签，不会替你切换代码或服务。建议 A/B/A 多轮复测，不能把一次小波动当成确定收益。

## 7. 可选 GPU event（默认关闭）

详见 [GPU event 使用与边界](docs/GPU_EVENTS.md)。简要流程：

1. 在**服务环境**安装本包；客户端和服务共享一个专用 telemetry 目录。
2. 服务启动前把本仓库 `hooks/` 加入 `PYTHONPATH`，设置 `LDB_EVENT_DIR` 和 leader TP ranks。
3. 按原参数启动 SGLang；不需要修改模型、量化、并行参数。
4. 客户端增加 `--event-dir`，稳态后自动发起有限轮次 capture。

```bash
# 以下在服务环境执行。REPO 必须是本仓库绝对路径。
REPO=/path/to/llm-decode-bench
export LDB_EVENT_DIR=/path/to/telemetry/experiment-001
mkdir -p "$LDB_EVENT_DIR"
# TP8/DP8 attention 场景通常每个 TP rank 对应一个独立 attention DP rank。
export LDB_EVENT_LEADER_TP_RANKS=0,1,2,3,4,5,6,7
export PYTHONPATH="$REPO/hooks:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# 随后执行你原本的服务启动命令，其他参数不变。
```

客户端在第 5 节 run 命令上增加：

```bash
--event-dir /path/to/telemetry/experiment-001 \
--event-steps 128 \
--event-timeout 600
```

这些参数是命令的续行，不是单独执行的 shell 命令。服务已启动但没有加载 hook 时，客户端不能“补上” GPU event：默认 SSE 仍可测，event 模式会明确失败。无需 event 时完全不加 `--event-dir`。

## 8. 换 GPU、模型、框架、PD 怎么用

| 场景 | 需要调整 |
| --- | --- |
| AMD / NVIDIA / 其他加速卡 | SSE 客户端没有硬件绑定。GPU event 仅面向 PyTorch CUDA/HIP 的兼容 SGLang 版本；其他平台只保证客户端层可用 |
| 新模型 | 修改 model、tokenizer、输入长度和输出上限；重新生成 token 数据，确认服务上下文容量 |
| vLLM / OpenAI-compatible 服务 | 选择 `openai` 或 `openai-chat`。metrics 使用自定义映射；本仓库不提供 vLLM scheduler hook |
| 纯 TP，无 attention DP | `--dp-size 1`，BS 是全局请求数；event 只选 TP leader，例如 `0` |
| attention DP | `dp-size` 指独立 attention scheduler 数，不能用 GPU 数机械代替 |
| PD 分离 | 请求发到正常的 PD router/gateway；metrics 指向 D。不要直接往只接受带 bootstrap 信息的 D 发送普通请求 |
| HiSparse / KV offload | 不自动开关该功能。相同输入/BS/上下文分别测试，检查 GPU 热池、CPU 池和 admission 是否改变实际 running BS |

对于 PD，`--metrics-preset sglang-pd` 会要求 running、waiting、prealloc、transfer 全部存在且队列归零。持续线上影子流量经常没有这种全空队列窗口；本工具主要做**固定 cohort 的 Decode 归因**，不是开放式 QPS 线上吞吐测试替代品。

跨机器用 router URL 发送请求、D metrics URL 读取状态；GPU event 的控制目录必须真正共享或挂载到同一存储。只有 URL 可达而没有共享目录时，不能使用此自动 event capture；可使用 SSE-only。

## 9. 容量和稳态注意事项

- 逻辑请求数为 `DP × BS`；MTP 增加的是每轮验证/草稿 token 数，不增加这里定义的请求 BS。
- 输入长度 + 输出上限必须小于模型/服务上下文限制。先从小 BS 和短输入验证。
- 大量独立长输入可能耗尽 KV 池或导致反复抢占；不要仅凭设置了最大并发就假定能满载。
- prefix warmup 不能消除所有 prefill。不开 prefix cache、缓存被驱逐、路由不同或 KV offload 都可能使其无效。
- 即使每 rank running 恰好达标，轮询 metrics 也不能证明两次采样间完全没有调度变化；高置信归因应同时检查 hook 请求集合/上下文和必要的 trace。
- 输出太短、某个请求很早结束、长短请求 TTFT 相差过大，会没有共同窗口。增加输出上限、减少并发、启用已支持的 prefix cache，或修正负载后重测；不要放宽条件后仍把结果叫严格同 BS。
- `--request-timeout` 默认 7200 秒，固定 cohort 等请求自然结束；客户端退出/超时不保证每个框架立即取消服务端请求。工具不调用全局 abort/flush/kill 接口。
- 生成大规模 token JSONL 与加载 cohort 消耗客户端 CPU/RAM。超长上下文先估算客户端内存和服务 KV 容量；本版本将一组 cohort 输入读入内存。

更多： [测量方法与限制](docs/METHODOLOGY.md)、[框架适配](docs/ADAPTERS.md)、[复现与发布说明](docs/VALIDATION.md)。

## 10. 开发测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q src hooks tests examples
```

测试包含本地 HTTP/SSE 模拟服务，不启动真实模型，也不需要 GPU。GPU hook 用 fake event 验证延迟 query、无 synchronize、调用原函数一次及异常传播；**这些测试不等于新硬件/新 SGLang 版本上的 GPU 验收**。

本仓库的首次通用化版本没有在生产 GPU 上重新跑一遍所有矩阵，不继承此前模型专用实验的性能/精度结论。迁移到新环境后应先验证短输入、短 capture，再做正式 A/B。

### 无 GPU 的完整流程演示

终端 1（只监听 localhost，不是模型服务）：

```bash
python examples/mock_server.py --port 8765 --dp-size 2
```

终端 2：

```bash
llm-decode-bench sweep \
  --base-url http://127.0.0.1:8765 --backend sglang --model mock \
  --dataset examples/mock-prompts.jsonl --dp-size 2 --batch-sizes 1 2 \
  --route-dp --ignore-eos --output-tokens 160 \
  --steady-mode metrics --metrics-url http://127.0.0.1:8765/metrics \
  --metrics-interval 0.1 --window-seconds 0.5 --guard-seconds 0.1 --min-gaps 5 \
  --output results/mock-matrix

llm-decode-bench plot results/mock-matrix/bs*-r1 --output results/mock-matrix/demo.svg
```

mock 的 token IDs 仅为协议演示，不能用于真实模型。图上的时间是模拟器 sleep 造成的，不是 GPU 性能。

也提供方便的环境变量脚本用于真实 native SGLang 矩阵：

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=your-served-model DATASET=data/mixed.jsonl DP_SIZE=8 \
BATCH_SIZES='8 16 17 18 19 32 64' REPEATS=2 RUN_LABEL=baseline \
bash scripts/run_mixed_matrix.sh
```

该模板开启严格 metrics、显式 DP 路由、warmup 和 ignore EOS；服务不支持这些扩展时请改用前面的基础 CLI 组合，不要直接套用模板。
