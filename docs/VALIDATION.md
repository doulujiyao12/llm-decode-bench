# v0.1.0 验证与交付说明

日期：2026-10-30。

## 2026-11-01：单文件入口与公开授权样本

新增 `decode_sse_bench.py` 单文件入口，包含四种输入模式、自动/手动每-rank KV 容量预检、显式共享前缀回退、输入预热与缓存命中探测、公共 SSE 稳态窗口和 SVG 图。默认不启用 GPU trace，也不要求服务插件。新增 gzip 读取，包入口和单文件模式 4 都能直接读取随仓库附带的压缩文本。

源数据所有者已授权公开发布 `shadow/`：512 个不同脱敏内容片段（来自 39 条源请求）以及 40 条完整渲染请求文本，压缩合计 10,579,092 字节。解码只在已有源服务器上进行，没有复制或发布模型/tokenizer。数据来源和处理方式见 `shadow/README.md` 与 `manifest.json`。

当前 CPU 验证包括单文件内置 **32 项测试**，以及原包与数据文件 **52 项回归测试**。新增数据检查覆盖 gzip/JSONL 解码、压缩和解压内容哈希、逐样本文本哈希、不同片段数、来源数、数据不够时不循环补齐，以及按参考长度构成 DP8×BS16/32/64 cohort。这些检查不等同于模型 GPU 推理或真实 tokenizer 的跨模型验收。

```bash
python decode_sse_bench.py --self-test
python -m unittest discover -s tests -v
```

## 验证范围

这次交付将固定 cohort 的混合长度测试整理为独立工具，并移除了特定模型、服务器路径、私有语料、容器控制和 kernel 补丁依赖。**没有启动或修改现有推理服务，没有上传业务原始流量。**

本地使用 Python 3.11.6、aiohttp 3.14.5 执行 **48 项单元和模拟 HTTP 测试，全部通过**。仓库 CI 另配置 Python 3.10/3.12；具体执行结果以 GitHub Actions 为准，配置 CI 不代表已成功运行。

| 检查 | 方法 |
| --- | --- |
| SSE 边界 | 单字节切分、UTF-8、CR/LF/CRLF、多行 data、heartbeat、截断、大小限制 |
| 进度口径 | native token 增量、chat reasoning/content、忽略 role/usage/终止帧、不除 MTP |
| metrics | 精确 rank/BS、队列、缺字段、重复序列、筛选标签、缺样和首尾覆盖 |
| 数据 | 固定 seed、嵌套 cohort、共享前缀、合法数据形状、不循环补齐不足的数据 |
| 结果 | 失败请求不能剔除、无完整稳态窗口判无效、输入/配置不一致拒绝 A/B |
| 可选 event | fake event 的延迟 query、无 synchronize、调用原方法一次、保留原模型异常 |
| event 控制 | 模拟服务控制文件握手、每 rank 文件归档、请求集合审计、capture 不完整判无效 |
| 本地 HTTP | SGLang native、OpenAI completions/chat、坏路由、HTTP 503、截断流 |
| CLI | sweep → analyze → compare → SVG plot 完整串联 |
| 隐私 | API key 不落盘；错误正文和生成文本不落盘；只发布通用代码/合成样例 |

运行方式：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m compileall -q src hooks tests examples
bash -n scripts/run_mixed_matrix.sh
```

## 尚未宣称验证的内容

- 通用化版本在 AMD/NVIDIA 真 GPU 上的 hook 开销与兼容性。
- 任意新版 SGLang 的内部 scheduler 字段兼容性。
- 各模型真实输出精度和 MTP 接受率。
- 真 PD router 对 native rid/DP 路由的透传行为。
- 线上 QPS、RDMA、KV offload、长期稳定性。
- `transformers` 对任意模型 tokenizer 的完整兼容性；生成逻辑使用 mock tokenizer 测试，真实 tokenizer 需用户在目标环境验证。

这些边界不能用 CPU mock 测试代替。以前专用实验的数据也不直接算作此仓库版本的硬件验收。

## 给同事复现

1. clone 仓库，在客户端虚拟环境安装；不改模型服务环境即可使用 SSE。
2. 用服务对应的 tokenizer 生成自己的输入，或准备自己有权限使用的 JSONL；不要复用其他模型的 token IDs。
3. 小并发接口检查后，再逐级提升 BS 和输入长度，确认容量足够。
4. 能拿到每 rank metrics 时用严格模式；拿不到时报告必须保留 `client_only_unverified` 标识。
5. 需要 GPU 时间时，单独按 GPU event 文档在维护窗口给服务加载 hook；先验证状态/短 capture，再正式 A/B。
6. 保存服务版本/启动参数/硬件状态到自己的实验记录，分享结果前审查 `config.json` 中的服务地址和模型名。

## 发布内容

`src/` 是通用工具，`hooks/` 是显式启用的 adapter，`scripts/` 是可参数化入口，`examples/` 仅含公开合成示例，`tests/` 是无 GPU 回归，`docs/` 为中文操作和边界说明。

`.venv/`、缓存、构建产物、生成的数据、压测结果、trace、认证文件均不跟踪。上传前应以 `git ls-files` 为发布清单，不使用包含整个工作目录的无过滤打包。
