# 配置与使用指南

在项目根目录执行命令。只浏览原文可用 BM25，不需要下载模型或付费 API。

## 已准备好的本机工作目录

当前工作目录的 `.venv` 已安装基础和语义依赖，`models/multilingual-e5-small` 已下载固定版本权重。直接运行：

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1
```

页面默认打开 CLIP。没有本地 PDF 时，先点击“下载真实论文并开始”，再点击“CLIP 的 ImageNet 准确率”；首次解析后直接显示数值和来源。也可填写方法与指标标签查表，分组指标用完整路径，遇到多个来源自行选择。此流程不需要下载语义权重，不调用生成模型。

右侧单元格检查可以选择行列、核对原页高亮并导出坐标 JSON。表格的分组表头只有具备明确横线几何证据时才绑定；复杂跨列和多行单元格需要人工核查。schema 1/2 索引仍能读取，但需重新 `ingest` 才能得到 schema 3 的表头来源路径。官方 PDF 下载失败可上传自己的文本 PDF，或选择侧栏的虚构示例。

## 在新电脑上准备语义检索

使用 Python 3.11 或 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-semantic.txt
python scripts/prepare_embedding.py
streamlit run app.py --server.address 127.0.0.1
```

Windows 用 `.venv\Scripts\activate` 激活。首次安装 PyTorch 等依赖需要下载较多文件。模型权重约 470 MB，下载脚本默认固定 Hugging Face 提交 `614241f622f53c4eeff9890bdc4f31cfecc418b3`，并写入 `provenance.json`。应用仅加载准备好的本地权重，不会在点击查询时自动下载。

模型放在其他位置时，设置 `PAPER_EVIDENCE_EMBEDDING_PATH`，或在 `config.toml` 的 `[retrieval]` 中填写 `model_path`。下载失败时重新运行准备脚本即可；未准备完成的模型不会被应用使用。

模型使用 [E5 官方模型卡](https://huggingface.co/intfloat/multilingual-e5-small)要求的查询和段落前缀。长片段分成最长 512 token、重叠 64 token 的窗口，实际加前缀后再次检查长度；检索仍返回原片段全文和坐标。表格不会因为模型窗口而被改写或截断为引用证据。窗口最大分数会偏好较长片段，需在真实数据上评测。

`.cache/embeddings/` 保存本地向量缓存；修改文本或模型版本会产生新缓存。关闭应用后删除此目录可清除这些向量，重启可清除进程内上传与解析缓存。模型权重保存在 `models/`，与论文向量缓存分开。

## 云端生成

项目使用兼容 Chat Completions 的第三方服务接口。配置模板以阿里云百炼为例，实际模型权限、JSON 支持、价格和地域应以所选账户的[官方接口文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)为准。

```bash
cp config.example.toml config.toml
```

在本地 `config.toml` 填写 `base_url`、`model` 与 `api_key_env`。然后在启动应用的同一终端设置命名环境变量，例如 `DASHSCOPE_API_KEY`；真实密钥不写入配置文件，也不需要发到聊天中。应用不会自动读取 `.env` 文件。

选择“云端 API 生成”，确认侧栏显示自己的服务和模型，然后查询。请求仅携带系统提示、问题和本次检索片段；整份 PDF、来源图片与本地文件路径不会放入模型请求。请求仍可能产生供应商费用。

命令行示例：

```bash
python -m paper_evidence ingest examples/demo-paper.pdf --out data/demo.json
python -m paper_evidence ask "提出的方法准确率是多少？" --index data/demo.json --retriever hybrid --provider compatible --k 3 --out data/answer.json
```

模板默认限制输出为 1200 token、检索证据为 16000 字符。字符预算不是输入 token 预算，也不是固定金额预算。完整片段超出预算时会提示减少检索数量，不会静默切掉表格。HTTP 错误和超时不会自动重试；人工重新点击可能再次计费。结果 JSON 记录配置模型、服务主机、检索模型版本和供应商返回的 token 用量，不能单独证明实际费用或模型别名背后的精确版本。

其他兼容供应商需要替换服务与模型，并删除 `enable_thinking`；不支持 `response_format` 时设 `json_mode = false`。系统提示仍要求 JSON，并继续校验输出。尚未声称所有兼容服务均已完成验收。

401/403 时检查密钥、服务地域与模型权限；429 时查看供应商额度或限流。回答未完成、引用不存在、数值没有出现在引用中，都会阻止该回答展示。引文通过校验仍可能对应错误表格行，不能当作语义正确的证明。

## 本地生成

安装 Ollama，下载适合硬件的本地模型并启动服务。在侧栏填写已下载的模型名，或运行：

```bash
python -m paper_evidence ask "提出的方法准确率是多少？" --index data/demo.json --retriever hybrid --provider ollama --model YOUR_DOWNLOADED_MODEL
```

当前适配器仅接受本地 HTTP 服务。此工作目录尚未完成真实 Ollama 生成验收。

## 复现验证

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python eval/retrieval_eval.py --out eval/results.json
python eval/semantic_eval.py
python eval/verify_local_embedding.py
```

单元测试用确定性的测试向量和离线 API 响应，不依赖下载模型。两个语义脚本使用真实本地 E5 模型：前者比较相同分块下的三种检索器，后者检查长文本窗口与原来源保留。详情见 [验证记录](VERIFICATION.zh-CN.md)。

真实论文开发集下载命令为 `python scripts/fetch_real_papers.py`；评测请使用单独输出路径，如 `python eval/real_eval.py --out eval/real/v0.5-development-results.json`。NLP 当前开发协议与复现命令见 [本轮报告](TABLE-WORKFLOW.zh-CN.md)。历史固定代码协议会拒绝当前修改后的解析器。
