# PaperEvidence 科研论文证据助手

这是从“多模态科研论文问答助手”想法开始实现的早期原型。当前流程是：上传 PDF，用关键词、语义或融合检索寻找证据，查看页码和原文位置，再选择是否让本地或云端模型归纳回答。

## 已实现

- 文本型 PDF 解析、常见双栏阅读顺序恢复、网格与横线数值表格提取。
- 简单表格的行名、表头、数值单元格绑定，来源区域高亮与坐标导出。
- 按常见章节标题、页码、列和内容类型分块；长表格按完整行拆分，重复表头与表注，保留来源坐标。
- BM25 关键词检索，无需 API Key 的原文证据浏览模式。
- 本地 multilingual-e5-small 多语言语义检索、BM25 与语义检索的排名融合。
- 长文本重叠 token 窗口、模型版本记录与向量缓存，检索保留原片段页码和坐标。
- Streamlit 上传、问答、无需提问的逐页浏览、来源区域高亮和 JSON 导出。
- 可选本地 Ollama 或云端兼容 API 生成；逐条校验引用片段、原文引文和数值出现。
- 云端调用证据预算、输出 token 限制与用量记录；拒绝未完成或格式异常的回答。
- 分块对照、中英配对检索评测、真实论文开发集、固定代码的 NLP 扩充检查、自动化测试和 CI 配置。

## 尚未实现

扫描件 OCR、图像理解、公式识别、跨页表格合并、BGE 重排、校准后的拒答策略、真实论文上的答案准确率和幻觉率评测。云端接口通过了离线测试，但尚未使用真实密钥验证模型生成。

简单表格默认第一行是表头、第一列是行名；合并、多行或复杂表头不能可靠绑定。已开始真实论文检索评测，新增固定代码的 NLP 扩充检查，仍未独立人工复核。

检索分数不是“答案正确概率”。引文存在也不等于结论得到原文支持。当前不宣称准确率提升或幻觉率下降。

## 启动

推荐 Python 3.11 或 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows 激活虚拟环境使用 `.venv\Scripts\activate`。页面默认使用本项目原创的虚构示例 PDF，其中数值仅为软件测试数据，不是真实实验结果。

不启用生成模型时，页面展示检索原文。启用本地 Ollama 时，需要先安装 Ollama 并下载适合硬件的模型；在侧栏填写模型名。云端模式需自行配置供应商和环境变量中的密钥。

中文查询英文论文，先准备本地语义模型：

```bash
python -m pip install -r requirements-semantic.txt
python scripts/prepare_embedding.py
```

脚本明确下载约 470 MB 模型权重并固定版本；准备后语义推理在本机执行。在侧栏选择“多语言语义”或“关键词与语义融合”。本工作目录已经准备好模型；源码压缩包不包含权重和虚拟环境。

详细操作见 [配置与使用指南](docs/SETUP.zh-CN.md)。无需将 API Key 发到聊天中。

## 命令行与验证

```bash
python -m paper_evidence ingest examples/demo-paper.pdf --out data/demo.json
python -m paper_evidence ask "What accuracy did the proposed method achieve?" --index data/demo.json
python -m paper_evidence ask "提出的方法准确率是多少？" --index data/demo.json --retriever hybrid --k 3
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python eval/retrieval_eval.py --out eval/results.json
python eval/semantic_eval.py
```

评测定义是：标注的“页码 + 原文引文”是否完整出现在 top-k 检索片段中。中英配对示例包含 7 个事实、14 个问题。Evidence Recall@3：BM25 英文为 1.0、中文为 0.0；语义与融合模式在两种语言上均为 1.0。另有 3 个无答案探针，语义检索仍会返回候选。它们是虚构数据上的流程测试，不能证明真实科研质量或拒答能力。

查看 [完整结果](eval/semantic-smoke-results.json) 和 [验证记录](docs/VERIFICATION.zh-CN.md)。

## 表格单元格引用

检索后在右侧选择表格片段，勾选“定位表格单元格”，选择数据行和数值列。下载真实论文后，可在“示例论文”中选择经过哈希校验的论文。页面会直接从原文读取 `Proposed · Accuracy：91.2%`，标出行名、表头和数值的三个区域，并导出引用坐标。

生成模型引用结构化表格数值时只能选择坐标，不能自行填写行名和数值。未识别为表格的正文仍只有引文和数值出现校验；选择到正确单元格也不等于回答了正确的问题。

## 无需提问的逐页浏览

展开“逐页浏览论文（无需提问）”，勾选打开后按 PDF 页码翻页。可以查看完整原文，或筛选本页全部、正文、已识别表格片段，再检查单元格、导出原文与来源块坐标。没有检索命中、甚至没有可索引文字时仍能浏览；扫描页尚不能检索。表格筛选数量不是论文真实表格数量。

## 真实论文开发集

```bash
python scripts/fetch_real_papers.py
python eval/real_eval.py
```

当前选择三篇公开 PMLR 论文、12 个事实的 24 个中英配对问题，另有 3 个无答案探针。当前 Evidence Recall@3：BM25 为 18/24，语义为 9/24，融合为 14/24。六个目标表格事实均已恢复来源单元格；12 道配对表格题的单元格 Recall@3 分别为 10/12、7/12、9/12。它们用于开发诊断，不是质量提升或幻觉率下降的证据。

查看 [真实评测报告与失败案例](docs/REAL-EVAL.zh-CN.md)。常见双栏已按列恢复阅读顺序，横线表格保留行列与坐标，长表格按行分块并重复表头和表注。详见 [解析器说明](docs/PARSER.zh-CN.md)，还需独立论文验证。

完整的功能边界见 [英文 README](README.md)，开发和开源路线见 [实施计划](docs/PLAN.zh-CN.md)。

## 固定代码的 NLP 扩充检查

```bash
python scripts/fetch_real_papers.py --dataset eval/nlp
python eval/real_eval.py --dataset eval/nlp --retrievers bm25 --out eval/nlp/bm25-results.json
# 语义依赖和本地 E5 权重准备后：
python eval/real_eval.py --dataset eval/nlp
```

在读取新 PDF 前固定核心代码，新增 BERT、Transformer 两篇论文，八个事实配成 16 道中英问题。证据 Recall@3：BM25 9/16、语义 6/16、融合 9/16。四个表格事实均未得到可靠行列绑定；宽表误拆与多层表头仍是实际问题。本轮没有用这些题调参，没有调用生成模型；标注未独立人工复核，样本也很小。

查看 [完整协议、失败原因与复现步骤](docs/NLP-EVAL.zh-CN.md)。`protocol.json` 会核对核心代码哈希，修改解析器后必须保留原结果并建立新的开发协议。
