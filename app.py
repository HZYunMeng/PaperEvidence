"""Run with: streamlit run app.py"""
from pathlib import Path
import json
import re

import streamlit as st
import pandas as pd

from paper_evidence.answering import OllamaGenerator, answer_question
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import highlight_page, parse_pdf
from paper_evidence.retrieval import BM25
from paper_evidence.config import load_settings
from paper_evidence.cloud import CompatibleGenerator
from paper_evidence.semantic import E5Embedder, default_model_path, make_retriever
from paper_evidence.tables import cell_reference, table_value_claim
from paper_evidence import __version__

try:
    settings = load_settings()
except Exception as exc:
    st.error(f"本地配置错误：{exc}")
    st.stop()


st.set_page_config(page_title="PaperEvidence · 论文证据助手", page_icon="📄", layout="wide")
st.title("PaperEvidence · 论文证据助手")
st.caption("上传论文，检索原文，逐条查看引用与页码。当前为早期工程原型。")


def show_text(text):
    if text.startswith("| ") and "| ---" in text:
        table_lines = [line for line in text.splitlines() if line.startswith("| ")]
        caption = "\n".join(line for line in text.splitlines() if not line.startswith("| "))
        rows = [[cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
                for line in table_lines]
        if len(rows) >= 3 and all(len(row) == len(rows[0]) for row in rows):
            labels = [f"{label} (第 {i+1} 列)" if rows[0].count(label)>1 else label for i,label in enumerate(rows[0])]
            st.table(pd.DataFrame(rows[2:], columns=labels))
            if caption:
                st.caption(caption)
            return
    st.text(text)
with st.sidebar:
    st.header("论文与模式")
    use_sample = st.checkbox("使用内置示例 PDF", value=True)
    examples = {"虚构演示论文":Path(__file__).parent / "examples/demo-paper.pdf"}
    real_examples = {"SimCLR · 真实论文":"simclr", "CLIP · 真实论文":"clip", "EfficientNet · 真实论文":"efficientnet"}
    for label, stem in real_examples.items():
        path = Path(__file__).parent / "data/real-papers" / f"{stem}.pdf"
        if path.exists():
            examples[label] = path
    example_name = st.selectbox("示例论文",list(examples),disabled=not use_sample)
    upload = st.file_uploader("上传文本型 PDF", type=["pdf"], disabled=use_sample, max_upload_size=30)
    retrieval_mode = st.selectbox("检索方式", ["BM25 关键词", "多语言语义", "关键词与语义融合"])
    mode = st.radio("回答模式", ["原文证据浏览（无需模型）", "本地 Ollama 生成", "云端 API 生成"])
    model = st.text_input("已下载的 Ollama 模型名", disabled=not mode.startswith("本地"))
    top_k = st.slider("检索片段数量", 1, 10, 5)
    if mode.startswith("云端"):
        from urllib.parse import urlparse
        st.caption(f"调用服务：{urlparse(settings.base_url).hostname or '尚未配置'} · 模型：{settings.model or '尚未配置'}")
        st.info("点击查询时，会将问题和检索到的原文片段发送到配置的模型服务，可能产生 API 费用。")
    if retrieval_mode.startswith("BM25"):
        st.info("关键词检索适合作为基线；中文问题查询英文论文可选择多语言语义模式。")
    else:
        st.caption("语义模型在本机运行。候选排序不会自动判断问题是否有答案。")
    st.caption("关键词、余弦相似度和融合排名分数都不是答案正确概率。")


@st.cache_data(show_spinner=False, max_entries=4)
def ingest(data, name, parser_revision="0.4.0"):
    paper = parse_pdf(data, name)
    return paper, chunk_blocks(paper.blocks)


@st.cache_resource(show_spinner=False)
def semantic_embedder(path):
    return E5Embedder(path)


def retriever_for(chunks):
    if retrieval_mode.startswith("BM25"):
        return BM25(chunks)
    path = settings.embedding_path or str(default_model_path())
    embedder = semantic_embedder(path)
    return make_retriever(chunks,"dense" if retrieval_mode == "多语言语义" else "hybrid",
                          embedder=embedder,cache_dir=Path(__file__).parent / ".cache" / "embeddings")


source = None
if use_sample:
    sample = examples[example_name]
    if sample.exists():
        source, name = sample.read_bytes(), sample.name
        if example_name == "虚构演示论文":
            st.warning("示例 PDF 是本项目编写的虚构论文，只用于演示解析和检索；其中数值不是科研实验结果。")
        else:
            manifest = json.loads((Path(__file__).parent / "eval/real/papers.json").read_text())
            item = next(p for p in manifest if p["id"]==sample.stem)
            st.caption(f"真实论文：{item['title']} · 原作者与出处见 [PMLR]({item['page_url']})")
    else:
        st.error("内置示例缺失，请上传 PDF。")
elif upload:
    source, name = upload.getvalue(), upload.name

if source is None:
    st.info("从侧栏选择示例或上传论文。")
    st.stop()
try:
    with st.spinner("解析论文并保留页码与坐标…"):
        paper, chunks = ingest(source, name, __version__)
except Exception as exc:
    st.error(f"PDF 解析失败：{exc}")
    st.stop()
if not chunks:
    st.error("没有可检索文本。扫描件需要 OCR，当前版本未实现 OCR。")
    st.stop()
st.caption(f"{paper.name} · {paper.pages} 页 · {len(chunks)} 个片段")
with st.expander("查看当前解析能力与限制"):
    for warning in paper.warnings:
        st.write(warning)
if st.session_state.get("document_id") != paper.id:
    st.session_state.pop("result", None)
    st.session_state["document_id"] = paper.id
request_options = (retrieval_mode, mode, model, top_k, settings.base_url, settings.model,
                   settings.embedding_path, settings.max_tokens, settings.max_evidence_chars,
                   settings.json_mode, settings.enable_thinking, __version__)
if st.session_state.get("request_options") != request_options:
    st.session_state.pop("result", None)
    st.session_state["request_options"] = request_options
question = st.text_input("向论文提问", value="What accuracy did the proposed method achieve?")
if st.button("检索并查看证据", type="primary"):
    st.session_state.pop("result", None)
    if mode.startswith("本地") and not model.strip():
        st.error("请填写已经下载的本地 Ollama 模型名。")
    else:
        try:
            generator = (OllamaGenerator(model) if mode.startswith("本地") else
                         CompatibleGenerator(settings) if mode.startswith("云端") else None)
            with st.spinner("检索与回答…"):
                st.session_state["result"] = answer_question(question, retriever_for(chunks), generator, top_k)
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("result")
if not result:
    st.stop()
left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("答案与引用")
    st.caption(f'本次问题：{result.get("question", question)}')
    if result["abstain"]:
        st.warning(result["reason"])
    else:
        if result["mode"] == "extractive":
            st.info("这里展示检索原文，尚未由大模型归纳回答。")
        else:
            st.caption("已核对引文存在与数值出现，语义是否支持结论仍需核查。")
        for index, claim in enumerate(result["claims"], 1):
            st.markdown(f"**{index}.**")
            show_text(claim["text"])
            for evidence in claim["evidence"]:
                st.caption(f'第 {evidence["page"]} 页 · {evidence["section"]} · {evidence["kind"]} · 引文存在已核对')
                if evidence.get("cell_bound"):
                    st.caption(f'单元格来源：{evidence["row_label"]} · {evidence["column_label"]} · {evidence["value"]}')
                with st.expander("查看引用原文"):
                    st.text(evidence["quote"])
    with st.expander("检索结果与原始分数"):
        st.json(result["hits"])
    st.download_button("导出本次问答 JSON", data=json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="paper-evidence-answer.json", mime="application/json")
with right:
    st.subheader("回到 PDF 原文")
    lookup = {c.id: c for c in chunks}
    ids = [hit["chunk_id"] for hit in result["hits"]]
    if ids:
        bound = next((e for claim in result["claims"] for e in claim["evidence"] if e.get("cell_bound")),None)
        selected = st.selectbox("选择证据片段", ids,
            index=ids.index(bound["chunk_id"]) if bound else 0,
            format_func=lambda cid: f"第 {lookup[cid].page} 页 · {lookup[cid].section} · {lookup[cid].kind}")
        chunk = lookup[selected]
        boxes = None
        binding = bound if bound and bound["chunk_id"] == selected else None
        if chunk.table and st.checkbox("定位表格单元格",value=bool(binding),key=f"inspect-{selected}"):
            st.caption("第 1 行作为表头，第 1 列作为行名。复杂或合并表头需人工核对。")
            choices = []
            for r in range(1,len(chunk.table.rows)):
                for c in range(1,len(chunk.table.rows[r])):
                    try:
                        cell_reference(chunk,r,c)
                        choices.append(r)
                        break
                    except ValueError:
                        pass
            if choices:
                row = st.selectbox("数据行",choices,
                    index=choices.index(binding["row"]) if binding and binding["row"] in choices else 0,
                    format_func=lambda r:f"第 {r+1} 行 · {' / '.join(c.text for c in chunk.table.rows[r][:2])}",key=f"row-{selected}")
                column = st.selectbox("数值列",range(1,len(chunk.table.rows[0])),
                    index=binding["column"]-1 if binding else 0,
                    format_func=lambda c:f"第 {c+1} 列 · {chunk.table.rows[0][c].text}",key=f"column-{selected}")
                try:
                    reference = cell_reference(chunk,row,column)
                    st.success(table_value_claim(chunk,row,column)["text"])
                    boxes = reference["boxes"]
                    st.download_button("导出单元格引用",json.dumps(reference,ensure_ascii=False,indent=2),
                        file_name="table-cell-citation.json",mime="application/json")
                except ValueError as exc:
                    st.warning(str(exc))
            else:
                st.warning("当前表格没有可可靠绑定的单元格，请核对 PDF 原文。")
        st.caption("黄色框标记选中单元格、行名、表头及必要的同名行上下文；未选择单元格时标记整个片段。")
        try:
            st.image(highlight_page(source, chunk,selected_boxes=boxes), caption=f"第 {chunk.page} 页", width="stretch")
        except Exception as exc:
            st.warning(f"PDF 页面预览不可用：{exc}")
        with st.expander("完整检索片段"):
            st.text(chunk.text)
