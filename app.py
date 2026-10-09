"""Run with: streamlit run app.py"""
from pathlib import Path
import json

import streamlit as st

from paper_evidence.answering import OllamaGenerator, answer_question
from paper_evidence.chunking import chunk_blocks
from paper_evidence.parsing import parse_pdf
from paper_evidence.retrieval import BM25
from paper_evidence.config import load_settings
from paper_evidence.cloud import CompatibleGenerator
from paper_evidence.semantic import E5Embedder, default_model_path, make_retriever
from paper_evidence.corpus import local_examples, verified_pdf, load_manifest, download_paper
from paper_evidence.demo import PRESETS
from paper_evidence.tables import find_cells, label_key
from paper_evidence.catalog import match_label
from paper_evidence.catalog_view import choose_catalog_cell, show_row_source
from paper_evidence.viewer import inspect_source, show_page, show_text, source_label
from paper_evidence import __version__

try:
    settings = load_settings()
except Exception as exc:
    st.error(f"本地配置错误：{exc}")
    st.stop()


st.set_page_config(page_title="PaperEvidence · 论文证据助手", page_icon="📄", layout="wide")
st.title("PaperEvidence · 论文证据助手")
st.caption("查论文里的实验数值，连同方法、指标和 PDF 原文一起核对。无需配置模型即可开始。")


with st.sidebar:
    st.header("论文与模式")
    use_sample = st.checkbox("使用示例论文", value=True)
    featured = next(p for p in load_manifest(Path(__file__).parent / "eval/real/papers.json") if p["id"]=="clip")
    examples = {"clip · 真实论文":(Path(__file__).parent / "data/real-papers/clip.pdf",featured)}
    downloaded, integrity_errors = local_examples(Path(__file__).parent)
    examples.update(downloaded)
    examples["虚构演示论文"] = (Path(__file__).parent / "examples/demo-paper.pdf",None)
    for error in integrity_errors:
        st.warning(f"示例校验失败，已排除：{error}")
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
def ingest(data, name, parser_revision="0.5.1"):
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


source, item = None, None
if use_sample:
    sample, item = examples[example_name]
    valid_sample = sample.exists()
    if item and valid_sample:
        try:
            verified_pdf(sample,item['sha256'])
        except ValueError:
            valid_sample = False
    if valid_sample:
        source, name = (verified_pdf(sample,item["sha256"]) if item else sample.read_bytes()), sample.name
        if example_name == "虚构演示论文":
            st.warning("示例 PDF 是本项目编写的虚构论文，只用于演示解析和检索；其中数值不是科研实验结果。")
        else:
            st.caption(f"真实论文：{item['title']} · 原作者与出处见 [{item['publication']}]({item['page_url']})")
    else:
        if item:
            st.subheader("用真实论文试一次")
            st.write("点击下载 CLIP 的官方 PDF，再点击一个实验指标，即可查看来源数值和高亮。无需 API Key 或模型权重。")
            st.caption(f"来源：[论文与原作者]({item['page_url']}) · {item.get('license_note','PDF 保留原作者版权')}。只在点击后从官方出处下载并校验哈希，保存在本机。")
            if st.button("下载真实论文并开始",type="primary"):
                try:
                    with st.spinner("下载并校验官方 PDF…"):
                        download_paper(item,sample.parent)
                    st.rerun()
                except Exception:
                    st.error("下载或校验未成功。可以重试、上传自己的 PDF，或在侧栏选择虚构演示论文。")
        else:
            st.error("演示文件缺失，请上传 PDF。")
        st.stop()
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
st.caption(f"{paper.name} · {paper.pages} 页 · {len(chunks)} 个片段")
with st.expander("查看当前解析能力与限制"):
    for warning in paper.warnings:
        st.write(warning)
if st.session_state.get("document_id") != paper.id:
    st.session_state.pop("result", None)
    st.session_state.pop("cell_result",None)
    st.session_state["document_id"] = paper.id
request_options = (retrieval_mode, mode, model, top_k, settings.base_url, settings.model,
                   settings.embedding_path, settings.max_tokens, settings.max_evidence_chars,
                   settings.json_mode, settings.enable_thinking, __version__)
if st.session_state.get("request_options") != request_options:
    st.session_state.pop("result", None)
    st.session_state["request_options"] = request_options
if st.session_state.get("cell_revision") != __version__:
    st.session_state.pop("cell_result",None)
    st.session_state["cell_revision"] = __version__
st.subheader("查实验数值")
st.caption("填写表格中的方法名与指标名，直接查找可绑定的单元格。仅匹配标签，不自动推断别名或计算数值。")
method_key,metric_key = f"method-{paper.id}",f"metric-{paper.id}"
presets = PRESETS.get(item["id"],[]) if item else []
if presets:
    st.write("点一个真实例子：")
    for column,preset in zip(st.columns(len(presets)),presets):
        with column:
            if st.button(preset["label"],key=f"preset-{paper.id}-{preset['method']}-{preset['metric']}"):
                st.session_state[method_key],st.session_state[metric_key] = preset['method'],preset['metric']
                st.session_state['cell_result'] = {'method':preset['method'],'metric':preset['metric'],
                                                  'matches':find_cells(chunks,preset['method'],preset['metric'])}
catalog_ref = choose_catalog_cell(source,chunks,paper.id)
if catalog_ref:
    st.session_state[method_key],st.session_state[metric_key] = catalog_ref['row_label'],catalog_ref['column_label']
    st.session_state['cell_result'] = {'method':catalog_ref['row_label'],'metric':catalog_ref['column_label'],
                                      'matches':[catalog_ref],'source_selected':True}
with st.form(f"cell-lookup-{paper.id}"):
    method = st.text_input("方法或模型名",key=method_key,placeholder="例如：BERT BASE")
    metric = st.text_input("指标名或完整分组路径",key=metric_key,placeholder="例如：MRPC 或 BLEU / EN-DE")
    if st.form_submit_button("查找数值",type="primary"):
        st.session_state['cell_result'] = {'method':method,'metric':metric,'matches':find_cells(chunks,method,metric)}
cell_result = st.session_state.get('cell_result')
if cell_result:
    matches = cell_result['matches']
    origin = '已按目录选定来源' if cell_result.get('source_selected') else '本次查表'
    st.caption(f"{origin}：{cell_result['method']} · {cell_result['metric']}")
    selected_ref = None
    if not matches:
        st.warning("没有找到可可靠绑定的匹配单元格。这不表示论文没有该数据：可从上方表格目录选择原表标签，或展开逐页浏览查看原文。")
    elif len(matches)>1:
        st.info(f"找到 {len(matches)} 个来源。可能存在同名行、不同表格或重复列名，请选择来源；也可填写完整分组路径缩小范围。")
        selected_match = st.selectbox("选择数值来源",[None]+list(range(len(matches))),
            format_func=lambda i:"请选择，尚未确定来源" if i is None else
                f"候选 {i+1} · "+match_label(matches[i],next(c for c in chunks if c.id==matches[i]['chunk_id'])),
            key=f"match-{paper.id}-{label_key(cell_result['method'])}-{label_key(cell_result['metric'])}")
        if selected_match is not None:
            selected_ref = matches[selected_match]
    else:
        selected_ref = matches[0]
    if selected_ref:
        selected_chunk = next(c for c in chunks if c.id==selected_ref['chunk_id'])
        value_panel,pdf_panel = st.columns(2,gap="large")
        with value_panel:
            st.metric(selected_ref['column_label'],selected_ref['value'])
            st.write(f"方法：{selected_ref['row_label']}")
            st.caption(f"PDF 第 {selected_ref['page']} 页 · 数值由解析单元格读取；实验条件和单位请核对原文。")
            st.text(selected_ref['table_caption'])
            if selected_ref['row_context']:
                st.json(selected_ref['row_context'])
            show_row_source(selected_chunk,selected_ref['row'],key_prefix='lookup')
        with pdf_panel:
            inspect_source(source,selected_chunk,binding=selected_ref,key_prefix="lookup",locked=True)
with st.expander("逐页浏览论文（无需提问）"):
    if st.checkbox("打开逐页浏览",key=f"browse-{paper.id}"):
        st.caption("查看已解析片段与完整 PDF。文字可提取不代表表格行列解析正确；未识别的表格、扫描页和图形请在原文核对。")
        page = st.selectbox("PDF 页码",range(1,paper.pages+1),key=f"page-{paper.id}")
        kind = st.radio("内容筛选",["全部片段","表格片段","正文片段"],horizontal=True,key=f"kind-{paper.id}")
        candidates = [c for c in chunks if c.page==page and
                      (kind=="全部片段" or (c.table is not None if kind=="表格片段" else c.table is None))]
        st.caption(f"本页筛选后有 {len(candidates)} 个片段。表格片段只含已识别结构，数量不代表原文中的实际表格数。")
        selected = st.selectbox("浏览片段",[None]+[c.id for c in candidates],
            format_func=lambda cid:"整页 PDF 原文" if cid is None else source_label(next(c for c in candidates if c.id==cid)),
            key=f"browse-source-{paper.id}-{page}-{kind}")
        if selected is None:
            try:
                show_page(source,page)
            except Exception as exc:
                st.warning(f"PDF 页面预览不可用：{exc}")
        else:
            chunk = next(c for c in candidates if c.id==selected)
            show_text(chunk.text)
            inspect_source(source,chunk,key_prefix="browse")
        st.download_button("下载当前 PDF",source,file_name=name,mime="application/pdf",key=f"pdf-download-{paper.id}")
if not chunks:
    st.warning("没有可检索文本。仍可逐页查看 PDF；扫描件需要 OCR，当前版本未实现 OCR。")
    st.stop()
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
            format_func=lambda cid: source_label(lookup[cid]))
        chunk = lookup[selected]
        binding = bound if bound and bound["chunk_id"] == selected else None
        inspect_source(source,chunk,binding=binding)
