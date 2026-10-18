"""Session-only collection UI shared by all source inspectors."""
import streamlit as st

from .collection import make_evidence, add_evidence, export_json, export_markdown, pdf_hash

KEY = 'evidence_collection'


def collect_cell(source, chunk, row, column, *, document=None, key):
    try:
        item = make_evidence(source,chunk,row,column,document)
    except ValueError as exc:
        st.caption(f'暂不能加入证据清单：{exc}')
        return
    items = st.session_state.get(KEY,[])
    exists = any(old['id']==item['id'] for old in items)
    if st.button('已在证据清单中' if exists else '加入证据清单',disabled=exists,key=f'collect-{key}'):
        st.session_state[KEY],_ = add_evidence(items,item)
        st.rerun()


def show_collection(source, paper):
    items = st.session_state.get(KEY,[])
    with st.expander(f'证据清单（{len(items)} 条）'):
        st.caption('只在本次会话中保留；切换论文不会清空，刷新或重启后可能丢失，请及时导出。清单不判断实验条件是否可比，不自动排名或计算提升。')
        if not items:
            st.info('在单元格来源检查中点击“加入证据清单”，再回来整理和导出。')
            return
        st.dataframe([{'来源':f'E{i}','文件':item['document']['name'],
                       '方法':item['reference']['row_label'],'指标':item['reference']['column_label'],
                       '原文值':item['reference']['value'],'PDF 页':item['reference']['page']}
                      for i,item in enumerate(items,1)],hide_index=True)
        st.download_button('导出证据清单 JSON',export_json(items),file_name='evidence-collection.json',mime='application/json')
        st.download_button('导出证据清单 Markdown',export_markdown(items),file_name='evidence-collection.md',mime='text/markdown')
        lookup={item['id']:item for item in items}
        selected=st.selectbox('查看已保存证据',[None]+list(lookup),
            format_func=lambda eid:'请选择证据' if eid is None else
                f"E{list(lookup).index(eid)+1} · {lookup[eid]['document']['name']} · {lookup[eid]['reference']['row_label']} · {lookup[eid]['reference']['column_label']}",
            key='collection-selected')
        if selected is None:
            return
        item=lookup[selected]
        st.caption(f"记录时的解析版本：{item['app_version']} · 文件 SHA256：{item['document']['sha256']}")
        with st.expander('查看保存的来源记录与同行字段'):
            st.json(item)
        if st.button('移除这条证据',key=f'collection-remove-{selected}'):
            st.session_state[KEY]=[i for i in items if i['id']!=selected]
            st.rerun()
        if source is None:
            st.info('请重新选择或上传原 PDF，才能显示已保存的来源高亮。清单仍可导出。')
        elif pdf_hash(source)!=item['document']['sha256']:
            st.info('当前 PDF 与这条证据的文件哈希不同。重新选择或上传原 PDF 后，才能核对已保存的来源高亮。')
        elif st.checkbox('显示已保存证据的 PDF 来源',key=f'collection-preview-{selected}'):
            from types import SimpleNamespace
            from .parsing import highlight_page
            r=item['reference']
            try:
                preview=highlight_page(source,SimpleNamespace(page=r['page'],boxes=()),selected_boxes=r['boxes'])
                st.image(preview,caption=f"{paper.name} · PDF 第 {r['page']} 页 · 保存时的来源区域",width='stretch')
            except Exception as exc:
                st.warning(f'PDF 页面预览不可用：{exc}')
