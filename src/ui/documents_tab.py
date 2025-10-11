import streamlit as st
import pandas as pd
import time
from src.utils.helpers import _persist_uploaded_file, get_documents_dataframe

def render_documents_tab(rag_system):
    """Renders the document management tab."""
    if not rag_system:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")
        return

    st.markdown("### 📤 ドキュメントアップロード")
    
    # PDF処理方式の表示
    st.info(f"📑 PDF処理方式: **Azure Document Intelligence**")
    
    uploaded_docs = st.file_uploader(
        "ファイルを選択またはドラッグ&ドロップ (.pdf)",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
        key=f"doc_uploader_v7_tab_documents_{rag_system.config.collection_name}"
    )

    if uploaded_docs:
        st.markdown(f"#### 選択されたファイル ({len(uploaded_docs)})")
        file_info = [{"ファイル名": f.name, "サイズ": f"{f.size / 1024:.1f} KB", "タイプ": f.type or "不明"} for f in uploaded_docs]
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

        if st.button("🚀 ドキュメントを処理 (インジェスト)", type="primary", use_container_width=True, key="process_docs_button_v7_tab_documents"):
            progress_bar = st.progress(0, text="処理開始...")
            status_text = st.empty()
            try:
                paths_to_ingest = []
                for i, file in enumerate(uploaded_docs):
                    status_text.info(f"一時保存中: {file.name}")
                    paths_to_ingest.append(str(_persist_uploaded_file(file)))
                    progress_bar.progress((i + 1) / (len(uploaded_docs) * 2), text=f"一時保存完了: {file.name}")

                status_text.info(f"インデックスを構築中... ({len(paths_to_ingest)}件のファイル)")
                rag_system.ingest_documents(paths_to_ingest)
                progress_bar.progress(1.0, text="インジェスト完了！")
                st.success(f"✅ {len(uploaded_docs)}個のファイルが正常に処理されました！")
                time.sleep(1)
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"ドキュメント処理中にエラーが発生しました: {type(e).__name__} - {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    st.markdown("### 📚 登録済みドキュメント")
    docs_df = get_documents_dataframe(rag_system)
    
    if 'doc_to_show_chunks' not in st.session_state:
        st.session_state.doc_to_show_chunks = None

    if not docs_df.empty:
        # Display header
        header_cols = st.columns([3, 1, 2, 2])
        header_cols[0].markdown("**Document ID**")
        header_cols[1].markdown("**Chunks**")
        header_cols[2].markdown("**Last Updated**")
        header_cols[3].markdown("**Actions**")

        for index, row in docs_df.iterrows():
            doc_id = row["Document ID"]
            cols = st.columns([3, 1, 2, 2])
            cols[0].markdown(f'<span style="color: #FAFAFA;">{doc_id}</span>', unsafe_allow_html=True)
            cols[1].markdown(f'<span style="color: #FAFAFA;">{row["Chunks"]}</span>', unsafe_allow_html=True)
            cols[2].markdown(f'<span style="color: #FAFAFA;">{row["Last Updated"]}</span>', unsafe_allow_html=True)
            
            if cols[3].button("チャンク表示/非表示", key=f"toggle_chunks_{doc_id}"):
                if st.session_state.doc_to_show_chunks == doc_id:
                    st.session_state.doc_to_show_chunks = None # Hide if already shown
                else:
                    st.session_state.doc_to_show_chunks = doc_id # Show this one
            
            if st.session_state.doc_to_show_chunks == doc_id:
                with st.spinner(f"'{doc_id}'のチャンクを取得中..."):
                    chunks_df = rag_system.get_chunks_by_document_id(doc_id)
                
                if not chunks_df.empty:
                    st.info(f"{len(chunks_df)}個のチャンクが見つかりました。")
                    
                    csv = chunks_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 全チャンクをCSVでダウンロード",
                        data=csv,
                        file_name=f"chunks_{doc_id}.csv",
                        mime="text/csv",
                        key=f"download_chunks_{doc_id}"
                    )
                    
                    for _, chunk_row in chunks_df.iterrows():
                        st.markdown("---")
                        st.markdown(f"**Chunk ID:** `{chunk_row['chunk_id']}`")
                        
                        # Use a markdown container with custom CSS for better readability and scrolling
                        st.markdown(
                            f"""
                            <div style="background-color: #262730; border-radius: 0.5rem; padding: 10px; height: 200px; overflow-y: auto; border: 1px solid #333;">
                                <pre style="white-space: pre-wrap; word-wrap: break-word; color: #FAFAFA;">{chunk_row['content']}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("このドキュメントにはチャンクデータが見つかりませんでした。")
        
        st.markdown("---")
        st.markdown("### 🗑️ ドキュメント削除")
        doc_ids_for_deletion = ["選択してください..."] + docs_df["Document ID"].tolist()
        doc_to_delete = st.selectbox(
            "削除するドキュメントIDを選択:",
            doc_ids_for_deletion,
            label_visibility="collapsed",
            key=f"doc_delete_selectbox_v7_tab_documents_{rag_system.config.collection_name}"
        )
        if doc_to_delete != "選択してください...":
            st.warning(f"**警告:** ドキュメント '{doc_to_delete}' を削除すると、関連する全てのチャンクがデータベースとベクトルストアから削除されます。この操作は元に戻せません。")
            if st.button(f"'{doc_to_delete}' を削除実行", type="secondary", key="doc_delete_button_v7_tab_documents"):
                try:
                    with st.spinner(f"削除中: {doc_to_delete}"):
                        success, message = rag_system.delete_document_by_id(doc_to_delete)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"ドキュメント削除中にエラーが発生しました: {type(e).__name__} - {e}")
    else:
        st.info("まだドキュメントが登録されていません。上のセクションからアップロードしてください。")
