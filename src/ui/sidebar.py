import streamlit as st
import os
import time
import tempfile
import shutil
from pathlib import Path
from src.core.rag_system import Config

def render_sidebar(rag_system, env_defaults):
    """Renders the sidebar and handles configuration updates."""
    # Term Dictionary Extraction
    with st.sidebar.expander("📚 用語辞書生成", expanded=False):
        # Check if jargon features are available
        if rag_system and (not hasattr(rag_system, 'jargon_manager') or rag_system.jargon_manager is None):
            st.warning("⚠️ 用語辞書機能は現在利用できません。")
        else:
            if rag_system and hasattr(rag_system, 'config') and rag_system.config.vector_store_type == "chromadb":
                st.markdown("専門用語・類義語辞書を ChromaDB に保存します。")
            else:
                st.markdown("専門用語・類義語辞書を PostgreSQL + pgvector に保存します。")

        input_mode = st.radio(
            "入力タイプ",
            ("フォルダ指定", "ファイルアップロード"),
            horizontal=True,
            key="term_input_mode"
        )

        uploaded_files = None
        input_dir = ""
        if input_mode == "フォルダ指定":
            input_dir = st.text_input("入力フォルダ", value="./docs", key="term_input_dir")
        else:
            uploaded_files = st.file_uploader(
                "入力ファイル (複数可)",
                accept_multiple_files=True,
                key="term_input_files"
            )

        output_json = st.text_input("出力 JSON パス", value="./output/terms.json", key="term_output_json")

        if st.button("🚀 抽出実行", key="run_term_dict"):
            if rag_system is None:
                st.error("RAGシステムが初期化されていません。")
            elif not hasattr(rag_system, 'jargon_manager') or rag_system.jargon_manager is None:
                st.error("用語辞書機能は現在利用できません。")
            else:
                temp_dir_path = None
                try:
                    if input_mode == "フォルダ指定":
                        input_path = (input_dir or "").strip()
                        if not input_path:
                            st.error("入力フォルダを指定してください。")
                            raise ValueError("input_dir_not_set")
                    else:
                        if not uploaded_files:
                            st.error("抽出するファイルをアップロードしてください。")
                            raise ValueError("no_files_uploaded")
                        temp_dir_path = Path(tempfile.mkdtemp(prefix="term_extract_"))
                        for uploaded in uploaded_files:
                            target = temp_dir_path / uploaded.name
                            with open(target, "wb") as f:
                                f.write(uploaded.getbuffer())
                        input_path = str(temp_dir_path)

                    output_path = Path(output_json)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with st.spinner("用語抽出中..."):
                        rag_system.extract_terms(input_path, str(output_path))
                    st.success(f"辞書を生成しました ✔️ → {output_path}")
                except ValueError:
                    # エラーメッセージは既に表示済み
                    pass
                except Exception as e:
                    st.error(f"用語抽出エラー: {e}")
                finally:
                    if temp_dir_path and temp_dir_path.exists():
                        shutil.rmtree(temp_dir_path, ignore_errors=True)

    with st.sidebar:
        st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        if rag_system:
            st.success(f"✅ System Online (Azure) - Collection: **{rag_system.config.collection_name}**")
        else:
            st.warning("⚠️ System Offline")
        
        st.info("すべての設定は「詳細設定」タブで行えます。")

        # Add search type selection
        st.session_state.search_type = st.radio(
            "検索タイプを選択",
            ('ハイブリッド検索', 'ベクトル検索'),
            index=0 if st.session_state.get('search_type', 'ハイブリッド検索') == 'ハイブリッド検索' else 1,
            key='search_type_radio'
        )
        
        # PDF処理方式の簡易選択
        st.markdown("---")
        st.markdown("#### 📑 PDF処理方式")
        
        pdf_options = {
            "legacy": "レガシー (既存)",
            "pymupdf": "PyMuPDF (高速)",
            "azure_di": "Azure DI (高精度)"
        }
        
        if rag_system and hasattr(rag_system, 'config'):
            current_pdf = getattr(rag_system.config, 'pdf_processor_type', 'legacy')
        else:
            current_pdf = 'legacy'
        
        selected_pdf = st.selectbox(
            "PDF処理エンジン",
            options=list(pdf_options.keys()),
            format_func=lambda x: pdf_options[x],
            index=list(pdf_options.keys()).index(current_pdf),
            key='sidebar_pdf_processor',
            help="Azure DIを使用する場合は詳細設定タブで認証情報を設定してください"
        )
        
        if selected_pdf != current_pdf:
            if st.button("PDF処理方式を変更", key="apply_pdf_change"):
                try:
                    from src.ui.state import initialize_rag_system
                    config = rag_system.config if rag_system else Config()
                    config.pdf_processor_type = selected_pdf
                    
                    with st.spinner("設定を更新中..."):
                        if "rag_system" in st.session_state:
                            del st.session_state["rag_system"]
                        st.cache_resource.clear()
                        st.session_state.rag_system = initialize_rag_system(config)
                    
                    st.success(f"✅ PDF処理方式を{pdf_options[selected_pdf]}に変更しました")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"変更エラー: {e}")


def render_langsmith_info():
    """Renders LangSmith tracing info in the sidebar."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    if langsmith_api_key:
        st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
    else:
        st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")
