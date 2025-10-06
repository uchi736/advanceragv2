import streamlit as st
import os
from sqlalchemy import text

def check_vector_store_has_data(rag_system):
    """Check if vector store has any documents."""
    try:
        if not rag_system or not hasattr(rag_system, 'engine'):
            return False
        with rag_system.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) FROM langchain_pg_embedding"
            ))
            count = result.scalar()
            return count > 0
    except:
        # If table doesn't exist or any error, assume no data
        return False

def render_sidebar(rag_system, env_defaults):
    """Renders the sidebar and handles configuration updates."""
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
