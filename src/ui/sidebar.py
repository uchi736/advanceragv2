import streamlit as st
import os
import re
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
            st.success(f"✅ System Online (Azure)")
        else:
            st.warning("⚠️ System Offline")

        # Collection switcher
        if rag_system:
            st.markdown("---")
            st.markdown("#### 📁 コレクション切り替え")

            # Get available collections
            available_collections = rag_system.get_available_collections()
            current_collection = rag_system.config.collection_name

            # Find current index
            try:
                current_index = available_collections.index(current_collection)
            except ValueError:
                current_index = 0

            # Collection selector
            selected_collection = st.selectbox(
                "コレクション",
                options=available_collections,
                index=current_index,
                key="collection_selector",
                help="文書・辞書・検索の対象となるコレクションを選択します"
            )

            # Switch collection if changed
            if selected_collection != current_collection:
                with st.spinner(f"コレクション '{selected_collection}' に切り替え中..."):
                    success = rag_system.switch_collection(selected_collection)
                    if success:
                        st.success(f"✅ '{selected_collection}' に切り替えました")
                        st.rerun()
                    else:
                        st.error("❌ コレクションの切り替えに失敗しました")

            # New collection creation
            st.markdown("#### 🆕 新規コレクション作成")

            col1, col2 = st.columns([3, 1])
            with col1:
                new_collection_name = st.text_input(
                    "新しいコレクション名",
                    placeholder="例: project_2024",
                    key="new_collection_input",
                    help="英数字とアンダースコアのみ使用可能",
                    label_visibility="collapsed"
                )
            with col2:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)  # Align button
                create_button = st.button("作成", type="secondary", use_container_width=True)

            # Validation & creation logic
            if create_button:
                if not new_collection_name:
                    st.error("❌ コレクション名を入力してください")
                elif not re.match(r'^[a-zA-Z0-9_]+$', new_collection_name):
                    st.error("❌ 英数字とアンダースコア(_)のみ使用できます")
                elif new_collection_name in available_collections:
                    st.warning(f"⚠️ '{new_collection_name}' は既に存在します")
                else:
                    # Create and switch to new collection
                    with st.spinner(f"コレクション '{new_collection_name}' を作成中..."):
                        success = rag_system.switch_collection(new_collection_name)
                        if success:
                            st.success(f"✅ '{new_collection_name}' を作成して切り替えました")
                            st.rerun()
                        else:
                            st.error("❌ コレクションの作成に失敗しました")

        st.markdown("---")
        st.info("すべての設定は「詳細設定」タブで行えます。")

        # Add search type selection
        st.session_state.search_type = st.radio(
            "検索タイプを選択",
            ('ハイブリッド検索', 'ベクトル検索'),
            index=0 if st.session_state.get('search_type', 'ハイブリッド検索') == 'ハイブリッド検索' else 1,
            key='search_type_radio'
        )
        
        # PDF処理方式の表示
        st.markdown("---")
        st.markdown("#### 📑 PDF処理方式")
        st.info("Azure Document Intelligence (固定)")


def render_langsmith_info():
    """Renders LangSmith tracing info in the sidebar."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    if langsmith_api_key:
        st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
    else:
        st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")
