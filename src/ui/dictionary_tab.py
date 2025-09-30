import streamlit as st
import pandas as pd
from datetime import datetime
from src.rag.term_extraction import JargonDictionaryManager
from src.rag.config import Config
from src.utils.helpers import render_term_card

@st.cache_data(ttl=60, show_spinner=False)
def get_all_terms_cached(_jargon_manager):
    return pd.DataFrame(_jargon_manager.get_all_terms())

def render_dictionary_tab(rag_system):
    """Renders the dictionary tab."""
    st.markdown("### 📖 専門用語辞書")
    st.caption("登録された専門用語・類義語を検索・確認・削除できます。")

    if not rag_system:
        st.warning("⚠️ RAGシステムが初期化されていません。")
        return

    # Check if jargon manager is available
    if not hasattr(rag_system, 'jargon_manager') or rag_system.jargon_manager is None:
        st.warning("⚠️ 専門用語辞書機能は現在利用できません。")
        return

    jargon_manager = rag_system.jargon_manager

    # Manual term registration form
    with st.expander("➕ 新しい用語を手動で登録する"):
        with st.form(key="add_term_form"):
            new_term = st.text_input("用語*", help="登録する専門用語")
            new_definition = st.text_area("定義*", help="用語の定義や説明")
            new_domain = st.text_input("分野", help="関連する技術分野やドメイン")
            new_aliases = st.text_input("類義語 (カンマ区切り)", help="例: RAG, 検索拡張生成")
            new_related_terms = st.text_input("関連語 (カンマ区切り)", help="例: LLM, Vector Search")
            
            submitted = st.form_submit_button("登録")
            if submitted:
                if not new_term or not new_definition:
                    st.error("「用語」と「定義」は必須項目です。")
                else:
                    aliases_list = [alias.strip() for alias in new_aliases.split(',') if alias.strip()]
                    related_list = [rel.strip() for rel in new_related_terms.split(',') if rel.strip()]
                    
                    if jargon_manager.add_term(
                        term=new_term,
                        definition=new_definition,
                        domain=new_domain,
                        aliases=aliases_list,
                        related_terms=related_list
                    ):
                        st.success(f"用語「{new_term}」を登録しました。")
                        get_all_terms_cached.clear()
                        st.rerun()
                    else:
                        st.error(f"用語「{new_term}」の登録に失敗しました。")
    
    st.markdown("---")

    # Search and refresh buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        search_keyword = st.text_input(
            "🔍 用語検索",
            placeholder="検索したい用語を入力してください...",
            key="term_search_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 更新", key="refresh_terms", use_container_width=True):
            get_all_terms_cached.clear()
            st.rerun()

    # Load term data
    with st.spinner("用語辞書を読み込み中..."):
        all_terms_df = get_all_terms_cached(jargon_manager)

    if all_terms_df.empty:
        st.info("まだ用語が登録されていません。サイドバーの「📚 用語辞書生成」から用語を抽出してください。")
        return

    # Filter terms
    if search_keyword:
        terms_df = all_terms_df[
            all_terms_df['term'].str.contains(search_keyword, case=False) |
            all_terms_df['definition'].str.contains(search_keyword, case=False) |
            all_terms_df['aliases'].apply(lambda x: any(search_keyword.lower() in str(s).lower() for s in x) if x else False)
        ]
    else:
        terms_df = all_terms_df

    if terms_df.empty:
        st.info(f"「{search_keyword}」に該当する用語が見つかりませんでした。")
        return

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("登録用語数", f"{len(terms_df):,}")
    with col2:
        total_synonyms = sum(len(syn_list) if syn_list else 0 for syn_list in terms_df['aliases'])
        st.metric("類義語総数", f"{total_synonyms:,}")
    with col3:
        avg_confidence = terms_df['confidence_score'].mean() if 'confidence_score' in terms_df and not terms_df['confidence_score'].isnull().all() else 0.0
        st.metric("平均信頼度", f"{avg_confidence:.2f}")

    st.markdown("---")

    # View mode selection
    view_mode = st.radio(
        "表示形式",
        ["カード形式", "テーブル形式"],
        horizontal=True,
        key="dict_view_mode"
    )

    if view_mode == "カード形式":
        for idx, row in terms_df.iterrows():
            render_term_card(row)
            # Use term as unique key instead of id (which doesn't exist in ChromaDB)
            delete_key = f"delete_card_{row['term']}_{idx}" if 'id' not in row else f"delete_card_{row['id']}"
            if st.button("削除", key=delete_key, use_container_width=True):
                deleted, errors = rag_system.delete_jargon_terms([row['term']])
                if deleted:
                    st.success(f"用語「{row['term']}」を削除しました。")
                    get_all_terms_cached.clear()
                    st.rerun()
                else:
                    st.error(f"用語「{row['term']}」の削除に失敗しました。")

    else: # テーブル形式
        display_df = terms_df.copy()
        display_df['aliases'] = display_df['aliases'].apply(lambda x: ', '.join(x) if x else '')
        display_df['related_terms'] = display_df['related_terms'].apply(lambda x: ', '.join(x) if x else '')
        
        # カラム名を日本語に
        column_mapping = {
            'term': '用語', 'definition': '定義', 'domain': '分野',
            'aliases': '類義語', 'related_terms': '関連語',
            'confidence_score': '信頼度', 'updated_at': '更新日時'
        }
        # Add 'id' mapping only if it exists
        if 'id' in display_df.columns:
            column_mapping['id'] = 'ID'
        display_df.rename(columns=column_mapping, inplace=True)
        
        # 削除ボタン用の列を追加
        display_df['削除'] = False
        
        edited_df = st.data_editor(
            display_df[['用語', '定義', '分野', '類義語', '関連語', '信頼度', '更新日時', '削除']],
            use_container_width=True,
            hide_index=True,
            height=min(600, (len(display_df) + 1) * 35 + 3),
            column_config={
                "削除": st.column_config.CheckboxColumn(
                    "削除",
                    default=False,
                )
            },
            key="dictionary_editor"
        )
        
        terms_to_delete = edited_df[edited_df['削除']]
        if not terms_to_delete.empty:
            if st.button("選択した用語を削除", type="primary"):
                terms_list = terms_to_delete['用語'].tolist()
                deleted_count, error_count = rag_system.delete_jargon_terms(terms_list)
                if deleted_count:
                    st.success(f"{deleted_count}件の用語を削除しました。")
                if error_count:
                    st.warning(f"{error_count}件の削除に失敗しました。")
                get_all_terms_cached.clear()
                st.rerun()

    # CSV download
    st.markdown("---")
    with st.expander("⚠️ 用語辞書を全削除する"):
        st.warning("この操作は取り消せません。全ての専門用語レコードが削除されます。", icon="⚠️")
        if st.button("‼️ 全用語を削除", type="secondary"):
            deleted_count, error_count = rag_system.delete_jargon_terms(terms_df['term'].tolist())
            if deleted_count:
                st.success(f"{deleted_count}件の用語を削除しました。")
            if error_count:
                st.warning(f"{error_count}件の削除に失敗しました。", icon="⚠️")
            get_all_terms_cached.clear()
            st.rerun()
    csv = terms_df.to_csv(index=False)
    st.download_button(
        label="📥 表示中の用語をCSVでダウンロード",
        data=csv,
        file_name=f"jargon_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="csv_download_button"
    )
