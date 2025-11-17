import streamlit as st
import pandas as pd
import tempfile
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from sqlalchemy import text
from src.rag.term_extraction import JargonDictionaryManager
from src.rag.config import Config
from src.utils.helpers import render_term_card

@st.cache_data(ttl=300, show_spinner=False)  # 60秒→300秒に延長してDB負荷を削減
def get_all_terms_cached(_jargon_manager):
    return pd.DataFrame(_jargon_manager.get_all_terms())

def check_vector_store_has_data(rag_system):
    """Check if vector store or document chunks have any data."""
    try:
        if not rag_system or not hasattr(rag_system, 'engine'):
            return False

        with rag_system.engine.connect() as conn:
            # Check vector store (langchain_pg_embedding)
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding"))
                vector_count = result.scalar()
            except:
                vector_count = 0

            # Check keyword search chunks (document_chunks)
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM document_chunks"))
                chunk_count = result.scalar()
            except:
                chunk_count = 0

            # Return True if either table has data
            return vector_count > 0 or chunk_count > 0
    except Exception as e:
        import logging
        logging.error(f"Error checking vector store: {e}")
        return False

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
                        # 次回のページ更新時に自動的に反映されます
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
            st.success("キャッシュをクリアしました。ページを再読み込みしてください。")

    # Load term data
    with st.spinner("用語辞書を読み込み中..."):
        all_terms_df = get_all_terms_cached(jargon_manager)

    # 用語生成UI - Always show at top
    st.markdown("### 📚 用語辞書を生成")

    # Check vector store status
    has_vector_data = check_vector_store_has_data(rag_system)
    if not has_vector_data:
        st.warning("⚠️ ベクトルストアにドキュメントが登録されていません。")
        st.info("""
💡 **事前準備が必要です**:
1. 「**ドキュメント**」タブでPDFをアップロード・登録
2. このタブに戻って用語を生成

定義生成とLLM判定を有効にするには、ドキュメント登録が必須です。
        """)
    else:
        st.success("✅ ベクトルストアにドキュメントが登録されています。用語生成の準備が整いました。")

    st.markdown("""
**📚 用語辞書生成の流れ**:
1. PDFから候補用語を抽出 (Sudachi形態素解析 + SemReRank)
2. ベクトルストアで類似ドキュメント検索 → 定義生成
3. LLMで専門用語を判定・フィルタ
    """)

    # Input mode selection
    input_mode = st.radio(
        "入力ソース",
        ("登録済みドキュメントから抽出", "新規ファイルをアップロード"),
        horizontal=True,
        key="term_input_mode"
    )

    uploaded_files = None
    input_dir = ""
    if input_mode == "登録済みドキュメントから抽出":
        st.info("登録済みの全ドキュメントから用語を抽出します。")
        input_dir = "./docs"  # Placeholder, will use vector store docs
    else:
        uploaded_files = st.file_uploader(
            "用語抽出用のファイルをアップロード (PDF推奨)",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
            key="term_input_files"
        )

    output_json = st.text_input(
        "出力先 (JSON)",
        value="./output/terms.json",
        key="term_output_json"
    )

    if st.button("🚀 用語を抽出・生成", type="primary", use_container_width=True, key="run_term_extraction", disabled=not has_vector_data):
        if not hasattr(rag_system, 'jargon_manager') or rag_system.jargon_manager is None:
            st.error("用語辞書機能は現在利用できません。")
        else:
            temp_dir_path = None
            try:
                if input_mode == "登録済みドキュメントから抽出":
                    # Extract text from registered documents in database
                    with rag_system.engine.connect() as conn:
                        result = conn.execute(text("""
                            SELECT content
                            FROM document_chunks
                            ORDER BY created_at
                        """))
                        all_chunks = [row[0] for row in result]

                    if not all_chunks:
                        st.error("登録済みドキュメントが見つかりません。")
                    else:
                        # Create temporary file with all content
                        temp_dir_path = Path(tempfile.mkdtemp(prefix="term_extract_registered_"))
                        temp_file = temp_dir_path / "registered_documents.txt"

                        # Write all chunks to file
                        with open(temp_file, "w", encoding="utf-8") as f:
                            f.write("\n\n".join(all_chunks))

                        input_path = str(temp_dir_path)
                        st.info(f"登録済みドキュメントから {len(all_chunks)} チャンクを抽出しました。")

                        output_path = Path(output_json)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # WebSocketタイムアウト対策: 疑似進捗バーで定期的にハートビート送信
                        progress_bar = st.progress(0, text="初期化中...")
                        status_text = st.empty()

                        import threading
                        import time

                        # 進捗更新用のフラグ
                        extraction_complete = threading.Event()

                        def update_progress_periodically():
                            """1分ごとに進捗を更新してWebSocketハートビートを維持"""
                            steps = [
                                (10, "📊 チャンク読み込み＆統計処理中..."),
                                (20, "🔍 候補用語抽出中..."),
                                (30, "📈 TF-IDF/C-value計算中..."),
                                (40, "🎯 SemReRank処理中..."),
                                (50, "📝 定義生成中... (これには数分かかります)"),
                                (60, "📝 定義生成中... (60%)"),
                                (70, "📝 定義生成中... (70%)"),
                                (80, "🔬 LLM専門用語判定中... (80%)"),
                                (90, "🔬 LLM専門用語判定中... (90%)"),
                                (95, "📦 結果を保存中..."),
                            ]

                            for percent, message in steps:
                                if extraction_complete.is_set():
                                    break
                                progress_bar.progress(percent / 100, text=message)
                                status_text.info(f"⏳ 処理中: {message}")
                                time.sleep(60)  # 1分待機（WebSocketハートビート維持）

                        try:
                            # バックグラウンドで進捗更新を開始
                            progress_thread = threading.Thread(target=update_progress_periodically, daemon=True)
                            progress_thread.start()

                            # 実際の用語抽出処理を実行
                            asyncio.run(rag_system.extract_terms(input_path, str(output_path)))

                            # 処理完了を通知
                            extraction_complete.set()
                            progress_bar.progress(1.0, text="✅ 完了！")

                        finally:
                            extraction_complete.set()
                            time.sleep(0.5)  # 最後の進捗表示を確認
                            progress_bar.empty()
                            status_text.empty()

                        st.session_state['term_extraction_output'] = str(output_path)
                        st.success(f"✅ 用語辞書を生成しました → {output_path}")
                        st.balloons()
                        get_all_terms_cached.clear()
                        # 次回のページ更新時に自動的に反映されます
                else:
                    if not uploaded_files:
                        st.error("抽出するファイルをアップロードしてください。")
                    else:
                        temp_dir_path = Path(tempfile.mkdtemp(prefix="term_extract_"))
                        for uploaded in uploaded_files:
                            target = temp_dir_path / uploaded.name
                            with open(target, "wb") as f:
                                f.write(uploaded.getbuffer())
                        input_path = str(temp_dir_path)

                        output_path = Path(output_json)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # WebSocketタイムアウト対策: 疑似進捗バーで定期的にハートビート送信
                        progress_bar = st.progress(0, text="初期化中...")
                        status_text = st.empty()

                        import threading
                        import time

                        # 進捗更新用のフラグ
                        extraction_complete = threading.Event()

                        def update_progress_periodically():
                            """1分ごとに進捗を更新してWebSocketハートビートを維持"""
                            steps = [
                                (10, "📊 チャンク読み込み＆統計処理中..."),
                                (20, "🔍 候補用語抽出中..."),
                                (30, "📈 TF-IDF/C-value計算中..."),
                                (40, "🎯 SemReRank処理中..."),
                                (50, "📝 定義生成中... (これには数分かかります)"),
                                (60, "📝 定義生成中... (60%)"),
                                (70, "📝 定義生成中... (70%)"),
                                (80, "🔬 LLM専門用語判定中... (80%)"),
                                (90, "🔬 LLM専門用語判定中... (90%)"),
                                (95, "📦 結果を保存中..."),
                            ]

                            for percent, message in steps:
                                if extraction_complete.is_set():
                                    break
                                progress_bar.progress(percent / 100, text=message)
                                status_text.info(f"⏳ 処理中: {message}")
                                time.sleep(60)  # 1分待機（WebSocketハートビート維持）

                        try:
                            # バックグラウンドで進捗更新を開始
                            progress_thread = threading.Thread(target=update_progress_periodically, daemon=True)
                            progress_thread.start()

                            # 実際の用語抽出処理を実行
                            asyncio.run(rag_system.extract_terms(input_path, str(output_path)))

                            # 処理完了を通知
                            extraction_complete.set()
                            progress_bar.progress(1.0, text="✅ 完了！")

                        finally:
                            extraction_complete.set()
                            time.sleep(0.5)  # 最後の進捗表示を確認
                            progress_bar.empty()
                            status_text.empty()

                        st.session_state['term_extraction_output'] = str(output_path)
                        st.success(f"✅ 用語辞書を生成しました → {output_path}")
                        st.balloons()
                        get_all_terms_cached.clear()
                        # 次回のページ更新時に自動的に反映されます

            except Exception as e:
                st.error(f"用語抽出エラー: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if temp_dir_path and temp_dir_path.exists():
                    shutil.rmtree(temp_dir_path, ignore_errors=True)

    # 用語抽出結果のプレビュー
    output_file = st.session_state.get('term_extraction_output', '')
    if output_file and Path(output_file).exists():
        st.markdown("---")
        with st.expander("📊 抽出結果のプレビュー", expanded=False):
            import json
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    terms = data.get('terms', [])

                st.success(f"✅ {len(terms)}件の用語を抽出しました")

                # 上位10件を表示
                st.markdown("**上位10件の用語:**")
                for i, term in enumerate(terms[:10], 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i}. {term['headword']}**")
                            if term.get('definition'):
                                st.caption(term['definition'][:100] + "..." if len(term['definition']) > 100 else term['definition'])
                        with col2:
                            st.metric("スコア", f"{term.get('score', 0):.3f}")
                            st.caption(f"頻度: {term.get('frequency', 0)}")

            except Exception as e:
                st.error(f"結果ファイルの読み込みエラー: {e}")

    # ナレッジグラフ構築セクション
    terms_file = Path(output_json) if output_json else Path("./output/terms.json")
    clustering_file = Path("output/term_clusters.json")

    if terms_file.exists() and clustering_file.exists():
        st.markdown("---")
        with st.expander("🕸️ ナレッジグラフ構築", expanded=False):
            st.caption("抽出した用語からナレッジグラフを構築します。グラフエクスプローラーで可視化できます。")

            st.success(f"✅ 用語ファイル: {terms_file.name}")
            st.success(f"✅ クラスタファイル: {clustering_file.name}")

            if st.button("🚀 ナレッジグラフを構築", type="primary", use_container_width=True):
                try:
                    with st.spinner("ナレッジグラフ構築中..."):
                        from src.scripts.knowledge_graph.graph_builder import (
                            KnowledgeGraphDB,
                            build_nodes_from_terms,
                            build_category_nodes_from_clusters,
                            build_hierarchy_from_clustering,
                            build_similarity_from_clusters,
                            build_term_category_relationships,
                            load_terms_from_json,
                            load_clustering_results
                        )

                        # ファイル読み込み
                        terms = load_terms_from_json(str(terms_file))
                        clustering_results = load_clustering_results(str(clustering_file))

                        st.info(f"📊 読み込み: {len(terms)}件の用語、{len(clustering_results.get('categories', {}))}個のカテゴリ")

                        # グラフ構築
                        config = Config()
                        pg_url = f"host={config.db_host} port={config.db_port} dbname={config.db_name} user={config.db_user} password={config.db_password}"

                        with KnowledgeGraphDB(pg_url) as db:
                            # 1. ノード作成
                            progress_text = st.empty()
                            progress_text.text("1/5: 用語ノード作成中...")
                            term_to_id = build_nodes_from_terms(db, terms)

                            # 2. カテゴリノード作成
                            progress_text.text("2/5: カテゴリノード作成中...")
                            category_to_id = build_category_nodes_from_clusters(db, clustering_results)

                            # 3. 階層関係構築
                            progress_text.text("3/5: 階層関係構築中...")
                            hierarchy_edges = build_hierarchy_from_clustering(db, clustering_results, term_to_id)

                            # 4. 類似関係構築
                            progress_text.text("4/5: 類似関係構築中...")
                            similarity_edges = build_similarity_from_clusters(db, clustering_results, term_to_id)

                            # 5. カテゴリ関係構築
                            progress_text.text("5/5: カテゴリ関係構築中...")
                            category_edges = build_term_category_relationships(
                                db, clustering_results, term_to_id, category_to_id
                            )

                            progress_text.empty()

                        # 結果表示
                        st.success("✅ ナレッジグラフ構築完了！")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("用語ノード", f"{len(term_to_id):,}")
                            st.metric("カテゴリノード", f"{len(category_to_id):,}")
                        with col2:
                            st.metric("階層エッジ", f"{hierarchy_edges:,}")
                            st.metric("類似エッジ", f"{similarity_edges:,}")
                            st.metric("カテゴリエッジ", f"{category_edges:,}")

                        total_edges = hierarchy_edges + similarity_edges + category_edges
                        st.info(f"📊 総エッジ数: {total_edges:,}")
                        st.info("💡 「グラフ」タブでナレッジグラフを可視化できます")

                except Exception as e:
                    st.error(f"ナレッジグラフ構築エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    st.markdown("---")

    # Show registered terms section
    if all_terms_df.empty:
        st.info("まだ用語が登録されていません。上記の「用語辞書を生成」を実行してください。")
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
    col1, col2 = st.columns(2)
    with col1:
        st.metric("登録用語数", f"{len(terms_df):,}")
    with col2:
        total_synonyms = sum(len(syn_list) if syn_list else 0 for syn_list in terms_df['aliases'])
        st.metric("類義語総数", f"{total_synonyms:,}")

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
                    # 次回のページ更新時に自動的に反映されます
                else:
                    st.error(f"用語「{row['term']}」の削除に失敗しました。")

    else: # テーブル形式（仮想スクロール対応）
        display_df = terms_df.copy()
        display_df['aliases'] = display_df['aliases'].apply(lambda x: ', '.join(x) if x else '')
        display_df['related_terms'] = display_df['related_terms'].apply(lambda x: ', '.join(x) if x else '')

        # カラム名を日本語に
        column_mapping = {
            'term': '用語', 'definition': '定義', 'domain': '分野',
            'aliases': '類義語', 'related_terms': '関連語',
            'updated_at': '更新日時'
        }
        # Add 'id' mapping only if it exists
        if 'id' in display_df.columns:
            column_mapping['id'] = 'ID'
        display_df.rename(columns=column_mapping, inplace=True)

        # 削除ボタン用の列を追加
        display_df['削除'] = False

        # 仮想スクロール対応: 固定高さで大量データでも高速
        edited_df = st.data_editor(
            display_df[['用語', '定義', '分野', '類義語', '関連語', '更新日時', '削除']],
            use_container_width=True,
            hide_index=True,
            height=600,  # 固定高さで仮想スクロール有効化
            column_config={
                "削除": st.column_config.CheckboxColumn(
                    "削除",
                    default=False,
                ),
                "用語": st.column_config.TextColumn("用語", width="medium"),
                "定義": st.column_config.TextColumn("定義", width="large"),
                "分野": st.column_config.TextColumn("分野", width="small"),
                "類義語": st.column_config.TextColumn("類義語", width="medium"),
                "関連語": st.column_config.TextColumn("関連語", width="medium"),
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
                # 次回のページ更新時に自動的に反映されます

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
            # 次回のページ更新時に自動的に反映されます
    csv = terms_df.to_csv(index=False)
    st.download_button(
        label="📥 表示中の用語をCSVでダウンロード",
        data=csv,
        file_name=f"jargon_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="csv_download_button"
    )
