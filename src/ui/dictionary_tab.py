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

def get_all_terms_cached(_jargon_manager, collection_name: str):
    # 以前はキャッシュしていたが、DB更新を即時UIに反映させるためキャッシュを外す
    return pd.DataFrame(_jargon_manager.get_all_terms())

def check_vector_store_has_data(rag_system, collection_name: str):
    """Check if vector store or document chunks have any data for the specified collection."""
    try:
        if not rag_system or not hasattr(rag_system, 'engine'):
            return False

        with rag_system.engine.connect() as conn:
            # Check vector store (langchain_pg_embedding) for this collection
            try:
                # Get collection_id for this collection_name
                result = conn.execute(
                    text("SELECT uuid FROM langchain_pg_collection WHERE name = :cname"),
                    {"cname": collection_name}
                )
                collection_id = result.scalar()

                if collection_id:
                    result = conn.execute(
                        text("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = :cid"),
                        {"cid": collection_id}
                    )
                    vector_count = result.scalar()
                else:
                    vector_count = 0
            except:
                vector_count = 0

            # Check keyword search chunks (document_chunks) for this collection
            try:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM document_chunks WHERE collection_name = :cname"),
                    {"cname": collection_name}
                )
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
    """Renders the dictionary tab with 3 sub-tabs."""
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

    # 3つのタブを作成
    tabs = st.tabs(["📋 用語一覧", "🔧 用語抽出", "📊 抽出分析"])

    with tabs[0]:
        render_term_list(rag_system, jargon_manager)

    with tabs[1]:
        render_term_extraction(rag_system, jargon_manager)

    with tabs[2]:
        render_term_analysis()


def render_term_list(rag_system, jargon_manager):
    """📋 用語一覧タブ"""

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
                        if hasattr(get_all_terms_cached, "clear"):
                            get_all_terms_cached.clear()
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
            if hasattr(get_all_terms_cached, "clear"):
                if hasattr(get_all_terms_cached, "clear"):
                    get_all_terms_cached.clear()
            st.success("キャッシュをクリアしました。ページを再読み込みしてください。")

    # Load term data
    with st.spinner("用語辞書を読み込み中..."):
        all_terms_df = get_all_terms_cached(jargon_manager, jargon_manager.collection_name)

    # Show registered terms section
    if all_terms_df.empty:
        st.info("まだ用語が登録されていません。「用語抽出」タブから実行してください。")
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
            delete_key = f"delete_card_{row['term']}_{idx}" if 'id' not in row else f"delete_card_{row['id']}"
            if st.button("削除", key=delete_key, use_container_width=True):
                deleted, errors = rag_system.delete_jargon_terms([row['term']])
                if deleted:
                    st.success(f"用語「{row['term']}」を削除しました。")
                    if hasattr(get_all_terms_cached, "clear"):
                        get_all_terms_cached.clear()
                else:
                    st.error(f"用語「{row['term']}」の削除に失敗しました。")

    else:  # テーブル形式（仮想スクロール対応）
        display_df = terms_df.copy()
        display_df['aliases'] = display_df['aliases'].apply(lambda x: ', '.join(x) if x else '')
        display_df['related_terms'] = display_df['related_terms'].apply(lambda x: ', '.join(x) if x else '')

        # カラム名を日本語に
        column_mapping = {
            'term': '用語', 'definition': '定義', 'domain': '分野',
            'aliases': '類義語', 'related_terms': '関連語',
            'updated_at': '更新日時'
        }
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
            height=600,
            column_config={
                "削除": st.column_config.CheckboxColumn("削除", default=False),
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
                if hasattr(get_all_terms_cached, "clear"):
                    get_all_terms_cached.clear()

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
            if hasattr(get_all_terms_cached, "clear"):
                get_all_terms_cached.clear()

    csv = terms_df.to_csv(index=False)
    st.download_button(
        label="📥 表示中の用語をCSVでダウンロード",
        data=csv,
        file_name=f"jargon_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="csv_download_button"
    )


def render_term_extraction(rag_system, jargon_manager):
    """🔧 用語抽出タブ"""
    st.markdown("### 📚 用語辞書を生成")

    # Check vector store status for current collection
    has_vector_data = check_vector_store_has_data(rag_system, rag_system.config.collection_name)
    if not has_vector_data:
        st.warning(f"⚠️ コレクション '{rag_system.config.collection_name}' にドキュメントが登録されていません。")
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
        input_dir = "./docs"
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

    # ボタンの無効化条件: 「登録済みドキュメントから抽出」モードかつベクトルストアにデータなし
    button_disabled = (input_mode == "登録済みドキュメントから抽出" and not has_vector_data)

    if st.button("🚀 用語を抽出・生成", type="primary", use_container_width=True, key="run_term_extraction", disabled=button_disabled):
        temp_dir_path = None
        try:
            if input_mode == "登録済みドキュメントから抽出":
                # Extract text from registered documents in database (current collection only)
                with rag_system.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT content
                            FROM document_chunks
                            WHERE collection_name = :cname
                            ORDER BY created_at
                        """),
                        {"cname": rag_system.config.collection_name}
                    )
                    all_chunks = [row[0] for row in result]

                if not all_chunks:
                    st.error(f"コレクション '{rag_system.config.collection_name}' に登録済みドキュメントが見つかりません。")
                else:
                    # Create temporary file with all content
                    temp_dir_path = Path(tempfile.mkdtemp(prefix="term_extract_registered_"))
                    temp_file = temp_dir_path / "registered_documents.txt"

                    # Write all chunks to file
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(all_chunks))

                    input_path = str(temp_dir_path)
                    st.info(f"コレクション '{rag_system.config.collection_name}' から {len(all_chunks)} チャンクを抽出しました。")

                    output_path = Path(output_json)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # WebSocketタイムアウト対策
                    progress_bar = st.progress(0, text="初期化中...")
                    status_text = st.empty()

                    import threading
                    import time

                    extraction_complete = threading.Event()

                    def update_progress_periodically():
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
                            time.sleep(60)

                    try:
                        progress_thread = threading.Thread(target=update_progress_periodically, daemon=True)
                        progress_thread.start()

                        asyncio.run(rag_system.extract_terms(input_path, str(output_path)))

                        extraction_complete.set()
                        progress_bar.progress(1.0, text="✅ 完了！")

                    finally:
                        extraction_complete.set()
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()

                    st.session_state['term_extraction_output'] = str(output_path)

                    # JSONファイルを自動的にデータベースに登録
                    try:
                        import json
                        with open(output_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            terms = data.get('terms', [])

                        if terms:
                            st.info(f"📥 {len(terms)}件の用語をデータベースに登録中...")

                            registered_count = 0
                            skipped_count = 0
                            error_count = 0

                            for term_data in terms:
                                try:
                                    if jargon_manager.add_term(
                                        term=term_data.get('headword', ''),
                                        definition=term_data.get('definition', ''),
                                        domain=None,
                                        aliases=term_data.get('synonyms', []),
                                        related_terms=term_data.get('related_terms', [])
                                    ):
                                        registered_count += 1
                                    else:
                                        skipped_count += 1
                                except Exception as e:
                                    error_count += 1
                                    import logging
                                    logging.error(f"Failed to register term '{term_data.get('headword', '')}': {e}")

                            st.success(f"✅ データベース登録完了: {registered_count}件登録、{skipped_count}件スキップ、{error_count}件エラー")
                        else:
                            st.warning("抽出された用語がありませんでした。")

                    except Exception as e:
                        st.error(f"データベース登録エラー: {e}")

                    st.success(f"✅ 用語辞書を生成しました → {output_path}")
                    if hasattr(get_all_terms_cached, "clear"):
                        get_all_terms_cached.clear()
                    st.rerun()
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

                    # Similar progress handling as above
                    # ... (同様の処理)

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


def render_term_analysis():
    """📊 抽出分析タブ"""
    st.subheader("📊 専門用語抽出の特徴分析")
    st.caption("Ground Truthとの比較により、TF-IDF+C-valueアプローチの有効性を検証します")

    st.info("""
**この分析では以下を確認できます:**
- カテゴリ別Recall（どのタイプの用語が抽出されているか）
- 頻度別Recall（低頻度用語は見逃されていないか）
- TF-IDF/C-valueスコアの分布
- 見逃された用語（False Negatives）
- 誤検出された用語（False Positives）
    """)

    # 1. Ground Truth アップロード
    ground_truth_file = st.file_uploader(
        "Ground Truth JSON",
        type=['json'],
        help="正解データ (例: test_data/ground_truth.json)",
        key="gt_upload"
    )

    # 2. 候補用語（デバッグファイル）の自動検出
    candidates_path = None
    candidates_file_obj = None

    # 2-1. セッション変数から抽出結果の場所を推測
    if 'term_extraction_output' in st.session_state:
        output_path = Path(st.session_state['term_extraction_output'])
        debug_path = output_path.parent / "term_extraction_debug.json"
        if debug_path.exists():
            candidates_path = debug_path
            st.success(f"✅ 候補用語データを自動検出: {candidates_path}")
        else:
            candidates_path = None

    # 2-2. デフォルトパスから検出
    if not candidates_path and Path("./output/term_extraction_debug.json").exists():
        candidates_path = Path("./output/term_extraction_debug.json")
        st.success(f"✅ 候補用語データを自動検出: {candidates_path}")

    # 2-3. どちらもない場合
    if not candidates_path:
        st.warning("⚠️ 候補用語データ（term_extraction_debug.json）が見つかりません")
        st.info("先に「用語抽出」タブで抽出を実行するか、手動でアップロードしてください")

    # 2-4. 手動アップロード（任意 or 必須）
    if candidates_path:
        st.caption("別のファイルを使う場合は下記からアップロード↓")
        label = "別の候補用語JSONを使う（任意）"
    else:
        label = "候補用語 JSON（必須）"

    manual_candidates = st.file_uploader(
        label,
        type=['json'],
        help="候補用語データ (例: output/term_extraction_debug.json)",
        key="candidates_upload"
    )

    # 手動アップロードがあればそちらを優先
    if manual_candidates:
        candidates_file_obj = manual_candidates
        st.info("✅ 手動アップロードされたファイルを使用します")
    elif candidates_path:
        # 自動検出されたファイルを使用
        pass
    else:
        candidates_file_obj = None

    # 3. 分析実行ボタンの有効/無効
    can_analyze = ground_truth_file and (candidates_path or candidates_file_obj)

    if not can_analyze:
        missing = []
        if not ground_truth_file:
            missing.append("Ground Truth JSON")
        if not (candidates_path or candidates_file_obj):
            missing.append("候補用語 JSON")
        st.warning(f"⚠️ 不足: {', '.join(missing)}")

    # 4. 分析実行
    if st.button("🔍 分析を実行", type="primary", use_container_width=True, disabled=not can_analyze):
        with st.spinner("分析中..."):
            try:
                import json
                from src.rag.term_analysis import TermFeatureAnalyzer

                # Ground Truth読み込み
                ground_truth = json.load(ground_truth_file)

                # 候補用語データ読み込み
                if candidates_file_obj:
                    candidates_data = json.load(candidates_file_obj)
                else:
                    with open(candidates_path, 'r', encoding='utf-8') as f:
                        candidates_data = json.load(f)

                # 候補用語リストを取得（複数フォーマット対応）
                candidate_terms = (
                    candidates_data.get('candidates') or
                    candidates_data.get('terms') or
                    (candidates_data if isinstance(candidates_data, list) else [])
                )

                st.info(f"📊 候補用語数: {len(candidate_terms)}件")

                # 分析実行（documentsは空リスト）
                analyzer = TermFeatureAnalyzer(ground_truth, candidate_terms, [])
                results = analyzer.analyze()

                # 結果表示
                st.success("✅ 分析完了")

                # 1. 概要メトリクス
                metrics = results['overall_metrics']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Recall", f"{metrics['recall']:.1%}")
                col2.metric("Precision", f"{metrics['precision']:.1%}")
                col3.metric("F1 Score", f"{metrics['f1_score']:.1%}")
                col4.metric("Ground Truth", metrics['total_ground_truth'])

                st.markdown("---")

                # 2. カテゴリ別Recall
                st.subheader("📈 カテゴリ別Recall")
                category_analysis = results['category_analysis']

                category_df = pd.DataFrame([
                    {
                        'カテゴリ': cat,
                        'Ground Truth数': data['total'],
                        '抽出数': data['extracted'],
                        'Recall': f"{data['recall']:.1%}"
                    }
                    for cat, data in sorted(category_analysis.items(), key=lambda x: x[1]['recall'], reverse=True)
                ])
                st.dataframe(category_df, use_container_width=True, hide_index=True)

                # 3. 頻度別Recall
                st.markdown("---")
                st.subheader("📊 頻度別Recall")
                freq_analysis = results['frequency_analysis']

                freq_df = pd.DataFrame([
                    {
                        '頻度範囲': label,
                        'Ground Truth数': data['total'],
                        '抽出数': data['extracted'],
                        'Recall': f"{data['recall']:.1%}"
                    }
                    for label, data in freq_analysis.items()
                ])
                st.dataframe(freq_df, use_container_width=True, hide_index=True)

                # 4. 見逃された用語
                st.markdown("---")
                with st.expander("❌ 見逃された用語 (False Negatives)", expanded=False):
                    missed_terms = results['missed_terms'][:30]
                    missed_df = pd.DataFrame(missed_terms)
                    st.dataframe(missed_df, use_container_width=True, hide_index=True)

                # 5. 誤検出された用語
                with st.expander("⚠️ 誤検出された用語 (False Positives)", expanded=False):
                    false_positives = results['false_positives'][:30]
                    fp_df = pd.DataFrame(false_positives)
                    st.dataframe(fp_df, use_container_width=True, hide_index=True)

                # 6. SemReRankスコア改善分析
                if 'semrerank_impact' in results and results['semrerank_impact']['all_changes']:
                    st.markdown("---")
                    st.subheader("🔄 SemReRankスコア改善分析")

                    impact = results['semrerank_impact']
                    freq_impact = impact['frequency_impact']

                    # 頻度別のスコア向上率
                    impact_df = pd.DataFrame([
                        {
                            '頻度範囲': label,
                            '対象用語数': data['count'],
                            '平均スコア向上率': f"{(data['mean_ratio'] - 1) * 100:.1f}%",
                            '中央値スコア向上率': f"{(data['median_ratio'] - 1) * 100:.1f}%"
                        }
                        for label, data in freq_impact.items()
                        if data['count'] > 0
                    ])
                    st.dataframe(impact_df, use_container_width=True, hide_index=True)

                    st.caption("💡 低頻度用語ほどSemReRankの恩恵を受けやすい傾向があります")

                    # Ground Truth用語の頻度分布
                    if 'gt_frequencies' in impact and impact['gt_frequencies']:
                        st.markdown("#### 📊 Ground Truth用語の頻度分布")
                        gt_freq_dist = impact['gt_freq_distribution']

                        # 頻度分布テーブル
                        gt_dist_df = pd.DataFrame([
                            {
                                '頻度範囲': label,
                                '用語数': count,
                                '割合': f"{count / sum(gt_freq_dist.values()) * 100:.1f}%" if sum(gt_freq_dist.values()) > 0 else "0%"
                            }
                            for label, count in gt_freq_dist.items()
                            if count > 0
                        ])
                        st.dataframe(gt_dist_df, use_container_width=True, hide_index=True)

                        st.caption(f"💡 合計 {len(impact['gt_frequencies'])} 件の正解用語が見つかりました")

                    # スコア分布の可視化
                    with st.expander("📊 スコア分布の詳細", expanded=False):
                        all_changes = impact['all_changes']

                        # Before/After散布図
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use('Agg')  # バックエンド設定

                        # 日本語フォント設定
                        import platform
                        if platform.system() == 'Windows':
                            plt.rcParams['font.family'] = 'Yu Gothic'
                        elif platform.system() == 'Darwin':  # macOS
                            plt.rcParams['font.family'] = 'Hiragino Sans'
                        else:  # Linux
                            plt.rcParams['font.family'] = 'Noto Sans CJK JP'
                        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

                        # 3つのグラフを配置
                        fig = plt.figure(figsize=(15, 10))
                        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                        ax1 = fig.add_subplot(gs[0, 0])
                        ax2 = fig.add_subplot(gs[0, 1])
                        ax3 = fig.add_subplot(gs[1, :])

                        # 左上: Base Score vs Revised Score
                        base_scores = [x['base_score'] for x in all_changes]
                        revised_scores = [x['revised_score'] for x in all_changes]

                        ax1.scatter(base_scores, revised_scores, alpha=0.6)
                        max_score = max(max(base_scores), max(revised_scores))
                        ax1.plot([0, max_score], [0, max_score], 'r--', label='y=x', linewidth=1)
                        ax1.set_xlabel('正規化スコア (Before)')
                        ax1.set_ylabel('正規化スコア (After)')
                        ax1.set_title('SemReRankによるスコア変化')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)

                        # 右上: 頻度別スコア向上率
                        freq_labels = [label for label, data in freq_impact.items() if data['count'] > 0]
                        mean_ratios = [(freq_impact[label]['mean_ratio'] - 1) * 100
                                       for label in freq_labels]

                        ax2.bar(freq_labels, mean_ratios, color='steelblue', alpha=0.7)
                        ax2.set_xlabel('出現頻度')
                        ax2.set_ylabel('平均スコア向上率 (%)')
                        ax2.set_title('頻度別スコア向上率')
                        ax2.grid(True, axis='y', alpha=0.3)
                        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)

                        # 下段: スコア分布ヒストグラム（Before/After重ね合わせ）
                        ax3.hist(base_scores, bins=30, alpha=0.5, label='適用前', color='orange', edgecolor='black')
                        ax3.hist(revised_scores, bins=30, alpha=0.5, label='適用後', color='blue', edgecolor='black')
                        ax3.set_xlabel('正規化スコア (0-1)')
                        ax3.set_ylabel('用語数')
                        ax3.set_title('スコア分布: SemReRank適用前後')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3, axis='y')

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    # Ground Truth用語の頻度ヒストグラム
                    if 'gt_frequencies' in impact and impact['gt_frequencies']:
                        with st.expander("📈 Ground Truth用語の頻度ヒストグラム", expanded=False):
                            # 日本語フォント設定
                            import platform
                            if platform.system() == 'Windows':
                                plt.rcParams['font.family'] = 'Yu Gothic'
                            elif platform.system() == 'Darwin':  # macOS
                                plt.rcParams['font.family'] = 'Hiragino Sans'
                            else:  # Linux
                                plt.rcParams['font.family'] = 'Noto Sans CJK JP'
                            plt.rcParams['axes.unicode_minus'] = False

                            fig, ax = plt.subplots(figsize=(10, 5))

                            gt_freqs = impact['gt_frequencies']
                            ax.hist(gt_freqs, bins=range(1, max(gt_freqs) + 2), alpha=0.7, color='green', edgecolor='black')
                            ax.set_xlabel('出現頻度')
                            ax.set_ylabel('用語数')
                            ax.set_title('Ground Truth用語の頻度分布')
                            ax.grid(True, alpha=0.3, axis='y')

                            # 統計情報を表示
                            mean_freq = sum(gt_freqs) / len(gt_freqs)
                            median_freq = sorted(gt_freqs)[len(gt_freqs) // 2]
                            ax.axvline(mean_freq, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_freq:.1f}')
                            ax.axvline(median_freq, color='blue', linestyle='--', linewidth=2, label=f'中央値: {median_freq}')
                            ax.legend()

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                # 6.5. スコア分布ヒストグラム（全候補用語）
                st.markdown("---")
                st.subheader("📊 候補用語スコア分布")
                st.caption("TF-IDF、C-value、総合スコアの分布を可視化")

                # 候補用語データからスコアを抽出
                tfidf_scores = [t.get('tfidf_score', 0) for t in candidate_terms if t.get('tfidf_score', 0) > 0]
                cvalue_scores = [t.get('cvalue_score', 0) for t in candidate_terms if t.get('cvalue_score', 0) > 0]
                base_scores_all = [t.get('base_score', 0) for t in candidate_terms if t.get('base_score', 0) > 0]
                revised_scores_all = [t.get('revised_score', 0) for t in candidate_terms if t.get('revised_score', 0) > 0]

                if tfidf_scores or cvalue_scores or base_scores_all:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')

                    # 日本語フォント設定
                    import platform
                    if platform.system() == 'Windows':
                        plt.rcParams['font.family'] = 'Yu Gothic'
                    elif platform.system() == 'Darwin':
                        plt.rcParams['font.family'] = 'Hiragino Sans'
                    else:
                        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
                    plt.rcParams['axes.unicode_minus'] = False

                    # 2x2グリッドでスコア分布を表示
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                    # TF-IDFスコア分布
                    if tfidf_scores:
                        axes[0, 0].hist(tfidf_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                        axes[0, 0].set_xlabel('TF-IDFスコア')
                        axes[0, 0].set_ylabel('用語数')
                        axes[0, 0].set_title(f'TF-IDFスコア分布 (n={len(tfidf_scores)})')
                        axes[0, 0].grid(True, alpha=0.3, axis='y')
                        # 統計情報
                        mean_tfidf = sum(tfidf_scores) / len(tfidf_scores)
                        axes[0, 0].axvline(mean_tfidf, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_tfidf:.2f}')
                        axes[0, 0].legend()

                    # C-valueスコア分布
                    if cvalue_scores:
                        axes[0, 1].hist(cvalue_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
                        axes[0, 1].set_xlabel('C-valueスコア')
                        axes[0, 1].set_ylabel('用語数')
                        axes[0, 1].set_title(f'C-valueスコア分布 (n={len(cvalue_scores)})')
                        axes[0, 1].grid(True, alpha=0.3, axis='y')
                        # 統計情報
                        mean_cvalue = sum(cvalue_scores) / len(cvalue_scores)
                        axes[0, 1].axvline(mean_cvalue, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_cvalue:.2f}')
                        axes[0, 1].legend()

                    # Base Score分布（正規化前）
                    if base_scores_all:
                        axes[1, 0].hist(base_scores_all, bins=30, alpha=0.7, color='orange', edgecolor='black')
                        axes[1, 0].set_xlabel('Base Score')
                        axes[1, 0].set_ylabel('用語数')
                        axes[1, 0].set_title(f'Base Score分布 (n={len(base_scores_all)})')
                        axes[1, 0].grid(True, alpha=0.3, axis='y')
                        # 統計情報
                        mean_base = sum(base_scores_all) / len(base_scores_all)
                        axes[1, 0].axvline(mean_base, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_base:.2f}')
                        axes[1, 0].legend()

                    # Revised Score分布（SemReRank適用後）
                    if revised_scores_all:
                        axes[1, 1].hist(revised_scores_all, bins=30, alpha=0.7, color='purple', edgecolor='black')
                        axes[1, 1].set_xlabel('Revised Score')
                        axes[1, 1].set_ylabel('用語数')
                        axes[1, 1].set_title(f'Revised Score分布 (SemReRank適用後, n={len(revised_scores_all)})')
                        axes[1, 1].grid(True, alpha=0.3, axis='y')
                        # 統計情報
                        mean_revised = sum(revised_scores_all) / len(revised_scores_all)
                        axes[1, 1].axvline(mean_revised, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_revised:.2f}')
                        axes[1, 1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Ground Truth vs 全候補の比較
                    with st.expander("📊 Ground Truth vs 全候補の比較", expanded=False):
                        st.caption("Ground Truth用語と全候補用語のスコア分布を比較")

                        # Ground Truth用語のスコアを抽出
                        gt_terms_set = set(ground_truth.get("all_documents", []))
                        if not gt_terms_set:
                            # all_documentsキーがない場合は、全ドキュメントの用語を統合
                            for key, doc_terms in ground_truth.items():
                                if isinstance(doc_terms, list):
                                    gt_terms_set.update(doc_terms)

                        gt_tfidf = []
                        gt_cvalue = []
                        gt_base = []
                        gt_revised = []

                        for term_data in candidate_terms:
                            term_name = term_data.get('term') or term_data.get('headword')
                            if term_name in gt_terms_set:
                                if term_data.get('tfidf_score', 0) > 0:
                                    gt_tfidf.append(term_data['tfidf_score'])
                                if term_data.get('cvalue_score', 0) > 0:
                                    gt_cvalue.append(term_data['cvalue_score'])
                                if term_data.get('base_score', 0) > 0:
                                    gt_base.append(term_data['base_score'])
                                if term_data.get('revised_score', 0) > 0:
                                    gt_revised.append(term_data['revised_score'])

                        if gt_tfidf or gt_cvalue or gt_base:
                            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

                            # TF-IDF比較
                            if tfidf_scores and gt_tfidf:
                                axes2[0, 0].hist(tfidf_scores, bins=30, alpha=0.5, label='全候補', color='gray', edgecolor='black')
                                axes2[0, 0].hist(gt_tfidf, bins=30, alpha=0.7, label='Ground Truth', color='blue', edgecolor='black')
                                axes2[0, 0].set_xlabel('TF-IDFスコア')
                                axes2[0, 0].set_ylabel('用語数')
                                axes2[0, 0].set_title('TF-IDFスコア: Ground Truth vs 全候補')
                                axes2[0, 0].legend()
                                axes2[0, 0].grid(True, alpha=0.3, axis='y')

                            # C-value比較
                            if cvalue_scores and gt_cvalue:
                                axes2[0, 1].hist(cvalue_scores, bins=30, alpha=0.5, label='全候補', color='gray', edgecolor='black')
                                axes2[0, 1].hist(gt_cvalue, bins=30, alpha=0.7, label='Ground Truth', color='green', edgecolor='black')
                                axes2[0, 1].set_xlabel('C-valueスコア')
                                axes2[0, 1].set_ylabel('用語数')
                                axes2[0, 1].set_title('C-valueスコア: Ground Truth vs 全候補')
                                axes2[0, 1].legend()
                                axes2[0, 1].grid(True, alpha=0.3, axis='y')

                            # Base Score比較
                            if base_scores_all and gt_base:
                                axes2[1, 0].hist(base_scores_all, bins=30, alpha=0.5, label='全候補', color='gray', edgecolor='black')
                                axes2[1, 0].hist(gt_base, bins=30, alpha=0.7, label='Ground Truth', color='orange', edgecolor='black')
                                axes2[1, 0].set_xlabel('Base Score')
                                axes2[1, 0].set_ylabel('用語数')
                                axes2[1, 0].set_title('Base Score: Ground Truth vs 全候補')
                                axes2[1, 0].legend()
                                axes2[1, 0].grid(True, alpha=0.3, axis='y')

                            # Revised Score比較
                            if revised_scores_all and gt_revised:
                                axes2[1, 1].hist(revised_scores_all, bins=30, alpha=0.5, label='全候補', color='gray', edgecolor='black')
                                axes2[1, 1].hist(gt_revised, bins=30, alpha=0.7, label='Ground Truth', color='purple', edgecolor='black')
                                axes2[1, 1].set_xlabel('Revised Score')
                                axes2[1, 1].set_ylabel('用語数')
                                axes2[1, 1].set_title('Revised Score: Ground Truth vs 全候補')
                                axes2[1, 1].legend()
                                axes2[1, 1].grid(True, alpha=0.3, axis='y')

                            plt.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)

                            # 統計情報テーブル
                            st.markdown("#### 📈 統計比較")
                            stats_data = []

                            if gt_tfidf and tfidf_scores:
                                stats_data.append({
                                    "スコア種別": "TF-IDF",
                                    "GT平均": f"{sum(gt_tfidf)/len(gt_tfidf):.3f}",
                                    "全体平均": f"{sum(tfidf_scores)/len(tfidf_scores):.3f}",
                                    "GT中央値": f"{sorted(gt_tfidf)[len(gt_tfidf)//2]:.3f}",
                                    "全体中央値": f"{sorted(tfidf_scores)[len(tfidf_scores)//2]:.3f}"
                                })

                            if gt_cvalue and cvalue_scores:
                                stats_data.append({
                                    "スコア種別": "C-value",
                                    "GT平均": f"{sum(gt_cvalue)/len(gt_cvalue):.3f}",
                                    "全体平均": f"{sum(cvalue_scores)/len(cvalue_scores):.3f}",
                                    "GT中央値": f"{sorted(gt_cvalue)[len(gt_cvalue)//2]:.3f}",
                                    "全体中央値": f"{sorted(cvalue_scores)[len(cvalue_scores)//2]:.3f}"
                                })

                            if gt_base and base_scores_all:
                                stats_data.append({
                                    "スコア種別": "Base Score",
                                    "GT平均": f"{sum(gt_base)/len(gt_base):.3f}",
                                    "全体平均": f"{sum(base_scores_all)/len(base_scores_all):.3f}",
                                    "GT中央値": f"{sorted(gt_base)[len(gt_base)//2]:.3f}",
                                    "全体中央値": f"{sorted(base_scores_all)[len(base_scores_all)//2]:.3f}"
                                })

                            if gt_revised and revised_scores_all:
                                stats_data.append({
                                    "スコア種別": "Revised Score",
                                    "GT平均": f"{sum(gt_revised)/len(gt_revised):.3f}",
                                    "全体平均": f"{sum(revised_scores_all)/len(revised_scores_all):.3f}",
                                    "GT中央値": f"{sorted(gt_revised)[len(gt_revised)//2]:.3f}",
                                    "全体中央値": f"{sorted(revised_scores_all)[len(revised_scores_all)//2]:.3f}"
                                })

                            if stats_data:
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                                st.caption("💡 Ground Truth用語の平均スコアが全体より高い場合、そのスコアは専門用語抽出に有効")

                        else:
                            st.info("Ground Truth用語が候補用語データに見つかりませんでした")

                else:
                    st.info("候補用語データにスコア情報が含まれていません")

                # 7. Ground Truth追跡分析（dropout_report.jsonがある場合）
                dropout_report_path = Path("output").glob("dropout_report_*.json")
                dropout_report_files = sorted(dropout_report_path, key=lambda p: p.stat().st_mtime, reverse=True)

                if dropout_report_files:
                    st.markdown("---")
                    st.subheader("📊 Ground Truth追跡レポート")
                    st.caption("各用語が抽出プロセスのどの段階で脱落したかを分析")

                    # 最新のレポートを読み込み
                    latest_dropout_report = dropout_report_files[0]

                    try:
                        with open(latest_dropout_report, 'r', encoding='utf-8') as f:
                            dropout_data = json.load(f)

                        summary = dropout_data.get("summary", {})
                        dropout_by_stage = dropout_data.get("dropout_by_stage", {})
                        extraction_funnel = dropout_data.get("extraction_funnel", [])
                        missed_terms = dropout_data.get("missed_terms", [])

                        # サマリーメトリクス
                        col1, col2, col3 = st.columns(3)
                        col1.metric("抽出成功", f"{summary.get('extracted', 0)}件")
                        col2.metric("脱落", f"{summary.get('missed', 0)}件")
                        col3.metric("Recall", f"{summary.get('recall', 0):.1%}")

                        # 抽出ファネル（段階別残存数）の可視化
                        if extraction_funnel:
                            st.markdown("#### 📉 抽出ファネル（段階別残存数）")

                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')

                            # 日本語フォント設定
                            import platform
                            if platform.system() == 'Windows':
                                plt.rcParams['font.family'] = 'Yu Gothic'
                            elif platform.system() == 'Darwin':
                                plt.rcParams['font.family'] = 'Hiragino Sans'
                            else:
                                plt.rcParams['font.family'] = 'Noto Sans CJK JP'
                            plt.rcParams['axes.unicode_minus'] = False

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                            # 左: 残存数の推移（折れ線グラフ）
                            stages = [entry['stage'] for entry in extraction_funnel]
                            remaining = [entry['remaining'] for entry in extraction_funnel]

                            ax1.plot(stages, remaining, marker='o', linewidth=2, markersize=8, color='steelblue')
                            ax1.set_xlabel('抽出段階')
                            ax1.set_ylabel('残存用語数')
                            ax1.set_title('抽出ファネル: Ground Truth用語の残存数')
                            ax1.grid(True, alpha=0.3)
                            ax1.tick_params(axis='x', rotation=45)

                            # 右: 段階別脱落数（棒グラフ）
                            dropout_counts = [entry['dropout'] for entry in extraction_funnel]
                            colors = ['red' if d > 0 else 'lightgray' for d in dropout_counts]

                            ax2.bar(stages, dropout_counts, color=colors, alpha=0.7)
                            ax2.set_xlabel('抽出段階')
                            ax2.set_ylabel('脱落用語数')
                            ax2.set_title('段階別脱落数')
                            ax2.grid(True, alpha=0.3, axis='y')
                            ax2.tick_params(axis='x', rotation=45)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        # 段階別脱落詳細
                        if dropout_by_stage:
                            st.markdown("#### 📋 段階別脱落詳細")

                            dropout_df = pd.DataFrame([
                                {
                                    '段階': stage,
                                    '脱落数': count,
                                    '割合': f"{count / summary['missed'] * 100:.1f}%" if summary['missed'] > 0 else "0%"
                                }
                                for stage, count in sorted(dropout_by_stage.items(), key=lambda x: x[1], reverse=True)
                                if count > 0
                            ])
                            st.dataframe(dropout_df, use_container_width=True, hide_index=True)

                        # 脱落した用語の詳細
                        with st.expander("❌ 脱落した用語の詳細", expanded=False):
                            if missed_terms:
                                # 脱落段階でグループ化
                                from collections import defaultdict
                                by_dropout_stage = defaultdict(list)

                                for term_info in missed_terms:
                                    stage = term_info.get("dropout_stage", "unknown")
                                    by_dropout_stage[stage].append(term_info["term"])

                                for stage, terms in sorted(by_dropout_stage.items()):
                                    st.markdown(f"**{stage}で脱落（{len(terms)}件）:**")
                                    st.write(", ".join(terms[:20]))
                                    if len(terms) > 20:
                                        st.caption(f"...他 {len(terms) - 20}件")
                            else:
                                st.info("すべてのGround Truth用語が抽出されました")

                        # レポートダウンロード
                        dropout_json = json.dumps(dropout_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            "📥 Ground Truth追跡レポートをダウンロード (JSON)",
                            data=dropout_json,
                            file_name="ground_truth_dropout_report.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Ground Truth追跡レポートの読み込みエラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                # 8. 通常レポートダウンロード
                st.markdown("---")
                md_report = analyzer.generate_markdown_report(results)
                st.download_button(
                    "📥 詳細レポートをダウンロード (Markdown)",
                    data=md_report,
                    file_name="term_analysis_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"分析エラー: {e}")
                import traceback
                st.code(traceback.format_exc())
