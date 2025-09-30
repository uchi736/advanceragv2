import streamlit as st
import tempfile
from pathlib import Path

def render_data_tab(rag_system):
    """Renders the data management tab for SQL analysis."""
    if not rag_system:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")
        return

    # Check if SQL features are available
    if not hasattr(rag_system, 'sql_handler') or rag_system.sql_handler is None:
        st.warning("⚠️ Text-to-SQL機能は現在利用できません。")
        st.info("ChromaDBを使用している場合、Text-to-SQL機能はサポートされていません。PGVectorを使用してください。")
        return

    if not all(hasattr(rag_system, attr) for attr in ['create_table_from_file', 'get_data_tables', 'delete_data_table']):
        st.warning("RAGシステムがデータテーブル管理機能をサポートしていません。")
        return

    st.markdown("### 📊 データファイル管理 (SQL分析用)")
    st.caption("Excel/CSVファイルをアップロードして、SQLで分析可能なテーブルを作成・管理します。")

    uploaded_files = st.file_uploader(
        "Excel/CSVファイルを選択 (.xlsx, .xls, .csv)",
        accept_multiple_files=True,
        type=["xlsx", "xls", "csv"],
        key="sql_data_file_uploader_v7_tab_data"
    )

    if uploaded_files:
        if st.button("🚀 選択したファイルからテーブルを作成/更新", type="primary", key="create_table_button_v7_tab_data"):
            progress_bar = st.progress(0, text="処理開始...")
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.info(f"処理中: {file.name}")
                try:
                    temp_dir = Path(tempfile.gettempdir()) / "rag_sql_data_uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / file.name
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())

                    success, message, schema_info = rag_system.create_table_from_file(str(temp_file_path))
                    if success:
                        st.success(f"✅ {file.name}: {message}")
                        if schema_info:
                            st.text("作成/更新されたテーブルスキーマ:")
                            st.code(schema_info, language='text')
                    else:
                        st.error(f"❌ {file.name}: {message}")
                except Exception as e:
                    st.error(f"❌ {file.name} の処理中にエラー: {type(e).__name__} - {e}")
                finally:
                    progress_bar.progress((i + 1) / len(uploaded_files), text=f"完了: {file.name}")

            progress_bar.empty()
            status_text.empty()
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 登録済みデータテーブル")
    
    try:
        tables_list = rag_system.get_data_tables()
        if tables_list:
            for table_info in tables_list:
                table_name = table_info.get('table_name', '不明なテーブル')
                row_count = table_info.get('row_count', 'N/A')
                schema_text = table_info.get('schema', 'スキーマ情報なし')
                with st.expander(f"📊 {table_name} ({row_count:,}行)"):
                    st.code(schema_text, language='text')
                    st.warning(f"**注意:** テーブル '{table_name}' を削除すると元に戻せません。")
                    if st.button(f"🗑️ テーブル '{table_name}' を削除", key=f"delete_table_{table_name}_v7_tab_data", type="secondary"):
                        with st.spinner(f"テーブル '{table_name}' を削除中..."):
                            del_success, del_msg = rag_system.delete_data_table(table_name)
                        if del_success:
                            st.success(del_msg)
                            st.rerun()
                        else:
                            st.error(del_msg)
        else:
            st.info("分析可能なデータテーブルはまだありません。上記からファイルをアップロードしてください。")
    except Exception as e:
        st.error(f"データテーブルのリスト取得中にエラーが発生しました: {e}")
