import streamlit as st
import os
import time
from src.core.rag_system import Config

def render_settings_tab(rag_system, env_defaults):
    """Renders the detailed settings tab."""
    st.markdown("### ⚙️ システム詳細設定")
    st.caption("RAGシステムの詳細な設定を行います。変更後は「設定を適用」ボタンをクリックしてください。システムの再初期化が必要な場合があります。")

    temp_default_cfg = Config()
    current_values = {}
    if rag_system and hasattr(rag_system, 'config'):
        current_values = rag_system.config.__dict__.copy()
    else:
        current_values = temp_default_cfg.__dict__.copy()
        for key, value in env_defaults.items():
            if key.lower() in current_values:
                current_values[key.lower()] = value
            if key.lower() == "openai_api_key":
                current_values[key.lower()] = None

    for key in temp_default_cfg.__dict__:
        if key not in current_values:
            current_values[key] = getattr(temp_default_cfg, key)
        if key == "openai_api_key":
            current_values[key] = None

    with st.form("detailed_settings_form_v7_tab_settings"):
        col1, col2 = st.columns(2)
        with col1:
            _render_azure_settings(current_values)
            _render_model_identifiers(current_values, temp_default_cfg)
            _render_chunking_settings(current_values, temp_default_cfg)
        with col2:
            _render_search_rag_settings(current_values, temp_default_cfg)
            _render_pdf_processing_settings(current_values, temp_default_cfg)

        st.markdown("---")
        st.markdown("#### 🗄️ データベース設定 (変更には注意が必要です)")
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            _render_db_connection_settings(current_values, temp_default_cfg)
        with db_col2:
            _render_db_auth_settings(current_values, temp_default_cfg)

        s_col, r_col = st.columns([3, 1])
        apply_button = s_col.form_submit_button("🔄 設定を適用", type="primary", use_container_width=True)
        reset_button = r_col.form_submit_button("↩️ デフォルトにリセット", use_container_width=True)

    if apply_button:
        _apply_settings(st.session_state.form_values)
    
    if reset_button:
        _reset_to_defaults(env_defaults)

    st.markdown("---")
    st.markdown("### 📋 現在の有効な設定")
    _display_current_config(rag_system)

def _render_azure_settings(values):
    st.markdown("#### 🔑 Azure OpenAI 設定")
    st.session_state.form_values = {}
    st.session_state.form_values['azure_openai_api_key'] = st.text_input("Azure OpenAI APIキー", value=values.get("azure_openai_api_key", ""), type="password", key="setting_azure_key_v7")
    st.session_state.form_values['azure_openai_endpoint'] = st.text_input("Azure OpenAI エンドポイント", value=values.get("azure_openai_endpoint", ""), key="setting_azure_endpoint_v7")
    st.session_state.form_values['azure_openai_api_version'] = st.text_input("Azure OpenAI APIバージョン", value=values.get("azure_openai_api_version", ""), key="setting_azure_version_v7")
    st.session_state.form_values['azure_openai_chat_deployment_name'] = st.text_input("Azure チャットデプロイメント名", value=values.get("azure_openai_chat_deployment_name", ""), key="setting_azure_chat_deploy_v7")
    st.session_state.form_values['azure_openai_embedding_deployment_name'] = st.text_input("Azure 埋め込みデプロイメント名", value=values.get("azure_openai_embedding_deployment_name", ""), key="setting_azure_embed_deploy_v7")

def _render_model_identifiers(values, defaults):
    st.markdown("#### 🤖 AIモデル識別子 (UI用)")
    emb_opts = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    current_emb = values.get("embedding_model_identifier", defaults.embedding_model_identifier)
    emb_idx = emb_opts.index(current_emb) if current_emb in emb_opts else 0
    st.session_state.form_values['embedding_model_identifier'] = st.selectbox("埋め込みモデル識別子", emb_opts, index=emb_idx, key="setting_emb_model_id_v7")

    llm_opts = ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    current_llm = values.get("llm_model_identifier", defaults.llm_model_identifier)
    llm_idx = llm_opts.index(current_llm) if current_llm in llm_opts else 0
    st.session_state.form_values['llm_model_identifier'] = st.selectbox("言語モデル識別子", llm_opts, index=llm_idx, key="setting_llm_model_id_v7")

def _render_chunking_settings(values, defaults):
    st.markdown("#### 📄 チャンク設定")

    st.session_state.form_values['chunk_size'] = st.number_input("チャンクサイズ", 100, 5000, int(values.get("chunk_size", defaults.chunk_size)), 100, key="setting_chunk_size_v7")
    st.session_state.form_values['chunk_overlap'] = st.number_input("チャンクオーバーラップ", 0, 1000, int(values.get("chunk_overlap", defaults.chunk_overlap)), 50, key="setting_chunk_overlap_v7")

def _render_search_rag_settings(values, defaults):
    st.markdown("#### 🔍 検索・RAG設定")
    st.session_state.form_values['collection_name'] = st.text_input("コレクション名", values.get("collection_name", defaults.collection_name), key="setting_collection_name_v7")

    # デフォルト検索モード選択
    search_mode_options = ["hybrid", "vector", "keyword"]
    search_mode_labels = {"hybrid": "ハイブリッド検索", "vector": "ベクトル検索", "keyword": "キーワード検索"}
    current_mode = values.get("default_search_type", defaults.default_search_type)
    mode_idx = search_mode_options.index(current_mode) if current_mode in search_mode_options else 0
    selected_mode = st.selectbox(
        "デフォルト検索モード",
        search_mode_options,
        index=mode_idx,
        format_func=lambda x: search_mode_labels[x],
        key="setting_default_search_type_v7",
        help="チャットで使用するデフォルトの検索モードを選択します"
    )
    st.session_state.form_values['default_search_type'] = selected_mode

    st.session_state.form_values['final_k'] = st.slider("最終検索結果数 (Final K)", 1, 20, int(values.get("final_k", defaults.final_k)), key="setting_final_k_v7")
    st.session_state.form_values['vector_search_k'] = st.number_input("ベクトル検索数 (Vector K)", 1, 50, int(values.get("vector_search_k", defaults.vector_search_k)), key="setting_vector_k_v7")
    st.session_state.form_values['keyword_search_k'] = st.number_input("キーワード検索数 (Keyword K)", 1, 50, int(values.get("keyword_search_k", defaults.keyword_search_k)), key="setting_keyword_k_v7")
    st.session_state.form_values['rrf_k_for_fusion'] = st.number_input("RAG-Fusion用RRF係数 (k)", 1, 100, int(values.get("rrf_k_for_fusion", defaults.rrf_k_for_fusion)), key="setting_rrf_k_v7")

def _render_db_connection_settings(values, defaults):
    st.session_state.form_values['db_host'] = st.text_input("DBホスト", values.get("db_host", defaults.db_host), key="setting_db_host_v7")
    st.session_state.form_values['db_name'] = st.text_input("DB名", values.get("db_name", defaults.db_name), key="setting_db_name_v7")
    st.session_state.form_values['db_user'] = st.text_input("DBユーザー", values.get("db_user", defaults.db_user), key="setting_db_user_v7")

def _render_db_auth_settings(values, defaults):
    st.session_state.form_values['db_port'] = st.text_input("DBポート", str(values.get("db_port", defaults.db_port)), key="setting_db_port_v7")
    st.session_state.form_values['db_password'] = st.text_input("DBパスワード", values.get("db_password", defaults.db_password), type="password", key="setting_db_pass_v7")
    fts_opts = ["english", "japanese", "simple", "german", "french"]
    current_fts = values.get("fts_language", defaults.fts_language)
    fts_idx = fts_opts.index(current_fts) if current_fts in fts_opts else 0
    st.session_state.form_values['fts_language'] = st.selectbox("FTS言語", fts_opts, index=fts_idx, key="setting_fts_lang_v7")

def _render_pdf_processing_settings(values, defaults):
    st.markdown("#### 📑 PDF処理設定")
    
    # Azure Document Intelligence設定
    st.markdown("##### Azure Document Intelligence 設定")

    st.session_state.form_values['azure_di_endpoint'] = st.text_input(
        "Azure DI エンドポイント",
        value=values.get("azure_di_endpoint", ""),
        key="setting_azure_di_endpoint_v7",
        placeholder="https://your-resource.cognitiveservices.azure.com/"
    )
    st.session_state.form_values['azure_di_api_key'] = st.text_input(
        "Azure DI APIキー",
        value=values.get("azure_di_api_key", ""),
        type="password",
        key="setting_azure_di_key_v7"
    )

    model_options = ["prebuilt-layout", "prebuilt-document", "prebuilt-read"]
    current_model = values.get("azure_di_model", defaults.azure_di_model)
    if current_model not in model_options:
        current_model = "prebuilt-layout"

    st.session_state.form_values['azure_di_model'] = st.selectbox(
        "使用モデル",
        options=model_options,
        index=model_options.index(current_model),
        key="setting_azure_di_model_v7",
        help="prebuilt-layout: 高精度なレイアウト解析、prebuilt-document: 汎用文書処理、prebuilt-read: OCR特化"
    )

    st.session_state.form_values['save_markdown'] = st.checkbox(
        "Markdownファイルとして保存",
        value=values.get("save_markdown", defaults.save_markdown),
        key="setting_save_markdown_v7",
        help="処理結果をMarkdownファイルとして保存します"
    )

def _apply_settings(form_values):
    from src.ui.state import initialize_rag_system
    try:
        form_values["openai_api_key"] = None
        new_config = Config(**form_values)
        
        with st.spinner("設定を適用し、システムを初期化しています..."):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
            st.cache_resource.clear()
            st.session_state.rag_system = initialize_rag_system(new_config)
        st.success("✅ 設定が正常に適用され、システムが初期化されました！")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 設定の適用中にエラーが発生しました: {type(e).__name__} - {e}")

def _reset_to_defaults(env_defaults):
    from src.ui.state import initialize_rag_system
    st.info("設定をデフォルト値にリセットし、システムを再初期化します...")
    
    default_config = Config()
    for key, value in env_defaults.items():
        if hasattr(default_config, key.lower()):
            setattr(default_config, key.lower(), value)
    default_config.openai_api_key = None

    with st.spinner("デフォルト設定でシステムを初期化しています..."):
        if "rag_system" in st.session_state:
            del st.session_state["rag_system"]
        st.cache_resource.clear()
        st.session_state.rag_system = initialize_rag_system(default_config)
    st.success("✅ 設定がデフォルトにリセットされ、システムが初期化されました！")
    time.sleep(1)
    st.rerun()

def _display_current_config(rag_system):
    if rag_system and hasattr(rag_system, 'config'):
        config_dict = rag_system.config.__dict__.copy()
        sensitive_keys = ["db_password", "openai_api_key", "azure_openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                value = str(config_dict[key])
                config_dict[key] = f"***{value[-4:]}" if len(value) > 7 else "********"
            elif key == "openai_api_key" and key in config_dict:
                config_dict[key] = "None (Fallback Disabled)"

        col1, col2 = st.columns(2)
        items = list(config_dict.items())
        midpoint = (len(items) + 1) // 2
        with col1:
            for k, v in items[:midpoint]:
                st.markdown(f"**{k.replace('_', ' ').capitalize()}:** `{str(v)}`")
        with col2:
            for k, v in items[midpoint:]:
                st.markdown(f"**{k.replace('_', ' ').capitalize()}:** `{str(v)}`")
    else:
        st.info("システムが初期化されていません。上記フォームから設定を適用してください。")
