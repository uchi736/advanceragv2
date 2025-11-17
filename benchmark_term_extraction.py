"""
専門用語抽出のパフォーマンス測定スクリプト
仮想ドキュメントを使用して各処理の時間を計測
"""
import asyncio
import time
import tempfile
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Windows環境での絵文字出力対応
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 環境変数読み込み
load_dotenv()

# ロギング設定（詳細ログを有効化）
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
)

from src.rag.config import Config
from src.rag.term_extraction import TermExtractor
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector


def generate_dummy_documents(output_dir: Path, num_docs: int = 10):
    """
    テスト用の仮想ドキュメントを生成

    Args:
        output_dir: 出力ディレクトリ
        num_docs: 生成するドキュメント数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 専門用語を含むサンプルテキスト
    sample_texts = [
        """
        # 船舶エンジニアリング技術文書

        ## 主機関システム
        主機関（Main Engine）は船舶推進の中核をなすシステムであり、SFOC（Specific Fuel Oil Consumption）の
        最適化が重要です。BMSシステム（Ballast Management System）との連携により、燃料消費効率を最大化します。

        ## 排ガス処理装置
        NOx（窒素酸化物）やSOx（硫黄酸化物）の排出削減のため、SCR（Selective Catalytic Reduction）システムを
        採用しています。EGR（Exhaust Gas Recirculation）技術との組み合わせにより、IMO規制値をクリアします。

        ## 電力管理システム
        DG（Diesel Generator）による電力供給と、PMS（Power Management System）による負荷分散制御を実施します。
        UPS（Uninterruptible Power Supply）により、重要システムへの安定供給を確保します。

        ## 自動化制御
        IAS（Integrated Automation System）により、機関室の集中監視制御を実現します。
        PLC（Programmable Logic Controller）ベースの制御システムにより、高度な自動運転が可能です。

        ## 航海計器システム
        GPS（Global Positioning System）、AIS（Automatic Identification System）、ECDIS（Electronic Chart Display
        and Information System）を統合したINS（Integrated Navigation System）により、安全な航海を支援します。

        ## 推進システム
        CPP（Controllable Pitch Propeller）により、可変ピッチ制御を実現します。FPP（Fixed Pitch Propeller）
        と比較して、燃費性能と操船性能が向上します。

        ## 補機システム
        補助ボイラー（Auxiliary Boiler）は、停泊中の蒸気供給を担います。熱交換器（Heat Exchanger）により、
        廃熱回収を行い、エネルギー効率を向上させます。

        ## 冷却システム
        中央冷却水システム（Central Cooling Water System）により、各機器への冷却水供給を一元管理します。
        海水冷却器（Sea Water Cooler）と清水冷却器（Fresh Water Cooler）の二段階冷却を採用しています。
        """,
        """
        # データベース管理技術

        ## リレーショナルデータベース
        RDBMS（Relational Database Management System）は、SQL（Structured Query Language）により
        データ操作を行います。ACID特性（Atomicity, Consistency, Isolation, Durability）により、
        トランザクションの整合性を保証します。

        ## インデックス最適化
        B-Tree インデックスやハッシュインデックスを適切に設計することで、クエリパフォーマンスが向上します。
        カバリングインデックス（Covering Index）により、インデックスオンリースキャンを実現します。

        ## レプリケーション
        マスタースレーブレプリケーション（Master-Slave Replication）により、読み取り負荷を分散します。
        マルチマスターレプリケーション（Multi-Master Replication）では、書き込みの高可用性を実現します。

        ## パーティショニング
        水平パーティショニング（Horizontal Partitioning）により、大規模テーブルを分割管理します。
        垂直パーティショニング（Vertical Partitioning）では、カラム単位での分割を行います。

        ## クエリ最適化
        実行計画（Execution Plan）の分析により、ボトルネックを特定します。統計情報（Statistics）の
        更新により、オプティマイザーの判断精度が向上します。
        """,
        """
        # クラウドインフラストラクチャ

        ## コンテナオーケストレーション
        Kubernetes（K8s）により、コンテナの自動デプロイ、スケーリング、管理を実現します。
        Pod、Service、Deploymentなどのリソースを定義することで、宣言的な構成管理が可能です。

        ## サービスメッシュ
        Istio や Linkerd などのサービスメッシュ（Service Mesh）により、マイクロサービス間の
        通信制御、監視、セキュリティを強化します。

        ## CI/CD パイプライン
        GitLab CI、GitHub Actions、Jenkins などを使用した継続的インテグレーション（CI）と
        継続的デリバリー（CD）により、デプロイメントを自動化します。

        ## インフラストラクチャ as Code
        Terraform、CloudFormation、Ansible などを使用したIaC（Infrastructure as Code）により、
        インフラの構成管理をコード化します。

        ## 監視とロギング
        Prometheus、Grafana によるメトリクス監視、ELK スタック（Elasticsearch, Logstash, Kibana）
        によるログ集約・分析を実施します。
        """
    ]

    print(f"🔧 生成中: {num_docs}件の仮想ドキュメント")

    for i in range(num_docs):
        # テキストをローテーション
        text = sample_texts[i % len(sample_texts)]

        # ドキュメントごとに少し内容を変える
        doc_text = f"# ドキュメント {i+1}\n\n" + text + f"\n\n## 追加セクション {i+1}\n専門用語抽出テスト用の追加コンテンツです。"

        # ファイル保存
        file_path = output_dir / f"test_doc_{i+1:03d}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc_text)

    print(f"✅ 完了: {output_dir} に {num_docs}件のドキュメントを生成")
    return output_dir


class PerformanceTimer:
    """処理時間測定用のコンテキストマネージャー"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  [{self.name}] 開始...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"✅ [{self.name}] 完了: {elapsed:.2f}秒")
        return False

    @property
    def elapsed(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


async def benchmark_term_extraction(num_docs: int = 10):
    """
    専門用語抽出のベンチマーク実行

    Args:
        num_docs: テストするドキュメント数
    """
    print("=" * 80)
    print(f"🚀 専門用語抽出パフォーマンス測定")
    print(f"📄 ドキュメント数: {num_docs}件")
    print("=" * 80)

    # タイマー記録用
    timers = {}

    # 設定読み込み
    with PerformanceTimer("設定読み込み") as timer:
        config = Config()
    timers["設定読み込み"] = timer.elapsed

    # LLMとEmbeddings初期化
    with PerformanceTimer("LLM/Embeddings初期化") as timer:
        llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_deployment=config.azure_openai_chat_deployment_name,
            temperature=0.1,
        )

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_deployment=config.azure_openai_embedding_deployment_name
        )
    timers["LLM/Embeddings初期化"] = timer.elapsed

    # ベクトルストア初期化
    with PerformanceTimer("ベクトルストア初期化") as timer:
        pg_url = config.pgvector_connection_string
        vector_store = PGVector(
            collection_name=config.collection_name,
            connection_string=pg_url,
            embedding_function=embeddings,
            pre_delete_collection=False
        )
    timers["ベクトルストア初期化"] = timer.elapsed

    # 仮想ドキュメント生成
    with PerformanceTimer("仮想ドキュメント生成") as timer:
        temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        doc_dir = generate_dummy_documents(temp_dir, num_docs)
    timers["仮想ドキュメント生成"] = timer.elapsed

    try:
        # TermExtractor初期化
        with PerformanceTimer("TermExtractor初期化") as timer:
            extractor = TermExtractor(
                config=config,
                llm=llm,
                embeddings=embeddings,
                vector_store=vector_store,
                pg_url=pg_url,
                jargon_table_name=config.jargon_table_name
            )
        timers["TermExtractor初期化"] = timer.elapsed

        # ファイルリスト取得
        files = list(doc_dir.glob("*.txt"))
        print(f"\n📁 処理対象ファイル: {len(files)}件")

        # 専門用語抽出実行（詳細計測）
        print("\n" + "=" * 80)
        print("📊 専門用語抽出プロセス開始")
        print("=" * 80)

        total_start = time.time()

        # extract_from_documents の内部処理を手動で計測
        all_chunks = []
        per_document_texts = []

        # 1. ドキュメント読み込みとチャンク分割
        with PerformanceTimer("1. ドキュメント読み込み＆チャンク分割") as timer:
            for file_path in files:
                loader = extractor._get_loader(file_path)
                docs = loader.load()
                chunks = extractor.text_splitter.split_documents(docs)
                all_chunks.extend([c.page_content for c in chunks])

                doc_text = "\n".join([c.page_content for c in chunks])
                per_document_texts.append({
                    "file_path": str(file_path),
                    "text": doc_text
                })
        timers["1. ドキュメント読み込み＆チャンク分割"] = timer.elapsed

        print(f"   📝 総チャンク数: {len(all_chunks)}")
        print(f"   📄 ドキュメント数: {len(per_document_texts)}")

        # 2. 統計的候補抽出
        all_candidates = {}
        with PerformanceTimer("2. 統計的候補抽出") as timer:
            from collections import defaultdict
            all_candidates = defaultdict(int)
            document_candidate_map = {}

            for doc_info in per_document_texts:
                file_path = doc_info["file_path"]
                text = doc_info["text"]

                doc_candidates = extractor.statistical_extractor.extract_candidates(text)
                document_candidate_map[file_path] = doc_candidates

                for term, freq in doc_candidates.items():
                    all_candidates[term] += freq
        timers["2. 統計的候補抽出"] = timer.elapsed
        print(f"   🔍 抽出候補数: {len(all_candidates)}")

        # 3. TF-IDF + C-value 計算
        with PerformanceTimer("3. TF-IDF + C-value 計算") as timer:
            full_text = "\n".join([doc["text"] for doc in per_document_texts])
            documents = extractor._split_into_sentences(full_text)

            tfidf_scores = extractor.statistical_extractor.calculate_tfidf(documents, all_candidates)
            cvalue_scores = extractor.statistical_extractor.calculate_cvalue(all_candidates, full_text=full_text)
        timers["3. TF-IDF + C-value 計算"] = timer.elapsed

        # 4. スコア計算
        with PerformanceTimer("4. 基底スコア計算") as timer:
            seed_scores = extractor.statistical_extractor.calculate_combined_scores(
                tfidf_scores, cvalue_scores, stage="seed"
            )
            base_scores = extractor.statistical_extractor.calculate_combined_scores(
                tfidf_scores, cvalue_scores, stage="final"
            )
        timers["4. 基底スコア計算"] = timer.elapsed

        # 5. SemReRank候補選択
        with PerformanceTimer("5. SemReRank候補選択") as timer:
            MAX_SEMRERANK_CANDIDATES = getattr(config, 'max_semrerank_candidates', 1500)

            if len(all_candidates) > MAX_SEMRERANK_CANDIDATES:
                sorted_candidates = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
                top_candidates = dict(sorted_candidates[:MAX_SEMRERANK_CANDIDATES])
                candidates_for_semrerank = {k: all_candidates[k] for k in top_candidates.keys()}
                seed_scores_for_semrerank = {k: seed_scores[k] for k in top_candidates.keys()}
                base_scores_for_semrerank = top_candidates
            else:
                candidates_for_semrerank = all_candidates
                seed_scores_for_semrerank = seed_scores
                base_scores_for_semrerank = base_scores
        timers["5. SemReRank候補選択"] = timer.elapsed
        print(f"   🎯 SemReRank対象: {len(candidates_for_semrerank)}/{len(all_candidates)}")

        # 6. SemReRank実行
        enhanced_scores = base_scores_for_semrerank
        if extractor.semrerank:
            with PerformanceTimer("6. SemReRank実行") as timer:
                try:
                    enhanced_scores = extractor.semrerank.enhance_scores(
                        candidates=list(candidates_for_semrerank.keys()),
                        base_scores=base_scores_for_semrerank,
                        seed_scores=seed_scores_for_semrerank
                    )
                except Exception as e:
                    print(f"   ⚠️  SemReRank失敗: {e}")
                    enhanced_scores = base_scores
            timers["6. SemReRank実行"] = timer.elapsed
        else:
            print("   ⏭️  SemReRank無効")
            timers["6. SemReRank実行"] = 0

        # 7. 類義語・関連語検出
        with PerformanceTimer("7. 類義語・関連語検出") as timer:
            synonym_map = extractor.statistical_extractor.detect_variants(
                candidates=list(candidates_for_semrerank.keys())
            )
            related_map = extractor.statistical_extractor.detect_related_terms(
                candidates=list(candidates_for_semrerank.keys()),
                full_text=full_text,
                max_related=config.max_related_terms_per_candidate,
                min_term_length=config.min_related_term_length
            )
        timers["7. 類義語・関連語検出"] = timer.elapsed

        # 8. ExtractedTermオブジェクト化
        with PerformanceTimer("8. ExtractedTerm化＆ソート") as timer:
            from src.rag.advanced_term_extraction import ExtractedTerm
            terms = [
                ExtractedTerm(
                    term=term,
                    score=enhanced_scores[term],
                    tfidf_score=tfidf_scores.get(term, 0.0),
                    cvalue_score=cvalue_scores.get(term, 0.0),
                    frequency=all_candidates.get(term, 0),
                    variants=synonym_map.get(term, []),
                    related_terms=related_map.get(term, [])
                )
                for term in enhanced_scores
            ]
            terms.sort(key=lambda x: x.score, reverse=True)
        timers["8. ExtractedTerm化＆ソート"] = timer.elapsed
        print(f"   📋 総用語数: {len(terms)}")

        # 9. 軽量LLMフィルタ
        abbreviations = [t for t in terms if extractor._is_abbreviation(t.term)]
        non_abbreviations = [t for t in terms if not extractor._is_abbreviation(t.term)]

        print(f"   🔤 略語: {len(abbreviations)}, 非略語: {len(non_abbreviations)}")

        if config.enable_lightweight_filter and llm:
            with PerformanceTimer("9. 軽量LLMフィルタ") as timer:
                definition_percentile = getattr(config, 'definition_generation_percentile', 50.0)
                n_candidates = max(1, int(len(non_abbreviations) * definition_percentile / 100))
                candidate_terms = non_abbreviations[:n_candidates]

                filtered_terms = await extractor._lightweight_llm_filter(candidate_terms)
                terms_for_definition = abbreviations + filtered_terms
            timers["9. 軽量LLMフィルタ"] = timer.elapsed
            print(f"   ✅ 通過: {len(filtered_terms)}/{len(candidate_terms)}")
        else:
            definition_percentile = 50.0
            n_candidates = max(1, int(len(non_abbreviations) * definition_percentile / 100))
            terms_for_definition = abbreviations + non_abbreviations[:n_candidates]
            timers["9. 軽量LLMフィルタ"] = 0
            print(f"   ⏭️  軽量フィルタ無効")

        print(f"   🎯 定義生成対象: {len(terms_for_definition)}")

        # 10. RAG定義生成（バルク処理）
        if vector_store and llm:
            with PerformanceTimer("10. RAG定義生成（バルク処理）") as timer:
                await extractor._bulk_generate_definitions(terms_for_definition)
            timers["10. RAG定義生成（バルク処理）"] = timer.elapsed

            defined_count = sum(1 for t in terms_for_definition if t.definition)
            print(f"   ✅ 定義生成完了: {defined_count}/{len(terms_for_definition)}")
        else:
            timers["10. RAG定義生成（バルク処理）"] = 0
            print(f"   ⏭️  定義生成スキップ")

        # 11. 重量LLMフィルタ
        if llm:
            with PerformanceTimer("11. 重量LLMフィルタ") as timer:
                from src.rag.prompts import get_technical_term_judgment_prompt
                from langchain_core.output_parsers import StrOutputParser

                terms_with_def = [t for t in terms if t.definition]
                technical_terms = []

                if terms_with_def:
                    prompt = get_technical_term_judgment_prompt()
                    chain = prompt | llm | StrOutputParser()

                    batch_size = config.llm_filter_batch_size

                    for i in range(0, len(terms_with_def), batch_size):
                        batch = terms_with_def[i:i+batch_size]
                        batch_inputs = [{"term": t.term, "definition": t.definition} for t in batch]

                        try:
                            result_texts = await chain.abatch(batch_inputs)
                            for term, result_text in zip(batch, result_texts):
                                result = extractor._parse_llm_json(result_text)
                                if result and result.get("is_technical", False):
                                    term.metadata["confidence"] = result.get("confidence", 0.0)
                                    term.metadata["reason"] = result.get("reason", "")
                                    technical_terms.append(term)
                        except Exception as e:
                            print(f"   ⚠️  バッチ失敗: {e}")

            timers["11. 重量LLMフィルタ"] = timer.elapsed
            print(f"   ✅ 専門用語: {len(technical_terms)}/{len(terms_with_def)}")
        else:
            timers["11. 重量LLMフィルタ"] = 0
            print(f"   ⏭️  重量フィルタスキップ")

        total_elapsed = time.time() - total_start
        timers["総処理時間"] = total_elapsed

    finally:
        # 一時ディレクトリ削除
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 処理時間サマリー")
    print("=" * 80)

    max_label_len = max(len(label) for label in timers.keys())

    for label, elapsed in timers.items():
        if elapsed is not None and elapsed > 0:
            percentage = (elapsed / total_elapsed * 100) if label != "総処理時間" else 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"{label:<{max_label_len}} : {elapsed:7.2f}秒 {bar} {percentage:5.1f}%")

    print("=" * 80)

    # ボトルネック特定
    print("\n🔍 ボトルネック分析")
    print("-" * 80)

    # 総処理時間を除外してソート
    processing_timers = {k: v for k, v in timers.items() if k != "総処理時間" and v > 0}
    sorted_timers = sorted(processing_timers.items(), key=lambda x: x[1], reverse=True)

    print("\n⚠️  処理時間TOP5:")
    for i, (label, elapsed) in enumerate(sorted_timers[:5], 1):
        percentage = elapsed / total_elapsed * 100
        print(f"  {i}. {label}: {elapsed:.2f}秒 ({percentage:.1f}%)")

    print("\n" + "=" * 80)
    print(f"✅ ベンチマーク完了: 総処理時間 {total_elapsed:.2f}秒")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # ドキュメント数を引数から取得（デフォルト10件）
    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    asyncio.run(benchmark_term_extraction(num_docs))
