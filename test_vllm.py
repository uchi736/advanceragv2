#!/usr/bin/env python3
"""
test_vllm.py
============
VLLM接続テストスクリプト
"""

import sys
import time
import requests
from statistics import mean, stdev
from dotenv import load_dotenv

# プロジェクトのルートディレクトリをパスに追加
sys.path.append('c:/work/RAG/advancedrag2')

from src.rag.config import Config
from src.rag.vllm_client import VLLMClient, VLLMChatClient


def test_vllm_connection():
    """VLLM接続テスト"""
    # 設定を読み込み
    load_dotenv()
    cfg = Config()

    print("=" * 60)
    print("VLLM Connection Test")
    print("=" * 60)
    print(f"USE_VLLM: {cfg.use_vllm}")
    print(f"Endpoint: {cfg.vllm_endpoint}")
    print(f"Temperature: {cfg.llm_temperature}")
    print(f"Top-P: {cfg.top_p}")
    print(f"Top-K: {cfg.top_k}")
    print(f"Min-P: {cfg.min_p}")
    print(f"Reasoning Effort: {cfg.vllm_reasoning_effort}")
    print("-" * 60)

    try:
        # VLLMClientを初期化
        client = VLLMClient(
            endpoint=cfg.vllm_endpoint,
            temperature=cfg.llm_temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
            reasoning_effort=cfg.vllm_reasoning_effort,
            timeout=30
        )

        # テストプロンプト
        test_prompts = [
            "こんにちは。簡単に自己紹介してください。",
            "1から5までを数えてください。",
            "Pythonとは何ですか？一文で説明してください。"
        ]

        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}]: {test_prompt}")
            print("[Sending request...]")

            start_time = time.time()

            # 実行
            response = client.invoke(test_prompt)

            end_time = time.time()
            latency = end_time - start_time

            print(f"[OK] Success! (Latency: {latency:.2f}s)")
            print(f"[Response]: {response[:200]}{'...' if len(response) > 200 else ''}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vllm_health(endpoint: str) -> bool:
    """VLLMサーバーのヘルスチェック"""
    try:
        # OpenAI互換のモデルリストエンドポイントをチェック
        response = requests.get(f"{endpoint}/models", timeout=5)
        if response.status_code == 200:
            print("[OK] VLLM server is healthy")
            data = response.json()
            if "data" in data and data["data"]:
                print(f"   Available models: {[m.get('id', 'unknown') for m in data['data']]}")
            return True
        else:
            print(f"⚠️ VLLM server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to VLLM server (Connection refused)")
        return False
    except requests.exceptions.Timeout:
        print("[ERROR] VLLM server health check timeout")
        return False
    except Exception as e:
        print(f"[ERROR] VLLM health check error: {e}")
        return False


def benchmark_vllm(client: VLLMClient, num_tests: int = 3):
    """パフォーマンステスト"""
    test_prompts = [
        "1から10までを数えてください。",
        "日本の首都はどこですか？",
        "機械学習とは何ですか？簡潔に説明してください。",
    ]

    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    latencies = []

    for prompt in test_prompts:
        prompt_latencies = []
        print(f"\nPrompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        for i in range(num_tests):
            try:
                start = time.time()
                response = client.invoke(prompt)
                end = time.time()

                latency = end - start
                prompt_latencies.append(latency)
                latencies.append(latency)
                print(f"  Test {i+1}: {latency:.2f}s")
            except Exception as e:
                print(f"  Test {i+1}: Failed - {e}")

        if prompt_latencies:
            print(f"  Average: {mean(prompt_latencies):.2f}s")

    if len(latencies) > 1:
        print("\n" + "-" * 60)
        print(f"Overall Average Latency: {mean(latencies):.2f}s")
        print(f"Standard Deviation: {stdev(latencies):.2f}s")
        print(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")


def test_chat_client():
    """VLLMChatClient のテスト（ChatModel互換性確認）"""
    load_dotenv()
    cfg = Config()

    print("\n" + "=" * 60)
    print("VLLMChatClient Test (ChatModel Compatibility)")
    print("=" * 60)

    try:
        # VLLMChatClientを初期化
        chat_client = VLLMChatClient(
            endpoint=cfg.vllm_endpoint,
            temperature=cfg.llm_temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
            reasoning_effort=cfg.vllm_reasoning_effort,
            timeout=30
        )

        test_prompt = "AIの将来について簡潔に述べてください。"
        print(f"[Test Prompt]: {test_prompt}")

        # ChatModel互換のinvokeメソッドをテスト
        response = chat_client.invoke(test_prompt)

        if hasattr(response, 'content'):
            print(f"[OK] ChatModel compatible response received")
            print(f"[Response]: {response.content[:200]}...")
            return True
        else:
            print("[ERROR] Response does not have 'content' attribute")
            return False

    except Exception as e:
        print(f"[ERROR] ChatClient test failed: {e}")
        return False


if __name__ == "__main__":
    load_dotenv()
    cfg = Config()

    # ヘルスチェック
    print("\n[Health Check] Checking VLLM server health...")
    health_ok = check_vllm_health(cfg.vllm_endpoint)

    if not health_ok:
        print("\n[WARNING] VLLM server is not available. Please check:")
        print(f"  1. Is the server running at {cfg.vllm_endpoint}?")
        print("  2. Is the port correct? (Expected: 8004)")
        print("  3. Try: vllm serve <model> --host 0.0.0.0 --port 8004")
        sys.exit(1)

    # 接続テスト
    print("\n[Connection Test] Testing VLLM connection...")
    success = test_vllm_connection()

    if success:
        # ChatClient互換性テスト
        print("\n[Compatibility Test] Testing ChatModel compatibility...")
        chat_success = test_chat_client()

        # パフォーマンステスト
        print("\n[Performance] Running performance benchmark...")
        client = VLLMClient(
            endpoint=cfg.vllm_endpoint,
            temperature=cfg.llm_temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
            reasoning_effort=cfg.vllm_reasoning_effort,
            timeout=30
        )
        benchmark_vllm(client, num_tests=3)

        print("\n" + "=" * 60)
        print("[SUCCESS] All tests completed successfully!")
        print("=" * 60)
    else:
        print("\n[WARNING] Basic connection test failed. Please check your configuration.")
        sys.exit(1)