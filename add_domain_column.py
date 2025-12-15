"""
domain列を追加するマイグレーションスクリプト
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# データベース接続
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL が設定されていません")

engine = create_engine(DATABASE_URL)

print("domain列を追加します...")

try:
    with engine.begin() as conn:
        # domain列を追加
        conn.execute(text("""
            ALTER TABLE jargon_dictionary
            ADD COLUMN IF NOT EXISTS domain TEXT;
        """))
        print("✅ domain列を追加しました")

        # 確認
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'jargon_dictionary'
            AND column_name = 'domain';
        """))

        row = result.fetchone()
        if row:
            print(f"✅ 確認完了: domain列 ({row[1]}型) が存在します")
        else:
            print("⚠️ domain列が見つかりません")

except Exception as e:
    print(f"❌ エラー: {e}")
    raise

print("\n完了しました。アプリを再起動してください。")
