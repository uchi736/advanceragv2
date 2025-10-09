import re
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.runnables import RunnableSequence
from langchain_community.callbacks.manager import get_openai_callback

class SQLHandler:
    def __init__(self, config, llm, connection_string, engine: Optional[Engine] = None):
        self.config = config
        self.llm = llm
        self.connection_string = connection_string
        self.engine: Engine = engine or create_engine(connection_string)
        # Note: Prompts/chains will be passed in or set after initialization
        # to avoid circular dependencies with a potential chains.py module.
        self.single_table_sql_chain: Optional[RunnableSequence] = None
        self.multi_table_sql_chain: Optional[RunnableSequence] = None
        self.sql_answer_generation_chain: Optional[RunnableSequence] = None
        self._cached_multi_table_schema: Optional[str] = None

    @property
    def multi_table_schema(self) -> str:
        """Get schema for all user tables (cached)."""
        if self._cached_multi_table_schema:
            return self._cached_multi_table_schema

        tables_data = self.get_data_tables()
        if not tables_data:
            return "No user tables found."

        schema_parts = []
        for table in tables_data:
            schema_parts.append(table.get('schema', ''))

        self._cached_multi_table_schema = "\n\n".join(schema_parts)
        return self._cached_multi_table_schema

    def create_table_from_file(self, file_path: str, table_name: Optional[str] = None) -> tuple[bool, str, str]:
        try:
            path = Path(file_path)
            if not path.exists(): return False, f"File not found: {file_path}", ""

            if not table_name:
                clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', path.stem.lower())
                table_name = f"{self.config.user_table_prefix}{clean_stem}"

            df = self._read_file_to_dataframe(file_path)
            if df is None or df.empty: return False, "File is empty or could not be read.", ""

            df.columns = self._normalize_columns(df.columns)
            
            with self.engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                df.to_sql(table_name, conn, if_exists='replace', index=False, schema='public')
                conn.commit()

            schema_info = self._get_table_schema(table_name)
            return True, f"Table '{table_name}' created with {len(df)} rows.", schema_info
        except Exception as e:
            return False, f"Table creation error: {type(e).__name__} - {e}", ""

    def _read_file_to_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        path = Path(file_path)
        if path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif path.suffix.lower() == '.csv':
            for enc in ['utf-8', 'shift_jis', 'cp932', 'latin1']:
                try:
                    return pd.read_csv(file_path, encoding=enc, header=0, skip_blank_lines=True)
                except (UnicodeDecodeError, Exception):
                    continue
        return None

    def _normalize_columns(self, columns: pd.Index) -> List[str]:
        normalized = []
        for i, col in enumerate(columns):
            col_str = str(col) if col is not None else ""
            if not col_str or col_str.startswith('Unnamed:') or col_str.strip() == '':
                normalized.append(f'col_{i}')
                continue
            
            norm_col = re.sub(r'\s+', '_', col_str.strip())
            if norm_col and norm_col[0].isdigit():
                norm_col = f'col_{norm_col}'
            
            normalized.append(norm_col if norm_col else f'col_{i}')
        return normalized

    def get_data_tables(self) -> List[Dict[str, Any]]:
        tables_data = []
        try:
            with self.engine.connect() as conn:
                res = conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE :prefix"),
                    {"prefix": f"{self.config.user_table_prefix}%"}
                )
                user_tables = [row[0] for row in res if row and row[0]]

                for table_name in user_tables:
                    count_res = conn.execute(text(f'SELECT COUNT(*) FROM public."{table_name}"')).scalar_one_or_none()
                    tables_data.append({
                        "table_name": table_name,
                        "row_count": count_res or 0,
                        "schema": self._get_table_schema(table_name)
                    })
            return tables_data
        except Exception as e:
            print(f"Error getting data tables: {e}")
            return []

    def delete_data_table(self, table_name: str) -> tuple[bool, str]:
        if not table_name or not table_name.startswith(self.config.user_table_prefix):
            return False, f"Invalid table name: {table_name}"
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                conn.commit()
            return True, f"Table '{table_name}' deleted successfully."
        except Exception as e:
            return False, f"Table deletion error: {e}"

    def _get_table_schema(self, table_name: str) -> str:
        try:
            with self.engine.connect() as conn:
                cols = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table AND table_schema = 'public' ORDER BY ordinal_position"), {"table": table_name}).fetchall()
                if not cols: return f"Table '{table_name}' not found."

                schema = f"Table: \"{table_name}\"\nColumns:\n" + "\n".join([f"  - \"{c_name}\": {c_type}" for c_name, c_type in cols])
                
                sample_rows = conn.execute(text(f'SELECT * FROM public."{table_name}" LIMIT 3')).fetchall()
                if sample_rows:
                    schema += "\nSample data:\n" + pd.DataFrame(sample_rows, columns=[c[0] for c in cols]).to_string(index=False)
                return schema
        except Exception as e:
            return f"Schema retrieval error: {e}"

    def _execute_and_summarize_sql(self, original_question: str, generated_sql: str, config=None) -> Dict[str, Any]:
        if not generated_sql:
            return {"success": False, "error": "No SQL query provided."}
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text(generated_sql))
                rows = res.fetchmany(self.config.max_sql_results)
                results_df = pd.DataFrame(rows, columns=res.keys())

            preview_str = results_df.head(self.config.max_sql_preview_rows_for_llm).to_string(index=False)
            if len(results_df) > self.config.max_sql_preview_rows_for_llm:
                preview_str += f"\n... and {len(results_df) - self.config.max_sql_preview_rows_for_llm} more rows."

            with get_openai_callback() as cb:
                payload = {
                    "original_question": original_question, "sql_query": generated_sql,
                    "sql_results_preview_str": preview_str, "max_preview_rows": self.config.max_sql_preview_rows_for_llm,
                    "total_row_count": len(results_df)
                }
                answer = self.sql_answer_generation_chain.invoke(payload, config=config)
                usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}

            return {
                "success": True, "generated_sql": generated_sql, "natural_language_answer": answer,
                "results_preview": results_df.head(20).to_dict('records'), "row_count_fetched": len(results_df),
                "columns": results_df.columns.tolist(), "usage": usage
            }
        except Exception as e:
            return {"success": False, "error": str(e), "generated_sql": generated_sql}

    def _extract_sql(self, llm_output: str) -> str:
        match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        
        lines = [line for line in llm_output.strip().split('\n') if line.strip().upper().startswith("SELECT")]
        return "\n".join(lines).strip() if lines else llm_output.strip()

    def get_chunks_by_document_id(self, document_id: str) -> pd.DataFrame:
        """Retrieves all chunks for a given document ID."""
        if not document_id:
            return pd.DataFrame()
        
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT chunk_id, content, tokenized_content, metadata
                    FROM document_chunks 
                    WHERE document_id = :doc_id AND collection_name = :coll_name
                    ORDER BY chunk_id
                """)
                df = pd.read_sql(query, conn, params={"doc_id": document_id, "coll_name": self.config.collection_name})
            return df
        except Exception as e:
            print(f"Error getting chunks for document {document_id}: {e}")
            return pd.DataFrame()
