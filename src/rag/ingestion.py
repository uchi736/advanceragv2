"""
Document ingestion handler with improved batch processing and connection management.
"""
import json
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from src.rag.text_processor import JapaneseTextProcessor
from sqlalchemy import text


class IngestionHandler:
    def __init__(self, config, vector_store, engine, text_processor=None):
        self.config = config
        self.vector_store = vector_store
        self.engine = engine
        self.text_processor = text_processor if text_processor else JapaneseTextProcessor()

        # Initialize Azure Document Intelligence processor
        from src.rag.pdf_processors.azure_di_processor import AzureDocumentIntelligenceProcessor
        self.pdf_processor = AzureDocumentIntelligenceProcessor(config)
        if not self.pdf_processor:
            print("Warning: Azure Document Intelligence processor could not be initialized")

    def load_documents(self, paths: List[str]) -> List[Document]:
        all_docs = []
        for p_str in paths:
            path = Path(p_str)
            if not path.exists():
                print(f"Path {path} does not exist.")
                continue

            # Load documents based on file type
            docs = self._load_single_document(str(path))
            all_docs.extend(docs)

        return all_docs

    def _load_single_document(self, path: str) -> List[Document]:
        """Load a single document and return a list of Document objects."""
        docs = []
        p = Path(path)
        suf = p.suffix.lower()

        try:
            if suf == ".pdf":
                # Use Azure Document Intelligence processor
                docs = self.pdf_processor.process(path)
            elif suf in {".txt", ".md"}:
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif suf == ".docx":
                docs.extend(Docx2txtLoader(path).load())
            # Add more formats as needed
        except Exception as e:
            print(f"Error loading {path}: {type(e).__name__} - {e}")

        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        return self._chunk_documents_standard(docs)

    def _chunk_documents_standard(self, docs: List[Document]) -> List[Document]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Fallback splitter for content without headers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        all_chunks = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}")
            doc_id = Path(src).name
            chunk_counter = 0  # Reset counter for each document

            try:
                # Normalize text content first
                normalized_content = self.text_processor.normalize_text(d.page_content)

                # Split by markdown headers
                md_header_splits = markdown_splitter.split_text(normalized_content)

                # Further split chunks that are too large
                doc_chunks = []
                for split in md_header_splits:
                    if len(split.page_content) > self.config.chunk_size:
                        sub_chunks = text_splitter.split_documents([split])
                        for sub_chunk in sub_chunks:
                            # Ensure metadata from header split is preserved
                            sub_chunk.metadata.update(split.metadata)
                            doc_chunks.append(sub_chunk)
                    else:
                        doc_chunks.append(split)

                # Add metadata to chunks for this document
                for chunk in doc_chunks:
                    chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{chunk_counter}"
                    chunk.metadata["document_id"] = doc_id
                    chunk.metadata["source"] = src
                    chunk.metadata["chunk_index"] = chunk_counter
                    chunk.metadata["is_parent"] = False
                    chunk_counter += 1
                    all_chunks.append(chunk)

            except Exception as e:
                print(f"Error chunking document {doc_id}: {e}")
                continue

        return all_chunks

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        """Store chunks in PostgreSQL for keyword search."""
        if not chunks or not self.engine:
            return

        sql = text("""
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, tokenized_content, metadata, created_at)
            VALUES(:coll_name, :doc_id, :cid, :cont, :tok_cont, :meta, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content = EXCLUDED.content, tokenized_content = EXCLUDED.tokenized_content,
                metadata = EXCLUDED.metadata, document_id = EXCLUDED.document_id,
                collection_name = EXCLUDED.collection_name, created_at = CURRENT_TIMESTAMP;
        """)

        try:
            with self.engine.connect() as conn, conn.begin():
                for c in chunks:
                    normalized_content = self.text_processor.normalize_text(c.page_content)
                    tokenized_content = self.text_processor.tokenize(normalized_content) if self.config.enable_japanese_search else ""
                    conn.execute(sql, {
                        "coll_name": self.config.collection_name,
                        "doc_id": c.metadata["document_id"],
                        "cid": c.metadata["chunk_id"],
                        "cont": normalized_content,
                        "tok_cont": " ".join(tokenized_content),
                        "meta": json.dumps(c.metadata or {})
                    })
        except Exception as e:
            print(f"Error storing chunks for keyword search: {type(e).__name__} - {e}")

    def ingest_documents(self, paths: List[str]):
        """Ingest documents with improved batch processing and error handling."""
        BATCH_SIZE = 50  # Process in smaller batches to avoid connection timeout
        total_chunks_processed = 0
        failed_files = []
        successful_files = []

        for path_idx, path in enumerate(paths):
            try:
                print(f"\nProcessing file {path_idx + 1}/{len(paths)}: {path}")

                # Load single document
                docs = self._load_single_document(path)
                if not docs:
                    print(f"No documents loaded from {path}")
                    continue

                # Chunk the document
                print(f"Chunking {len(docs)} document(s) from {path}...")
                chunks = self.chunk_documents(docs)

                # Filter valid chunks
                valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

                if not valid_chunks:
                    print(f"No valid chunks from {path}")
                    continue

                # Process in batches
                print(f"Ingesting {len(valid_chunks)} chunks in batches of {BATCH_SIZE}...")
                file_processed = False

                for i in range(0, len(valid_chunks), BATCH_SIZE):
                    batch = valid_chunks[i:i+BATCH_SIZE]
                    batch_ids = [c.metadata['chunk_id'] for c in batch]

                    try:
                        # Store batch in vector store
                        self.vector_store.add_documents(batch, ids=batch_ids)

                        # Store batch for keyword search
                        self._store_chunks_for_keyword_search(batch)

                        batch_end = min(i + BATCH_SIZE, len(valid_chunks))
                        print(f"  Processed chunks {i+1}-{batch_end}/{len(valid_chunks)}")
                        total_chunks_processed += len(batch)
                        file_processed = True

                    except Exception as batch_error:
                        print(f"  Error processing batch {i//BATCH_SIZE + 1}: {batch_error}")
                        # Try to reconnect for next batch
                        if self.engine:
                            try:
                                self.engine.dispose()  # Close all connections
                                print("  Reconnecting to database...")
                            except:
                                pass
                        continue

                if file_processed:
                    successful_files.append(path)
                    print(f"✓ Successfully processed {path}")
                else:
                    failed_files.append(path)
                    print(f"✗ Failed to process {path}")

            except Exception as e:
                print(f"Error processing {path}: {type(e).__name__} - {e}")
                failed_files.append(path)
                continue  # Continue with next file

        # Summary
        print(f"\n{'='*60}")
        print(f"Ingestion complete:")
        print(f"  Total chunks processed: {total_chunks_processed}")
        print(f"  Successful files: {len(successful_files)}/{len(paths)}")
        if failed_files:
            print(f"  Failed files ({len(failed_files)}):")
            for f in failed_files:
                print(f"    - {f}")
        print(f"{'='*60}")

    def delete_document_by_id(self, doc_id: str) -> tuple[bool, str]:
        """Delete a document and all its chunks."""
        if not doc_id:
            return False, "Document ID cannot be empty."

        try:
            # Delete from vector store
            if self.vector_store:
                # Get all chunk IDs for this document
                if self.engine:
                    with self.engine.connect() as conn:
                        result = conn.execute(
                            text("SELECT chunk_id FROM document_chunks WHERE document_id = :doc_id"),
                            {"doc_id": doc_id}
                        )
                        chunk_ids = [row[0] for row in result]

                        if chunk_ids:
                            # Delete from vector store
                            self.vector_store.delete(chunk_ids)

                        # Delete from document_chunks table
                        conn.execute(
                            text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                            {"doc_id": doc_id}
                        )
                        conn.commit()

            return True, f"Successfully deleted document: {doc_id}"

        except Exception as e:
            return False, f"Error deleting document: {str(e)}"
