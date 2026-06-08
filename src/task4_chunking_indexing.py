"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import os
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # Vì sao? Multilingual, tốt cho tiếng Việt
EMBEDDING_DIM = 1536

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": relative_path.as_posix(),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        def split_text(text: str) -> list[str]:
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            return [
                text[start:start + CHUNK_SIZE]
                for start in range(0, len(text), step)
                if text[start:start + CHUNK_SIZE].strip()
            ]

    for doc in documents:
        splits = split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API in .env")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks via OpenRouter API...")
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_MODEL
        )
        for j, data in enumerate(response.data):
            chunks[i+j]["embedding"] = data.embedding
            
    return chunks



def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.init import Auth

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    try:
        if client.collections.exists("DrugLawDocsOpenAI"):
            collection = client.collections.get("DrugLawDocsOpenAI")
        else:
            collection = client.collections.create(
                name="DrugLawDocsOpenAI",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="source_path", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                ],
            )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "source_path": metadata.get("source_path", ""),
                        "doc_type": metadata.get("type", ""),
                    },
                    vector=chunk["embedding"],
                )
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
