"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
LEGAL_EXTENSIONS = {".pdf"}


def get_pageindex_client():
    """Create a PageIndex client across SDK versions."""
    try:
        from pageindex import PageIndexClient

        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except ImportError:
        from pageindex import PageIndex

        return PageIndex(api_key=PAGEINDEX_API_KEY)


def get_legal_pdf_files() -> list[Path]:
    """Return source legal PDF files to upload to PageIndex."""
    if not LEGAL_PDF_DIR.exists():
        return []
    return [
        file
        for file in sorted(LEGAL_PDF_DIR.rglob("*"))
        if file.is_file() and file.suffix.lower() in LEGAL_EXTENSIONS
    ]


def upload_documents():
    """
    Upload toàn bộ PDF pháp luật gốc trong data/landing/legal/ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Missing PAGEINDEX_API_KEY. Please set it in .env")

    pi = get_pageindex_client()
    uploaded_docs = {}

    print("Bắt đầu tải các PDF pháp luật lên PageIndex...")
    for pdf_file in get_legal_pdf_files():
        try:
            metadata = {
                "filename": pdf_file.name,
                "source_path": pdf_file.relative_to(LEGAL_PDF_DIR).as_posix(),
                "type": "legal",
                "format": "pdf",
            }

            if hasattr(pi, "submit_document"):
                result = pi.submit_document(str(pdf_file))
            else:
                result = pi.upload(file_path=str(pdf_file), metadata=metadata)

            doc_id = None
            if isinstance(result, dict):
                doc_id = result.get("doc_id") or result.get("id") or result.get("document_id")
            else:
                doc_id = getattr(result, "doc_id", None) or getattr(result, "id", None)

            uploaded_docs[pdf_file.name] = doc_id
            print(f"  ✓ Uploaded: {pdf_file.name} (ID: {doc_id})")
        except Exception as e:
            print(f"  ✕ Lỗi khi tải tệp {pdf_file.name}: {e}")

    print(f"\nHoàn thành! Đã tải lên thành công {len(uploaded_docs)} tài liệu.")
    return uploaded_docs


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Missing PAGEINDEX_API_KEY. Please set it in .env")

    pi = get_pageindex_client()
    if hasattr(pi, "query"):
        raw_results = pi.query(query=query, top_k=top_k)
    else:
        raw_results = pi.search(query=query, top_k=top_k)

    results = []
    for result in raw_results:
        if isinstance(result, dict):
            content = result.get("text") or result.get("content") or result.get("chunk") or ""
            score = result.get("score") or result.get("relevance_score") or 0.0
            metadata = result.get("metadata") or {}
        else:
            content = getattr(result, "text", None) or getattr(result, "content", "")
            score = getattr(result, "score", 0.0)
            metadata = getattr(result, "metadata", {}) or {}

        metadata = {**metadata, "type": metadata.get("type", "legal"), "format": "pdf"}
        results.append({
            "content": content,
            "score": float(score),
            "metadata": metadata,
            "source": "pageindex",
        })

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
