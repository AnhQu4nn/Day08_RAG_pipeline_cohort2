"""
Convert toàn bộ file trong data/landing/ thành Markdown.

Pipeline:
    1. data/landing/legal/
        - PDF có text sẵn (digital) → PyMuPDF extract text trực tiếp (nhanh, không cần OCR)
        - PDF scan (không có text)   → OCR từng trang bằng pymupdf + pytesseract (tiếng Việt)
        - DOCX/DOC                   → python-docx
        - Xử lý từng trang một, giải phóng RAM sau mỗi trang → tránh std::bad_alloc

    2. data/landing/news/
        - JSON → Markdown (title, url, date_crawled, content_markdown)

Output:
    data/standardized/
    Giữ nguyên cấu trúc thư mục.

Cài đặt (Windows):
    1. Cài Tesseract:
       - Tải installer tại: https://github.com/UB-Mannheim/tesseract/wiki
       - Chọn bản: tesseract-ocr-w64-setup-*.exe
       - Trong installer: tick "Additional language data" → chọn "Vietnamese"
       - Cài vào mặc định: C:\\Program Files\\Tesseract-OCR\\
    2. pip install pymupdf pytesseract python-docx pillow

Nếu Tesseract cài vào thư mục khác, đặt biến môi trường:
    set TESSERACT_PATH=C:\\đường\\dẫn\\của\\bạn\\tesseract.exe
"""

import gc
import json
import os
import re
import sys
from pathlib import Path

# ── Kiểm tra thư viện bắt buộc ─────────────────────────────────────────────

def _check_imports() -> None:
    missing = []
    try:
        import fitz  # noqa: F401  (pymupdf)
    except ImportError:
        missing.append("pymupdf  →  pip install pymupdf")
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        missing.append("pytesseract  →  pip install pytesseract")
    try:
        import docx  # noqa: F401
    except ImportError:
        missing.append("python-docx  →  pip install python-docx")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow  →  pip install Pillow")
    if missing:
        print("⚠ Thiếu thư viện:\n  " + "\n  ".join(missing))
        sys.exit(1)


_check_imports()

import fitz  # PyMuPDF
import pytesseract
from docx import Document as DocxDocument
from PIL import Image

# ── Cấu hình Tesseract cho Windows ────────────────────────────────────────

def _setup_tesseract() -> None:
    """
    Tự động tìm Tesseract trên Windows.
    Ưu tiên theo thứ tự:
      1. Biến môi trường TESSERACT_PATH
      2. Các đường dẫn cài đặt mặc định phổ biến trên Windows
      3. Nếu không tìm thấy: in hướng dẫn và thoát
    """
    if sys.platform != "win32":
        return  # Linux/macOS: tesseract đã có trong PATH

    # 1. Biến môi trường do người dùng đặt
    env_path = os.environ.get("TESSERACT_PATH")
    if env_path and Path(env_path).is_file():
        pytesseract.pytesseract.tesseract_cmd = env_path
        print(f"  Tesseract: {env_path} (từ TESSERACT_PATH)")
        return

    # 2. Các đường dẫn mặc định phổ biến
    username = os.environ.get("USERNAME", "")
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        rf"C:\Users\{username}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"D:\Tessract-OCR\tesseract.exe",

    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            print(f"  Tesseract: {candidate}")
            return

    # 3. Không tìm thấy
    print("\n" + "=" * 60)
    print("Khong tim thay Tesseract tren may!")
    print()
    print("Huong dan cai dat:")
    print("  1. Tai installer tai:")
    print("     https://github.com/UB-Mannheim/tesseract/wiki")
    print("  2. Chon: tesseract-ocr-w64-setup-*.exe")
    print("  3. Trong installer: tick 'Additional language data'")
    print("     -> chon 'Vietnamese'")
    print("  4. Cai vao mac dinh:")
    # print(r"     C:\Program Files\Tesseract-OCR\")
    print()
    print("Hoac dat bien moi truong neu cai noi khac:")
    print("  set TESSERACT_PATH=C:\\duong\\dan\\tesseract.exe")
    print("=" * 60 + "\n")
    sys.exit(1)


def _check_tesseract_lang() -> bool:
    """Kiểm tra gói ngôn ngữ tiếng Việt đã cài chưa. Trả về True nếu có."""
    try:
        langs = pytesseract.get_languages(config="")
        if "vie" not in langs:
            print("\n[!] Chua cai goi ngon ngu tieng Viet cho Tesseract!")
            tess_cmd = pytesseract.pytesseract.tesseract_cmd
            tess_dir = Path(tess_cmd).parent / "tessdata"
            print(f"    Copy file vie.traineddata vao: {tess_dir}")
            print("    Tai file tai: https://github.com/tesseract-ocr/tessdata")
            print("    Script se fallback sang chi dung tieng Anh (eng).\n")
            return False
        return True
    except Exception:
        return False


_setup_tesseract()

# ── Cấu hình đường dẫn ─────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent.parent
LANDING_DIR = BASE_DIR / "data" / "landing"
OUTPUT_DIR  = BASE_DIR / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}

# ── Cấu hình OCR ───────────────────────────────────────────────────────────

# DPI render trang PDF thành ảnh để OCR
# 150 DPI: nhẹ RAM hơn (đủ dùng với văn bản in rõ)
# 200 DPI: cân bằng giữa chất lượng và RAM   ← mặc định
# 300 DPI: chất lượng cao nhất nhưng tốn RAM
OCR_DPI = 200

# Số ký tự tối thiểu trên một trang để coi là "có text" (không cần OCR)
MIN_TEXT_CHARS = 50

# Ngôn ngữ OCR: tự động chọn vie+eng hoặc fallback sang eng
_HAS_VIE_LANG = _check_tesseract_lang()
TESSERACT_LANG = "vie+eng" if _HAS_VIE_LANG else "eng"

# ── Tiền xử lý ảnh cho OCR ─────────────────────────────────────────────────

def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Tăng độ tương phản và chuyển grayscale để Tesseract nhận dạng tốt hơn.
    Không resize để tránh mất chi tiết chữ nhỏ.
    """
    img = img.convert("L")  # Grayscale
    return img


# ── Phát hiện PDF scan vs digital ─────────────────────────────────────────

def _is_scanned_pdf(doc: fitz.Document, sample_pages: int = 5) -> bool:
    """
    Kiểm tra xem PDF có phải scan hay không bằng cách thử extract text
    trên tối đa `sample_pages` trang đầu.

    Returns True nếu hầu hết các trang không có text (→ cần OCR).
    """
    total = min(sample_pages, len(doc))
    if total == 0:
        return True

    text_pages = 0
    for i in range(total):
        page = doc[i]
        text = page.get_text("text")
        if len(text.strip()) >= MIN_TEXT_CHARS:
            text_pages += 1
        # Giải phóng page
        page = None

    # Nếu < 30% số trang mẫu có text → coi là PDF scan
    return (text_pages / total) < 0.30


# ── Extract text từ PDF digital (không cần OCR) ────────────────────────────

def _extract_digital_pdf(doc: fitz.Document) -> str:
    """
    Extract text từ PDF có text sẵn bằng PyMuPDF.
    Xử lý từng trang, giải phóng bộ nhớ sau mỗi trang.
    """
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Chuẩn hoá khoảng trắng thừa
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            pages_text.append(f"<!-- Trang {page_num + 1} -->\n\n{text}")

        # Giải phóng page object khỏi RAM
        page = None
        if page_num % 10 == 0:
            gc.collect()

    return "\n\n---\n\n".join(pages_text)


# ── OCR từng trang PDF scan ────────────────────────────────────────────────

def _ocr_pdf_page_by_page(doc: fitz.Document, filepath: Path) -> str:
    """
    OCR toàn bộ trang của PDF scan.
    Mỗi trang:
        1. Render → Pixmap (ảnh)
        2. Chuyển sang PIL Image
        3. Tiền xử lý
        4. Chạy Tesseract
        5. Giải phóng pixmap + PIL image khỏi RAM ngay lập tức
    → Tránh std::bad_alloc do load nhiều trang cùng lúc.
    """
    pages_text: list[str] = []
    matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)  # scale từ 72dpi gốc

    total_pages = len(doc)

    for page_num in range(total_pages):
        # ── Bước 1: Render trang → Pixmap ─────────────────────────
        page = doc[page_num]
        try:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        except Exception as e:
            print(f"    ⚠ Không thể render trang {page_num + 1}: {e}")
            page = None
            continue

        # ── Bước 2: Pixmap → PIL Image (trong bộ nhớ) ─────────────
        try:
            img = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )
        except Exception as e:
            print(f"    ⚠ Không thể tạo PIL image trang {page_num + 1}: {e}")
            pixmap = None
            page = None
            gc.collect()
            continue
        finally:
            # Giải phóng pixmap ngay sau khi dùng
            pixmap = None

        # ── Bước 3: Tiền xử lý ảnh ────────────────────────────────
        img = _preprocess_image_for_ocr(img)

        # ── Bước 4: OCR bằng Tesseract ────────────────────────────
        try:
            ocr_text = pytesseract.image_to_string(
                img,
                lang=TESSERACT_LANG,
                config="--psm 3 --oem 1",  # psm 3: auto layout, oem 1: LSTM
            )
            ocr_text = ocr_text.strip()
            if ocr_text:
                pages_text.append(f"<!-- Trang {page_num + 1} -->\n\n{ocr_text}")
        except pytesseract.TesseractError as e:
            print(f"    ⚠ Tesseract lỗi trang {page_num + 1}: {e}")
        except Exception as e:
            print(f"    ⚠ OCR lỗi trang {page_num + 1}: {e}")
        finally:
            # Giải phóng PIL image
            img.close()
            img = None

        # ── Bước 5: Giải phóng page và gọi GC định kỳ ────────────
        page = None
        if (page_num + 1) % 5 == 0:
            gc.collect()
            print(f"    → Đã OCR {page_num + 1}/{total_pages} trang...")

    gc.collect()
    return "\n\n---\n\n".join(pages_text)


# ── Convert một file PDF ────────────────────────────────────────────────────

def _convert_pdf(filepath: Path) -> str:
    """
    Tự động phát hiện PDF digital hay scan, rồi chọn pipeline phù hợp.
    """
    doc = fitz.open(str(filepath))
    try:
        if _is_scanned_pdf(doc):
            print(f"    → PDF scan: dùng OCR ({len(doc)} trang)")
            return _ocr_pdf_page_by_page(doc, filepath)
        else:
            print(f"    → PDF digital: extract text ({len(doc)} trang)")
            return _extract_digital_pdf(doc)
    finally:
        doc.close()
        doc = None
        gc.collect()


# ── Convert DOCX ───────────────────────────────────────────────────────────

def _convert_docx(filepath: Path) -> str:
    """
    Extract text từ DOCX bằng python-docx.
    Giữ lại heading (dựa trên style name) và paragraph thường.
    """
    doc = DocxDocument(str(filepath))
    lines: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else ""

        # Map Heading → Markdown heading
        if style_name.startswith("Heading 1"):
            lines.append(f"# {text}")
        elif style_name.startswith("Heading 2"):
            lines.append(f"## {text}")
        elif style_name.startswith("Heading 3"):
            lines.append(f"### {text}")
        elif style_name.startswith("Heading"):
            lines.append(f"#### {text}")
        else:
            lines.append(text)

    return "\n\n".join(lines)


# ── Đường dẫn output ───────────────────────────────────────────────────────

def get_output_path(input_path: Path) -> Path:
    """
    Giữ nguyên cấu trúc thư mục từ data/landing sang data/standardized.

    Ví dụ:
        data/landing/legal/a.pdf  →  data/standardized/legal/a.md
    """
    relative_path = input_path.relative_to(LANDING_DIR)
    output_path = OUTPUT_DIR / relative_path
    return output_path.with_suffix(".md")


def save_markdown(output_path: Path, content: str) -> None:
    """Tạo thư mục cha và ghi file markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# ── Convert toàn bộ legal docs ─────────────────────────────────────────────

def convert_legal_docs() -> tuple[int, int]:
    """
    Convert PDF/DOCX/DOC trong data/landing/legal/ sang Markdown.

    Returns:
        success_count, failed_count
    """
    legal_dir = LANDING_DIR / "legal"

    if not legal_dir.exists():
        print(f"⚠ Không tìm thấy thư mục: {legal_dir}")
        return 0, 0

    success = 0
    failed = 0

    files = [
        f for f in legal_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in LEGAL_EXTENSIONS
    ]

    if not files:
        print("  Không có file nào để convert.")
        return 0, 0

    print(f"  Tìm thấy {len(files)} file cần convert.")

    for filepath in files:
        output_path = get_output_path(filepath)

        # Bỏ qua nếu đã convert rồi (resume-friendly)
        if output_path.exists():
            print(f"  ⏭ Đã tồn tại, bỏ qua: {filepath.name}")
            success += 1
            continue

        print(f"  Converting: {filepath.name}")

        try:
            ext = filepath.suffix.lower()

            if ext == ".pdf":
                markdown = _convert_pdf(filepath)
            elif ext in {".docx", ".doc"}:
                markdown = _convert_docx(filepath)
            else:
                print(f"    ⚠ Định dạng không hỗ trợ: {ext}")
                failed += 1
                continue

            if not markdown.strip():
                print(f"    ⚠ Kết quả rỗng: {filepath.name}")

            save_markdown(output_path, markdown)
            success += 1
            print(f"    ✓ Saved: {output_path.relative_to(BASE_DIR)}")

        except Exception as e:
            failed += 1
            print(f"    ✗ Failed: {filepath.name}")
            print(f"      Error: {type(e).__name__}: {e}")

        finally:
            # Đảm bảo GC sau mỗi file
            gc.collect()

    return success, failed


# ── Convert news articles ──────────────────────────────────────────────────

def convert_news_articles() -> tuple[int, int]:
    """
    Convert JSON crawled articles trong data/landing/news/ sang Markdown.

    JSON kỳ vọng có các field:
        - title
        - url
        - date_crawled
        - content_markdown

    Returns:
        success_count, failed_count
    """
    news_dir = LANDING_DIR / "news"

    if not news_dir.exists():
        print(f"⚠ Không tìm thấy thư mục: {news_dir}")
        return 0, 0

    success = 0
    failed = 0

    files = list(news_dir.rglob("*.json"))

    if not files:
        print("  Không có file JSON nào.")
        return 0, 0

    for filepath in files:
        output_path = get_output_path(filepath)

        if output_path.exists():
            print(f"  ⏭ Đã tồn tại, bỏ qua: {filepath.name}")
            success += 1
            continue

        print(f"  Converting news: {filepath.name}")

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title           = data.get("title", "Unknown")
            url             = data.get("url", "N/A")
            date_crawled    = data.get("date_crawled", "N/A")
            content_markdown = data.get("content_markdown", "")

            markdown = (
                f"# {title}\n\n"
                f"**Source:** {url}\n\n"
                f"**Crawled:** {date_crawled}\n\n"
                f"---\n\n"
                f"{content_markdown}"
            )

            save_markdown(output_path, markdown)
            success += 1
            print(f"    ✓ Saved: {output_path.relative_to(BASE_DIR)}")

        except Exception as e:
            failed += 1
            print(f"    ✗ Failed: {filepath.name}")
            print(f"      Error: {type(e).__name__}: {e}")

    return success, failed


# ── Entry point ────────────────────────────────────────────────────────────

def convert_all() -> None:
    """Convert toàn bộ legal docs và news articles."""
    print("=" * 60)
    print("Convert landing data to Markdown")
    print("=" * 60)

    if not LANDING_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {LANDING_DIR}")

    print("\n--- Legal Documents ---")
    legal_success, legal_failed = convert_legal_docs()

    print("\n--- News Articles ---")
    news_success, news_failed = convert_news_articles()

    total_success = legal_success + news_success
    total_failed  = legal_failed  + news_failed

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Legal  : {legal_success} OK, {legal_failed} lỗi")
    print(f"  News   : {news_success} OK, {news_failed} lỗi")
    print(f"  Tổng   : {total_success} OK, {total_failed} lỗi")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    convert_all()