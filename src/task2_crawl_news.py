"""
Task 2 - Crawl news articles about Vietnamese artists related to drugs.

Install before running:
    pip install crawl4ai
    playwright install chromium

Run:
    python src/task2_crawl_news.py

Output:
    data/landing/news/article_01.json
    data/landing/news/article_02.json
    ...
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://viettimes.vn/ca-si-chi-dan-nguoi-mau-an-tay-bi-bat-post180088.html",
    "https://danviet.vn/clip-an-tay-bat-khoc-vi-ma-tuy-huy-hoai-su-nghiep-tuong-lai-chi-dan-khuyen-gioi-tre-bo-y-dinh-dung-thu-20241115155132479.htm",
    "https://vietnamnet.vn/huu-tin-va-nhung-sao-viet-noi-tieng-ten-tin-deu-dinh-vao-ma-tuy-2029495.html",
    "https://vietnamnet.vn/tu-vu-huu-tin-bi-bat-vi-dung-ma-tuy-nghe-si-can-trong-khi-phat-ngon-2030279.html",
    "https://vnexpress.net/hiep-ga-va-cuoc-song-khon-kho-vi-ma-tuy-1892564.html",
]


def setup_directory() -> None:
    """Create the output directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str, fallback: str) -> str:
    """Create a readable, filesystem-safe filename suffix."""
    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:80] or fallback


def get_result_value(result: Any, key: str, default: Any = None) -> Any:
    """Read Crawl4AI result values across dict-like and object-like versions."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def extract_title(result: Any, url: str) -> str:
    """Prefer crawler metadata title, then first markdown heading, then URL."""
    metadata = get_result_value(result, "metadata", {}) or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    if title:
        return str(title).strip()

    markdown = get_result_value(result, "markdown", "") or ""
    if not isinstance(markdown, str):
        markdown = str(markdown)

    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()

    return url


async def crawl_article(crawler: Any, url: str) -> dict[str, Any]:
    """
    Crawl one article and return normalized metadata plus content.

    Required metadata:
        - URL goc: url
        - Ngay crawl: date_crawled
        - Tieu de bai bao: title
    """
    result = await crawler.arun(url=url)

    success = bool(get_result_value(result, "success", True))
    error_message = get_result_value(result, "error_message", None)
    markdown = get_result_value(result, "markdown", "") or ""
    html = get_result_value(result, "html", "") or ""

    return {
        "url": url,
        "source_url": url,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "title": extract_title(result, url),
        "success": success,
        "error_message": error_message,
        "content_markdown": markdown if isinstance(markdown, str) else str(markdown),
        "content_html": html if isinstance(html, str) else str(html),
    }


def save_article(article: dict[str, Any], index: int) -> Path:
    """Save each article as one JSON file."""
    title_slug = slugify(article["title"], fallback=f"article-{index:02d}")
    filepath = DATA_DIR / f"article_{index:02d}_{title_slug}.json"
    filepath.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filepath


async def crawl_all() -> None:
    """Crawl all configured article URLs and save one JSON file per article."""
    from crawl4ai import AsyncWebCrawler

    setup_directory()

    async with AsyncWebCrawler() as crawler:
        for index, url in enumerate(ARTICLE_URLS, start=1):
            print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
            try:
                article = await crawl_article(crawler, url)
            except Exception as exc:
                article = {
                    "url": url,
                    "source_url": url,
                    "date_crawled": datetime.now(timezone.utc).isoformat(),
                    "title": url,
                    "success": False,
                    "error_message": str(exc),
                    "content_markdown": "",
                    "content_html": "",
                }

            filepath = save_article(article, index)
            print(f"Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
