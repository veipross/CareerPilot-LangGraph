"""Safe resume-file parsing helpers used by the web and upload APIs."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Optional

from pypdf import PdfReader


MAX_RESUME_FILE_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 60
MIN_EXTRACTED_CHARS = 20

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
    "",
}


class ResumeParseError(ValueError):
    """Raised when an uploaded resume cannot be parsed safely."""


@dataclass(frozen=True)
class ResumeParseResult:
    """Normalized text and metadata extracted from an uploaded resume."""

    text: str
    filename: str
    file_type: str
    page_count: int
    char_count: int


def _normalize_text(text: str) -> str:
    """Normalize PDF/text extraction artifacts without rewriting content."""

    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def _validate_upload(filename: str, content_type: Optional[str], data: bytes) -> tuple[str, str]:
    safe_name = Path(filename or "resume").name
    suffix = Path(safe_name).suffix.lower()
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()

    if suffix not in _ALLOWED_EXTENSIONS:
        raise ResumeParseError("仅支持 PDF、TXT 或 Markdown 简历文件。")

    if normalized_type not in _ALLOWED_CONTENT_TYPES:
        raise ResumeParseError(f"不支持的文件类型：{normalized_type or 'unknown'}。")

    if not data:
        raise ResumeParseError("上传的简历文件为空。")

    if len(data) > MAX_RESUME_FILE_BYTES:
        max_mb = MAX_RESUME_FILE_BYTES // (1024 * 1024)
        raise ResumeParseError(f"简历文件不能超过 {max_mb} MB。")

    return safe_name, suffix


def _extract_pdf(data: bytes) -> tuple[str, int]:
    if not data.startswith(b"%PDF-"):
        raise ResumeParseError("文件扩展名是 PDF，但内容不是有效的 PDF 文件。")

    try:
        reader = PdfReader(BytesIO(data), strict=False)
    except Exception as exc:
        raise ResumeParseError(f"PDF 读取失败：{exc}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ResumeParseError("PDF 已加密，请上传未加密版本。") from exc
        if not unlocked:
            raise ResumeParseError("PDF 已加密，请上传未加密版本。")

    page_count = len(reader.pages)
    if page_count == 0:
        raise ResumeParseError("PDF 中没有可读取的页面。")
    if page_count > MAX_PDF_PAGES:
        raise ResumeParseError(f"PDF 页数不能超过 {MAX_PDF_PAGES} 页。")

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as exc:
            raise ResumeParseError(f"PDF 第 {page_number} 页文本提取失败。") from exc

    return _normalize_text("\n\n".join(page_texts)), page_count


def _extract_plain_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return _normalize_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ResumeParseError("文本文件编码无法识别，请使用 UTF-8 编码。")


def parse_resume_upload(
    *,
    filename: str,
    content_type: Optional[str],
    data: bytes,
) -> ResumeParseResult:
    """Parse a PDF/TXT/Markdown resume uploaded by the user.

    The file is processed entirely in memory and is never persisted to disk.
    """

    safe_name, suffix = _validate_upload(filename, content_type, data)

    if suffix == ".pdf":
        text, page_count = _extract_pdf(data)
        file_type = "PDF"
    else:
        text = _extract_plain_text(data)
        page_count = 1
        file_type = "Markdown" if suffix in {".md", ".markdown"} else "TXT"

    if len(text) < MIN_EXTRACTED_CHARS:
        if suffix == ".pdf":
            raise ResumeParseError(
                "PDF 未提取到足够文本，可能是扫描件或图片版简历。"
                "请上传可复制文字的 PDF，或把简历内容粘贴到文本框。"
            )
        raise ResumeParseError("文件中的可用文本过少，请检查内容后重试。")

    return ResumeParseResult(
        text=text,
        filename=safe_name,
        file_type=file_type,
        page_count=page_count,
        char_count=len(text),
    )
