from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from careerpilot.api import app
from careerpilot.resume_parser import ResumeParseError, parse_resume_upload


client = TestClient(app)


def _make_text_pdf(text: str) -> bytes:
    """Create a tiny searchable PDF without introducing a test dependency."""

    buffer = BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )

    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(
        f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)

    writer.write(buffer)
    return buffer.getvalue()


def test_parse_searchable_pdf_resume():
    parsed = parse_resume_upload(
        filename="resume.pdf",
        content_type="application/pdf",
        data=_make_text_pdf("Python LangGraph Agent FastAPI project experience"),
    )

    assert "Python LangGraph" in parsed.text
    assert parsed.file_type == "PDF"
    assert parsed.page_count == 1
    assert parsed.char_count > 20


def test_parse_utf8_text_resume():
    parsed = parse_resume_upload(
        filename="resume.txt",
        content_type="text/plain",
        data="Python、LangGraph、RAG 项目经验".encode("utf-8"),
    )

    assert "LangGraph" in parsed.text
    assert parsed.file_type == "TXT"


def test_rejects_unsupported_resume_extension():
    try:
        parse_resume_upload(
            filename="resume.docx",
            content_type="application/octet-stream",
            data=b"not a supported resume",
        )
    except ResumeParseError as exc:
        assert "仅支持" in str(exc)
    else:
        raise AssertionError("Unsupported files must be rejected")


def test_resume_extract_endpoint_returns_text_and_metadata():
    response = client.post(
        "/resume/extract",
        files={
            "resume_file": (
                "candidate.pdf",
                _make_text_pdf("Python LangGraph Agent FastAPI resume content"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "candidate.pdf"
    assert payload["file_type"] == "PDF"
    assert payload["page_count"] == 1
    assert "LangGraph" in payload["text"]


def test_web_analyze_accepts_uploaded_resume_without_pasted_text():
    response = client.post(
        "/web/analyze",
        data={
            "resume_text": "",
            "jd_text": "Python LangGraph RAG Agent FastAPI Docker",
            "target_role": "大模型/Agent 工程实习生",
            "offline": "on",
            "provider": "deepseek",
            "model": "",
        },
        files={
            "resume_file": (
                "candidate.pdf",
                _make_text_pdf("Python LangGraph Agent FastAPI project experience"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert "已解析 candidate.pdf" in response.text
    assert "LangGraph 执行轨迹" in response.text
    assert "Python LangGraph Agent FastAPI" in response.text


def test_web_analyze_requires_file_or_resume_text():
    response = client.post(
        "/web/analyze",
        data={
            "resume_text": "",
            "jd_text": "Python LangGraph RAG Agent FastAPI Docker",
            "target_role": "大模型/Agent 工程实习生",
            "offline": "on",
            "provider": "deepseek",
            "model": "",
        },
    )

    assert response.status_code == 400
    assert "请上传简历文件" in response.text
