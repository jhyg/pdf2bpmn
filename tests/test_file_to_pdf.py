import sys
from pathlib import Path

import pytest

# Add project root to path (tests are sometimes run without installing package)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf2bpmn.config import Config
from pdf2bpmn.converters.file_to_pdf import convert_to_pdf, FileToPdfError  # noqa: E402


def _assert_pdf_header(path: str) -> None:
    p = Path(path)
    assert p.exists(), f"PDF가 생성되지 않았습니다: {p}"
    with p.open("rb") as f:
        head = f.read(5)
    assert head == b"%PDF-", f"PDF 헤더가 아닙니다: {p} head={head!r}"


def test_pdf_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Config, "ENABLE_FILE_CONVERSION", True)

    pdf = tmp_path / "already.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%....\n")

    out = convert_to_pdf(str(pdf), str(tmp_path))
    assert Path(out).resolve() == pdf.resolve()


def test_image_to_pdf_creates_valid_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Config, "ENABLE_FILE_CONVERSION", True)

    try:
        from PIL import Image
    except Exception as e:
        pytest.skip(f"Pillow가 없어 이미지 변환 테스트를 스킵합니다: {e}")

    img_path = tmp_path / "hello.png"
    img = Image.new("RGB", (200, 80), color=(255, 0, 0))
    img.save(img_path)

    out_pdf = convert_to_pdf(str(img_path), str(tmp_path))
    _assert_pdf_header(out_pdf)


def test_office_rtf_to_pdf_if_soffice_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Config, "ENABLE_FILE_CONVERSION", True)
    # Let converter find soffice from PATH unless user configured explicitly

    rtf = tmp_path / "sample.rtf"
    rtf.write_text(r"{\rtf1\ansi\deff0{\fonttbl{\f0 Arial;}}\f0\fs24 Hello RTF\par}", encoding="utf-8")

    try:
        out_pdf = convert_to_pdf(str(rtf), str(tmp_path))
    except FileToPdfError as e:
        # Most common reason in local dev: LibreOffice not installed
        if "LibreOffice" in str(e) or "soffice" in str(e):
            pytest.skip(f"LibreOffice(soffice)가 없어 Office 변환 테스트를 스킵합니다: {e}")
        raise

    _assert_pdf_header(out_pdf)


def test_same_stem_does_not_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    동일 stem(예: same.png / same.jpg)을 연속 변환해도
    출력 PDF가 덮어써지지 않고 각각 생성되어야 합니다.
    """
    monkeypatch.setattr(Config, "ENABLE_FILE_CONVERSION", True)

    try:
        from PIL import Image
    except Exception as e:
        pytest.skip(f"Pillow가 없어 이미지 변환 테스트를 스킵합니다: {e}")

    png_path = tmp_path / "same.png"
    jpg_path = tmp_path / "same.jpg"

    Image.new("RGB", (100, 40), color=(0, 255, 0)).save(png_path)
    Image.new("RGB", (100, 40), color=(0, 0, 255)).save(jpg_path)

    out1 = convert_to_pdf(str(png_path), str(tmp_path))
    out2 = convert_to_pdf(str(jpg_path), str(tmp_path))

    assert out1 != out2, "서로 다른 입력이 같은 출력 PDF 경로를 반환했습니다(덮어쓰기 위험)"
    _assert_pdf_header(out1)
    _assert_pdf_header(out2)

