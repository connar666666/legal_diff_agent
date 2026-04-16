from app.utils.download_filename import stable_filename_for_download


def test_pdf_from_url_and_content_type():
    u = "https://example.gov.cn/attachment/7385197.pdf"
    name = stable_filename_for_download(u, "application/pdf")
    assert name.endswith(".pdf")
    assert "7385197" in name


def test_pdf_from_content_type_only():
    u = "https://example.gov.cn/dl?id=1"
    name = stable_filename_for_download(u, "application/pdf")
    assert name.endswith(".pdf")
