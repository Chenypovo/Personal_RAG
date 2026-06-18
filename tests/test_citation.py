from app.generator.generator import OpenAICompatibleGenerator as G


def test_citation_includes_single_page():
    label = G._citation_label({"chunk_id": 1, "source": "/x/a.pdf", "page_start": 3, "page_end": 3})
    assert label == "Chunk 1, a.pdf, p.3"


def test_citation_includes_page_span():
    label = G._citation_label({"chunk_id": 2, "source": "/x/a.pdf", "page_start": 3, "page_end": 5})
    assert label == "Chunk 2, a.pdf, p.3-5"


def test_citation_without_page_unchanged():
    assert G._citation_label({"chunk_id": 1, "source": "/x/a.md"}) == "Chunk 1, a.md"
