from app.loader.headings import assign_headings, extract_headings, heading_path_for_offset

TEXT = "# Title\n\nintro\n\n## Section A\n\naaa\n\n### Sub A1\n\nbbb\n\n## Section B\n\nccc"


def test_extract_headings_titles_and_levels():
    hs = extract_headings(TEXT)
    assert [t for _, _, t in hs] == ["Title", "Section A", "Sub A1", "Section B"]
    assert [lvl for _, lvl, _ in hs] == [1, 2, 3, 2]


def test_heading_path_nested():
    hs = extract_headings(TEXT)
    off = TEXT.index("bbb")
    assert heading_path_for_offset(off, hs) == ["Title", "Section A", "Sub A1"]


def test_heading_path_pops_to_sibling():
    hs = extract_headings(TEXT)
    off = TEXT.index("ccc")
    assert heading_path_for_offset(off, hs) == ["Title", "Section B"]


def test_heading_path_before_any_heading_is_empty():
    assert heading_path_for_offset(0, [(5, 1, "X")]) == []


def test_assign_headings_sets_path_and_string():
    chunks = [{"start": TEXT.index("bbb"), "text": "bbb"}]
    assign_headings(chunks, TEXT)
    assert chunks[0]["heading_path"] == ["Title", "Section A", "Sub A1"]
    assert chunks[0]["heading"] == "Title > Section A > Sub A1"


def test_assign_headings_noop_without_headings():
    chunks = [{"start": 0, "text": "plain"}]
    assign_headings(chunks, "no headings at all")
    assert "heading" not in chunks[0]
