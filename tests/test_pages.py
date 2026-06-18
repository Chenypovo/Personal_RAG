from app.loader.pages import assign_pages, page_for_offset

# page 1 starts at char 0, page 2 at 100, page 3 at 250
OFFS = [(0, 1), (100, 2), (250, 3)]


def test_page_for_offset_within_pages():
    assert page_for_offset(0, OFFS) == 1
    assert page_for_offset(50, OFFS) == 1
    assert page_for_offset(100, OFFS) == 2
    assert page_for_offset(249, OFFS) == 2
    assert page_for_offset(300, OFFS) == 3


def test_page_for_offset_before_first_clamps_to_first():
    assert page_for_offset(-5, OFFS) == 1


def test_page_for_offset_empty_is_none():
    assert page_for_offset(10, []) is None


def test_assign_pages_sets_start_and_end_page():
    chunks = [
        {"chunk_id": 0, "start": 10, "end": 80},
        {"chunk_id": 1, "start": 120, "end": 300},  # spans page 2 -> 3
    ]
    assign_pages(chunks, OFFS)
    assert chunks[0]["page_start"] == 1 and chunks[0]["page_end"] == 1
    assert chunks[1]["page_start"] == 2 and chunks[1]["page_end"] == 3


def test_assign_pages_noop_without_offsets():
    chunks = [{"chunk_id": 0, "start": 0, "end": 5}]
    assign_pages(chunks, [])
    assert "page_start" not in chunks[0]
