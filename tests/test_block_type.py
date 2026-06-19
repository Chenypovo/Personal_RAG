from app.chunker.block_type import assign_block_types, classify_block_type


def test_pure_heading():
    assert classify_block_type("# Only A Heading") == "heading"


def test_paragraph_with_leading_heading():
    assert classify_block_type("# Title\n\n## Sec\n\nsome ordinary paragraph text.") == "paragraph"


def test_list():
    assert classify_block_type("- a\n- b\n- c") == "list"
    assert classify_block_type("1. one\n2. two") == "list"


def test_table():
    assert classify_block_type("| a | b |\n| --- | --- |\n| 1 | 2 |") == "table"


def test_code_fence():
    assert classify_block_type("```python\nprint(1)\nx = 2\n```") == "code"


def test_plain_paragraph():
    assert classify_block_type("Just a normal sentence without markdown.") == "paragraph"


def test_empty_is_paragraph():
    assert classify_block_type("") == "paragraph"


def test_assign_block_types_sets_field():
    chunks = [{"text": "- a\n- b"}, {"text": "plain text"}]
    assign_block_types(chunks)
    assert chunks[0]["block_type"] == "list"
    assert chunks[1]["block_type"] == "paragraph"
