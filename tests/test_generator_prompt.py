from app.generator.generator import compose_user_prompt

MEMORY_HEADER = "About the user"


def test_includes_query_and_context():
    up = compose_user_prompt("What is FAISS?", "[Chunk 1] FAISS is a library", "")
    assert "What is FAISS?" in up
    assert "FAISS is a library" in up


def test_memory_block_injected_when_present():
    up = compose_user_prompt("hi", "ctx", "- likes green tea\n- lives in Singapore")
    assert MEMORY_HEADER in up
    assert "likes green tea" in up


def test_no_memory_section_when_empty_backward_compatible():
    up = compose_user_prompt("hi", "ctx", "")
    assert MEMORY_HEADER not in up
