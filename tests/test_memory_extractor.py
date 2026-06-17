from app.memory.extractor import MemoryExtractor


def make_extractor(canned: str):
    calls = {}

    def fake_complete(system_prompt: str, user_prompt: str) -> str:
        calls["system"] = system_prompt
        calls["user"] = user_prompt
        return canned

    return MemoryExtractor(complete_fn=fake_complete), calls


def test_parses_facts_object_format():
    ext, _ = make_extractor('{"facts": [{"fact_object": "guitar", "fact_content": "started learning guitar", "visibility": "PUBLIC"}]}')
    facts = ext.extract("I started learning guitar", source="chat")
    assert len(facts) == 1
    assert facts[0].fact_object == "guitar"
    assert facts[0].fact_content == "started learning guitar"
    assert facts[0].visibility == "PUBLIC"


def test_parses_bare_array():
    ext, _ = make_extractor('[{"fact_content": "lives in Singapore"}]')
    facts = ext.extract("I live in Singapore")
    assert len(facts) == 1
    assert facts[0].fact_content == "lives in Singapore"


def test_parses_nested_facts():
    ext, _ = make_extractor('[{"facts": [{"fact_content": "likes tea"}]}]')
    facts = ext.extract("I like tea")
    assert [f.fact_content for f in facts] == ["likes tea"]


def test_strips_code_fences():
    ext, _ = make_extractor('```json\n{"facts": [{"fact_content": "plays chess"}]}\n```')
    facts = ext.extract("I play chess")
    assert [f.fact_content for f in facts] == ["plays chess"]


def test_empty_input_skips_llm_call():
    called = {"n": 0}

    def fake_complete(s, u):
        called["n"] += 1
        return "[]"

    ext = MemoryExtractor(complete_fn=fake_complete)
    assert ext.extract("   ") == []
    assert called["n"] == 0


def test_parse_failure_returns_empty():
    ext, _ = make_extractor("totally not json")
    assert ext.extract("something") == []


def test_normalizes_numeric_visibility_and_defaults_public():
    ext, _ = make_extractor('{"facts": [{"fact_content": "has diabetes", "visibility": 2}, {"fact_content": "likes hiking"}]}')
    facts = ext.extract("...")
    assert facts[0].visibility == "PRIVATE"   # 2 -> PRIVATE
    assert facts[1].visibility == "PUBLIC"    # missing -> PUBLIC


def test_drops_facts_without_content():
    ext, _ = make_extractor('{"facts": [{"fact_object": "x", "fact_content": ""}, {"fact_content": "valid one"}]}')
    facts = ext.extract("...")
    assert [f.fact_content for f in facts] == ["valid one"]


def test_passes_input_text_to_prompt():
    ext, calls = make_extractor("[]")
    ext.extract("I moved to Tokyo last week", message_time="2026-06-17")
    assert "I moved to Tokyo last week" in calls["user"]
    assert "2026-06-17" in calls["user"]
