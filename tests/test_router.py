from app.agent.router import Router


def const(canned: str):
    def _complete(system_prompt: str, user_prompt: str) -> str:
        return canned
    return _complete


def test_parses_decision():
    r = Router(complete_fn=const('{"use_docs": true, "use_memory": false, "write_memory": true, "rewritten_query": "what is faiss"}'))
    d = r.route("what's faiss again?")
    assert d.use_docs is True
    assert d.use_memory is False
    assert d.write_memory is True
    assert d.rewritten_query == "what is faiss"


def test_fallback_on_garbage_is_all_on():
    r = Router(complete_fn=const("not json"))
    d = r.route("tell me about my project")
    assert d.use_docs and d.use_memory and d.write_memory
    assert d.rewritten_query == "tell me about my project"


def test_missing_rewritten_query_defaults_to_original():
    r = Router(complete_fn=const('{"use_docs": true, "use_memory": true, "write_memory": false}'))
    d = r.route("original question")
    assert d.rewritten_query == "original question"
    assert d.write_memory is False


def test_empty_message_no_llm_call_defaults_all_off_writes_off():
    called = {"n": 0}

    def fake(s, u):
        called["n"] += 1
        return "{}"

    r = Router(complete_fn=fake)
    d = r.route("   ")
    assert called["n"] == 0
    assert d.write_memory is False
