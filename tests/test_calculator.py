from app.agent.tools.calculator import CalculatorTool


def run(expr):
    return CalculatorTool().run({"expression": expr})


def test_basic_arithmetic():
    assert run("1 + 2 * 3").data["result"] == 7
    assert run("(120 - 80) / 80 * 100").data["result"] == 50.0
    assert run("2 ** 10").data["result"] == 1024
    assert run("7 // 2").data["result"] == 3
    assert run("7 % 3").data["result"] == 1
    assert run("-5 + +2").data["result"] == -3


def test_result_content_readable():
    r = run("1+1")
    assert r.ok and "2" in r.content


def test_division_by_zero():
    r = run("1 / 0")
    assert not r.ok and r.error == "division by zero"


def test_rejects_non_arithmetic_nodes():
    for bad in ["__import__('os')", "abs(1)", "a + 1", "[1,2][0]", "(1).real", "'x' * 3"]:
        r = run(bad)
        assert not r.ok
        assert r.error == "unsupported expression"


def test_empty_expression():
    r = run("")
    assert not r.ok and r.error == "empty expression"
