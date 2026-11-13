import verdict_gate


def test_version_is_a_string():
    assert isinstance(verdict_gate.__version__, str) and verdict_gate.__version__


def test_public_api():
    assert callable(verdict_gate.next_batch)
