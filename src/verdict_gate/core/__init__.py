"""Analysis core: sees only task-level outcome tables plus metadata.

No LLM, HTTP or benchmark code here, and nothing from verdict_gate.adapters or the CLI (tests/test_core_isolation.py
enforces this). Every threshold comes from a policy or validity file.
"""
