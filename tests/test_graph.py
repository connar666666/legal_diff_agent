"""图构建冒烟测试（不调用远程 Ollama）。"""

import pytest


def test_import_builder():
    from app.graph.builder import build_agent_graph

    try:
        g = build_agent_graph()
        assert g is not None
    except Exception as e:
        if "Connection" in str(e) or "ollama" in str(e).lower():
            pytest.skip("Ollama 不可用")
        raise
