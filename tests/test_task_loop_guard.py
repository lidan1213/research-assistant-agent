"""任务级 LoopGuard：跨阶段重复结果与后续路径封禁。"""
from app.agent.loop_guard import TaskLoopGuard


def test_repeated_result_blocks_tool_across_stages():
    guard = TaskLoopGuard()

    assert guard.record("knowledge_search", "没有找到 GraphRAG 相关资料") is True
    # 下一阶段换了参数，但工具返回相同内容：识别为没有新进展。
    assert guard.record("knowledge_search", "没有找到 GraphRAG 相关资料") is False

    blocked, reason = guard.is_blocked("knowledge_search")
    assert blocked is True
    assert "跨步骤" in reason


def test_different_result_is_progress():
    guard = TaskLoopGuard()
    assert guard.record("knowledge_search", "结果 A") is True
    assert guard.record("knowledge_search", "结果 B") is True
    assert guard.is_blocked("knowledge_search")[0] is False


def test_block_is_scoped_to_one_tool():
    guard = TaskLoopGuard()
    guard.record("knowledge_search", "无结果")
    guard.record("knowledge_search", "无结果")
    assert guard.is_blocked("knowledge_search")[0] is True
    assert guard.is_blocked("arxiv_search")[0] is False
