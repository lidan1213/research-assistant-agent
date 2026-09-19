"""内容审计：敏感词监控 + 越权尝试记录。

审计日志写入 TraceLedger（kind="audit"），管理端 /api/admin/audit 读取。
敏感词只记录不拦截（教育平台语境下"色情/毒品"等词可能出现在学术讨论中，
拦截会造成误伤；记录供管理员复核）。
"""
from __future__ import annotations

import json

from app.llm.trace import get_trace_ledger

# 默认敏感词库（管理员可在此扩展；命中即记录）
SENSITIVE_WORDS = [
    "自杀", "自残", "轻生", "赌博", "色情", "淫秽", "毒品", "吸毒", "冰毒",
    "枪支", "枪杀", "爆炸物", "作弊", "代考", "替考", "论文代写", "抄袭答案",
    "黑客入侵", "攻击服务器", "破解系统", "售卖答案", "诈骗", "洗钱",
]

# 越权访问类型 -> 记录标签
UNAUTHORIZED_KINDS = {
    "session_access": "越权访问他人会话",
    "search_cross_user": "越权跨用户检索",
    "kb_cross_user": "越权访问他人知识库",
}


def check_sensitive(text: str) -> list[str]:
    """返回文本命中的敏感词列表（未命中返回空列表）。"""
    if not text:
        return []
    return [w for w in SENSITIVE_WORDS if w in text]


def record_audit(username: str, kind: str, detail: dict) -> None:
    """写一条审计日志。kind: sensitive / unauthorized / other。"""
    try:
        get_trace_ledger().record(
            "audit",
            f"{kind}:{username}",
            success=True,
            duration_ms=0,
            detail=json.dumps(detail, ensure_ascii=False)[:2000],
        )
    except Exception:  # noqa: BLE001
        pass  # 审计失败不影响主流程


def record_sensitive_hit(username: str, word: str, message: str, session_id: str | None = None) -> None:
    record_audit(
        username,
        "sensitive",
        {"word": word, "message": message[:150], "session": session_id or ""},
    )


def record_unauthorized(username: str, kind: str, target: str) -> None:
    record_audit(
        username,
        "unauthorized",
        {"kind": UNAUTHORIZED_KINDS.get(kind, kind), "target": target[:100]},
    )
