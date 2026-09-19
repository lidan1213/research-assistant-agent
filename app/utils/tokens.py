"""Token 估算与截断工具（轻量、无依赖）。

说明：
- 仅用于护栏（上下文预算、observation 截断）的近似估算，并非真实 tokenizer。
- 中文按 ~1.6 字符/token、英文/数字按 ~4 字符/token 的混合启发式。
"""
from __future__ import annotations

_CJK_LO = ord("\u4e00")
_CJK_HI = ord("\u9fff")


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（中文 1.6 字符/token，其余 4 字符/token）。"""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if _CJK_LO <= o <= _CJK_HI:
            cjk += 1
        else:
            other += 1
    return int(cjk / 1.6 + other / 4) + 1


def truncate_tokens(text: str, max_tokens: int, tail: bool = True) -> str:
    """按 token 预算截断文本；tail=True 保留头部（默认），否则保留尾部。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    # 近似字符预算：取更保守的系数，避免溢出
    budget = max(1, int(max_tokens * 1.4))
    if tail:
        return text[:budget] + f"…(已截断，原 {estimate_tokens(text)} tokens)"
    return "…" + text[-budget:]
