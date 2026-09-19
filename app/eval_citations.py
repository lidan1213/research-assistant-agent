"""答案引用覆盖率评估：用检索来源和答案中的 [n] 标记衡量证据覆盖。

这是轻量可解释指标，不替代人工或 LLM judge：
- citation_count：答案中出现的引用编号数量
- valid_citation_count：落在实际来源范围内的引用数量
- citation_coverage：有效引用数 / 答案中可引用的关键句数（保守估计）
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(text or "")}


def citation_coverage(answer: str, source_count: int) -> dict[str, float | int]:
    text = answer or ""
    refs = []
    for group in re.findall(r"\[(\d+(?:[-,]\d+)*)\]", text):
        for part in re.split(r"[-,]", group):
            if part.isdigit():
                refs.append(int(part))
    unique_refs = sorted(set(refs))
    valid = [n for n in unique_refs if 1 <= n <= source_count]
    # 排除标题/空行，按句估算需要证据的事实句；无句子时避免除零。
    sentences = [s.strip() for s in re.split(r"[。！？.!?\n]+", text) if s.strip()]
    factual = [s for s in sentences if len(s) >= 8 and not s.startswith(("#", "```"))]
    denominator = max(len(factual), 1)
    return {
        "citation_count": len(unique_refs),
        "valid_citation_count": len(valid),
        "source_count": source_count,
        "factual_sentence_count": len(factual),
        "citation_coverage": round(min(len(valid) / denominator, 1.0), 4),
        "validity": round(len(valid) / len(unique_refs), 4) if unique_refs else 0.0,
    }


def citation_support(answer: str, sources: list[str], threshold: float = 0.35) -> dict:
    """Check whether each cited sentence has lexical support in cited sources.

    This is an auditable CI-safe guard, not semantic entailment. It is stricter
    than merely checking that a citation number is in range.
    """
    details = []
    for sentence in [s.strip() for s in re.split(r"[。！？.!?\n]+", answer or "") if s.strip()]:
        groups = re.findall(r"\[(\d+(?:[-,]\d+)*)\]", sentence)
        refs = sorted({int(n) for group in groups for n in re.split(r"[-,]", group) if n.isdigit()})
        if not refs:
            continue
        claim = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", sentence)
        claim_tokens = _tokens(claim)
        valid_refs = [n for n in refs if 1 <= n <= len(sources)]
        cited_tokens = _tokens(" ".join(sources[n - 1] for n in valid_refs))
        overlap = len(claim_tokens & cited_tokens) / len(claim_tokens) if claim_tokens else 0.0
        details.append({"claim": claim, "refs": refs, "valid_refs": valid_refs,
                        "support": round(overlap, 4), "supported": bool(valid_refs) and overlap >= threshold})
    supported = sum(1 for row in details if row["supported"])
    return {"score": round(supported / len(details), 4) if details else 0.0,
            "supported_claims": supported, "cited_claims": len(details),
            "threshold": threshold, "claims": details}
