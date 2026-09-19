"""Conservative and auditable cross-document evidence conflict analysis."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|％|mg|kg|g|mm|cm|m|℃|°c|年|天|小时)?", re.I)
_NEG_RE = re.compile(r"(?:不|未|无|不能|不会|并非|否认|not|no|never|without|failed?\s+to)", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_SAMPLE_RE = re.compile(r"(?:样本(?:量|数)?|sample(?:\s+size)?|n)\s*(?:为|是|=|:)?\s*(\d{2,})", re.I)
# 版本必须带 v/version 前缀，避免把 25.7% 等指标值误识别成版本。
_VERSION_RE = re.compile(r"\b(?:v|version\s+)(\d+(?:\.\d+){1,2})\b", re.I)
_DATASET_PATTERNS = (
    re.compile(r"(?:基于|使用|采用|在)\s*([A-Za-z][A-Za-z0-9_. -]{1,30})\s*(?:数据集|dataset)", re.I),
    re.compile(r"([\u4e00-\u9fffA-Za-z0-9_.-]{2,30})数据集", re.I),
)
_METRIC_TERMS = ("准确率", "精度", "召回率", "效率", "稳定性", "延迟", "吞吐量", "损失",
                 "accuracy", "precision", "recall", "f1", "mrr", "ndcg", "latency", "loss")
_OPPOSITE_PAIRS = (
    (("无需", "无须", "不需要", "不用", "避免"), ("必须", "需要", "仍要", "仍然必须")),
    (("可以", "能够", "可直接"), ("不能", "无法", "不可")),
    (("提高", "增加", "改善"), ("降低", "减少", "恶化")),
    (("支持", "证实", "成立"), ("反驳", "否认", "不成立")),
)
_STOP = {"研究", "结果", "表明", "显示", "认为", "the", "and", "that", "with", "from"}


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.lower()
        if token in _STOP:
            continue
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            tokens.update(token[i:i + 2] for i in range(len(token) - 1))
    return tokens


def _source(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    return str(item.get("source") or meta.get("source") or meta.get("doc_id") or item.get("doc_id") or "?")


def _first_value(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_dataset(text: str, meta: dict[str, Any]) -> str | None:
    explicit = _first_value(meta, "dataset", "data_set", "benchmark")
    if explicit:
        return str(explicit).strip()
    for pattern in _DATASET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" ，,。.;；")
    return None


def _extract_metric(text: str, query: str) -> str | None:
    lower_text = text.lower()
    in_text = next((metric for metric in _METRIC_TERMS if metric in lower_text), None)
    if in_text:
        return in_text
    lower_query = query.lower()
    return next((metric for metric in _METRIC_TERMS if metric in lower_query), None)


def _extract_values(text: str) -> list[dict[str, Any]]:
    values = []
    for match in _NUMBER_RE.finditer(text):
        raw, unit = match.groups()
        value = float(raw)
        normalized = (unit or "number").lower().replace("％", "%")
        if normalized == "number" and value.is_integer() and value <= 10:
            continue
        values.append({"value": value, "unit": normalized})
    return values


def _stance(text: str) -> str:
    if _NEG_RE.search(text):
        return "negative"
    negative_terms = {term for _, side in _OPPOSITE_PAIRS for term in side}
    positive_terms = {term for side, _ in _OPPOSITE_PAIRS for term in side}
    if any(term in text for term in negative_terms):
        return "negative"
    if any(term in text for term in positive_terms):
        return "positive"
    return "neutral"


def _quality(item: dict[str, Any], conditions: dict[str, Any]) -> dict[str, Any]:
    """Score evidence only from auditable metadata, never retrieval similarity."""
    meta = item.get("metadata") or {}
    score, reasons = 0.50, ["基础分 0.50"]
    source_type = str(_first_value(meta, "source_type", "document_type", "type") or "").lower()
    if source_type in {"original", "original_research", "journal", "conference", "paper"}:
        score += 0.12
        reasons.append("原始研究/正式论文 +0.12")
    elif source_type in {"review", "survey"}:
        score += 0.05
        reasons.append("综述来源 +0.05")
    elif source_type in {"blog", "news", "unknown"}:
        score -= 0.08
        reasons.append("非学术或类型不明 -0.08")
    if meta.get("peer_reviewed") is True:
        score += 0.10
        reasons.append("已同行评审 +0.10")
    if conditions.get("dataset"):
        score += 0.04
        reasons.append("报告数据集 +0.04")
    if conditions.get("sample_size"):
        score += 0.05
        reasons.append("报告样本量 +0.05")
    if conditions.get("year"):
        score += 0.03
        reasons.append("报告年份 +0.03")
    if meta.get("retracted") is True:
        score -= 0.50
        reasons.append("已撤稿 -0.50")
    return {"score": round(max(0.0, min(score, 1.0)), 2), "reasons": reasons,
            "source_type": source_type or "unknown"}


def extract_claim(item: dict[str, Any], query: str = "") -> dict[str, Any]:
    """Extract claim, stance, metric, values and experimental conditions."""
    text = str(item.get("text") or "").strip()
    meta = item.get("metadata") or {}
    year_match, sample_match, version_match = _YEAR_RE.search(text), _SAMPLE_RE.search(text), _VERSION_RE.search(text)
    sample_raw = _first_value(meta, "sample_size", "n")
    year_raw = _first_value(meta, "publication_year", "year")
    conditions = {
        "dataset": _extract_dataset(text, meta),
        "sample_size": int(sample_raw or sample_match.group(1)) if (sample_raw or sample_match) else None,
        "year": int(year_raw or year_match.group(1)) if (year_raw or year_match) else None,
        "version": str(_first_value(meta, "model_version", "version") or "") or
                   (f"v{version_match.group(1)}" if version_match else None),
    }
    claim = {"source": _source(item), "claim": text[:500], "stance": _stance(text),
             "metric": _extract_metric(text, query), "values": _extract_values(text),
             "conditions": conditions}
    claim["quality"] = _quality(item, conditions)
    return claim


def _condition_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    differences = []
    labels = {"dataset": "数据集", "sample_size": "样本量", "year": "年份", "version": "版本"}
    for key, label in labels.items():
        lv, rv = left["conditions"].get(key), right["conditions"].get(key)
        if lv is not None and rv is not None and str(lv).lower() != str(rv).lower():
            differences.append(f"{label}不同：{lv} vs {rv}")
    return differences


def _contradiction_reasons(left: dict[str, Any], right: dict[str, Any], shared: set[str]) -> list[str]:
    reasons = []
    if left["metric"] and right["metric"] and left["metric"] == right["metric"]:
        left_units, right_units = defaultdict(set), defaultdict(set)
        for value in left["values"]:
            left_units[value["unit"]].add(value["value"])
        for value in right["values"]:
            right_units[value["unit"]].add(value["value"])
        for unit in sorted(left_units.keys() & right_units.keys()):
            if left_units[unit] != right_units[unit]:
                reasons.append(f"同一指标“{left['metric']}”的 {unit} 数值不同："
                               f"{sorted(left_units[unit])} vs {sorted(right_units[unit])}")
    if (left["stance"] != "neutral" and right["stance"] != "neutral" and
            left["stance"] != right["stance"] and len(shared) >= 2):
        reasons.append("相关陈述呈相反肯定/否定极性")
    for positive, negative in _OPPOSITE_PAIRS:
        lt, rt = left["claim"], right["claim"]
        if ((any(x in lt for x in positive) and any(x in rt for x in negative)) or
                (any(x in lt for x in negative) and any(x in rt for x in positive))):
            reasons.append("相关陈述包含相反的技术论断")
            break
    return reasons


def _adjudicate(left: dict[str, Any], right: dict[str, Any], members: list[int]) -> dict[str, Any]:
    lq, rq = left["quality"]["score"], right["quality"]["score"]
    if abs(lq - rq) >= 0.15:
        preferred = members[0] if lq > rq else members[1]
        return {"status": "provisional_preference", "preferred_member": preferred,
                "reason": f"来源质量元数据存在可核验差异（{lq:.2f} vs {rq:.2f}）"}
    return {"status": "unresolved", "preferred_member": None,
            "reason": "现有来源质量和实验条件不足以可靠裁决"}


def analyze_evidence(results: list[dict[str, Any]], query: str = "") -> dict[str, Any]:
    """Separate genuine conflicts from condition-dependent divergences."""
    claims = [extract_claim(item, query) for item in results]
    conflicts, contextual = [], []
    for i, left in enumerate(claims):
        for j in range(i + 1, len(claims)):
            right = claims[j]
            if left["source"] == right["source"]:
                continue
            shared = _tokens(left["claim"]) & _tokens(right["claim"])
            if not shared:
                continue
            reasons = _contradiction_reasons(left, right, shared)
            if not reasons:
                continue
            members = [i + 1, j + 1]
            differences = _condition_differences(left, right)
            base = {"members": members, "sources": [left["source"], right["source"]],
                    "reasons": reasons, "condition_differences": differences}
            if differences:
                contextual.append({"id": f"context-{len(contextual) + 1}", **base,
                                   "status": "condition_dependent",
                                   "reason": "结论差异可能由实验条件不同造成，不能直接判为同条件矛盾"})
            else:
                conflicts.append({"id": f"conflict-{len(conflicts) + 1}", **base,
                                  **_adjudicate(left, right, members)})
    conflicted = {n for group in conflicts for n in group["members"]}
    policy = "normal_synthesis"
    if conflicts:
        policy = ("provisional_adjudication" if all(x["status"] == "provisional_preference" for x in conflicts)
                  else "disclose_both_sides")
    elif contextual:
        policy = "explain_condition_differences"
    return {"has_conflict": bool(conflicts), "has_contextual_difference": bool(contextual),
            "conflicts": conflicts, "contextual_differences": contextual, "claims": claims,
            "source_count": len({x["source"] for x in claims}),
            "independent_source_count": len({x["source"] for x in claims}),
            "conflicted_result_indices": sorted(conflicted), "policy": policy}


def annotate_evidence(results: list[dict[str, Any]], query: str = "") -> dict[str, Any]:
    report = analyze_evidence(results, query)
    by_index: dict[int, list[str]] = defaultdict(list)
    for group in report["conflicts"] + report["contextual_differences"]:
        for member in group["members"]:
            by_index[member].append(group["id"])
    for index, item in enumerate(results, 1):
        item["evidence_conflict"] = any(x.startswith("conflict-") for x in by_index.get(index, []))
        item["conflict_group_ids"] = by_index.get(index, [])
        item["structured_claim"] = report["claims"][index - 1]
    return report
