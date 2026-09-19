"""引用整理工具：去重并把文献条目格式化为 APA / BibTeX。"""
from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolResult, tool


class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""


class CitationArgs(BaseModel):
    papers: list[Paper] = Field(..., description="文献条目列表")
    style: str = Field("apa", description="apa | bibtex")


def _dedupe_key(p: Paper) -> str:
    return hashlib.md5(p.title.strip().lower().encode()).hexdigest()


def _to_apa(p: Paper) -> str:
    authors = ", ".join(p.authors) if p.authors else "Anonymous"
    year = f" ({p.year})" if p.year else ""
    venue = f". {p.venue}" if p.venue else ""
    return f"{authors}{year}. {p.title}{venue}."


def _to_bibtex(p: Paper, idx: int) -> str:
    key = f"ref{idx}_{hashlib.md5(p.title.encode()).hexdigest()[:6]}"
    author = " and ".join(p.authors)
    lines = [
        f"@article{{{key},",
        f"  title = {{{p.title}}},",
        f"  author = {{{author}}},",
    ]
    if p.year:
        lines.append(f"  year = {{{p.year}}},")
    if p.venue:
        lines.append(f"  journal = {{{p.venue}}},")
    lines.append("}")
    return "\n".join(lines)


@tool(name="citation", description="对文献列表去重并格式化为 APA / BibTeX 引用", params=CitationArgs)
async def citation(papers: list[Paper], style: str = "apa") -> dict[str, Any]:
    # 经 Pydantic 校验后入参为 dict，这里还原为 Paper 对象以便统一处理
    papers = [p if isinstance(p, Paper) else Paper(**p) for p in papers]
    try:
        seen: set[str] = set()
        unique: list[Paper] = []
        for p in papers:
            k = _dedupe_key(p)
            if k not in seen:
                seen.add(k)
                unique.append(p)

        if style.lower() == "bibtex":
            formatted = "\n\n".join(_to_bibtex(p, i) for i, p in enumerate(unique))
        else:
            formatted = "\n".join(_to_apa(p) for p in unique)

        return {
            "count": len(unique),
            "duplicates_removed": len(papers) - len(unique),
            "formatted": formatted,
        }
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"引用格式化失败: {e}")
