"""从 arXiv API 下载论文摘要，构造 RAG 知识文档目录。

用法: python scripts/fetch_arxiv_docs.py
输出: data/knowledge_docs/arxiv_<id>.txt  （Title / Authors / Abstract 纯文本）

只依赖标准库 urllib + xml.etree，无需第三方包。
"""
from __future__ import annotations

import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "knowledge_docs"

# 与「科研助手 Agent」主题相关的论文 (id 列表)
PAPER_IDS = [
    "2005.11401",  # RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "2210.03629",  # ReAct: Synergizing Reasoning and Acting in Language Models
    "2307.09288",  # Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
    "2312.10997",  # RAG Survey: Retrieval-Augmented Generation for Large Language Models: A Survey
    "2404.16130",  # GraphRAG: From Local to Global: A Graph RAG Approach to Query-Focused Summarization
    "2308.11432",  # Agent Survey: The Rise and Potential of Large Language Model Based Agents: A Survey
]

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def fetch_entries(ids: list[str]) -> list[dict]:
    """批量查询 arXiv API，返回 [{id, title, authors, published, summary}]。"""
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids) + "&max_results=20"
    req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read().decode("utf-8"))

    entries = []
    for e in root.findall(f"{ATOM}entry"):
        eid = e.find(f"{ATOM}id").text.split("/abs/")[-1]
        title = " ".join((e.find(f"{ATOM}title").text or "").split())
        summary = " ".join((e.find(f"{ATOM}summary").text or "").split())
        authors = [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")]
        published = e.find(f"{ATOM}published").text
        entries.append(
            {"id": eid, "title": title, "authors": authors, "published": published, "summary": summary}
        )
    return entries


def to_txt(paper: dict) -> str:
    """论文元数据 -> 知识文档纯文本。"""
    return (
        f"Title: {paper['title']}\n"
        f"ArXiv ID: {paper['id']}\n"
        f"Authors: {', '.join(paper['authors'])}\n"
        f"Published: {paper['published']}\n"
        f"Abstract: {paper['summary']}\n"
    )


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    entries = fetch_entries(PAPER_IDS)
    if not entries:
        print("[fetch] 未获取到任何论文，请检查网络", file=sys.stderr)
        return 1

    for p in entries:
        out = DOCS_DIR / f"arxiv_{p['id']}.txt"
        out.write_text(to_txt(p), encoding="utf-8")
        print(f"[fetch] {out.name}  <- {p['title'][:60]}...")
        time.sleep(3)  # 礼貌限速
    print(f"\n[fetch] 完成：{len(entries)} 篇论文摘要 -> {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
