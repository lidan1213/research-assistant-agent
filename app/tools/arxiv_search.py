"""arXiv 论文检索工具。

基于官方 `arxiv` 包（>=2.1，已验证 4.0 兼容），返回论文元数据
（标题、作者、摘要、链接、发表日期、PDF 链接等），适合做文献调研与综述。

可选 `download=True`：把命中论文的 PDF 下载到 `download_dir`，并在结果中附带
`local_path`，便于后续直接交给 `pdf_reader` 进行全文精读，形成「检索 → 精读」工作流。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import arxiv
import httpx
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.tools.base import ToolResult, tool

logger = get_logger("tools.arxiv")


class ArxivArgs(BaseModel):
    query: str = Field(..., description="检索式，如 'large language model agent'")
    max_results: int = Field(5, ge=1, le=50, description="最大返回数量")
    download: bool = Field(False, description="是否把 PDF 下载到本地（与 pdf_reader 配合）")
    download_dir: str = Field("./data/papers", description="PDF 下载目录")


def _client() -> "arxiv.Client":
    return arxiv.Client(page_size=100, delay_seconds=1, num_retries=1)


async def _download_pdf(pdf_url: str, dest_dir: str) -> str | None:
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, os.path.basename(pdf_url.split("?")[0]) or "paper.pdf")
    if os.path.isfile(fname):
        return fname
    # arXiv 对 PDF 端点有频控，失败重试几次可以提高成功率
    last_err: str = ""
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(pdf_url)
                resp.raise_for_status()
                if not resp.content or "html" in resp.headers.get("content-type", ""):
                    raise ValueError("返回内容不是 PDF")
                with open(fname, "wb") as f:
                    f.write(resp.content)
            return fname
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:120]
            await asyncio.sleep(2 * (attempt + 1))
    logger.warning("PDF 下载失败（已重试）: %s | %s", pdf_url, last_err)
    return None


@tool(name="arxiv_search", description="在 arXiv 检索学术论文（含摘要与 PDF 链接，可下载全文）", params=ArxivArgs)
async def arxiv_search(
    query: str,
    max_results: int = 5,
    download: bool = False,
    download_dir: str = "./data/papers",
) -> list[dict[str, Any]] | ToolResult:
    last_err = ""
    # arXiv API 偶发 429 限流：退避重试，提高实际检索与测试的稳定性
    for attempt in range(3):
        try:
            client = _client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results: list[dict[str, Any]] = []
            for r in client.results(search):
                item = {
                    "title": r.title.strip(),
                    "authors": [a.name for a in r.authors],
                    "summary": r.summary.replace("\n", " ").strip(),
                    "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                    "entry_id": r.entry_id,
                    "pdf_url": r.pdf_url,
                    "categories": r.categories,
                    "comment": getattr(r, "comment", "") or "",
                }
                if download and r.pdf_url:
                    local = await _download_pdf(r.pdf_url, download_dir)
                    if local:
                        item["local_path"] = local
                results.append(item)
            return results
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if "429" in last_err and attempt < 1:
                await asyncio.sleep(1)
                logger.warning("arXiv 429 限流，第 %d 次退避重试", attempt + 1)
                continue
            break
    return ToolResult(success=False, error=f"arXiv 检索失败: {last_err}")
