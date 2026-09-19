"""网络检索工具。

支持多种后端（通过配置切换）：
- bing_html（默认）：抓取 www.bing.com/search 结果页并解析，免费、无需 key、国内直连可用。
- duckduckgo：免 key，抓取 lite 端点并解析结果（国内被墙）。
- serpapi / bing：需要对应 API Key。

返回结构化结果列表：[{title, url, snippet}]。
"""
from __future__ import annotations

import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.exceptions import ToolError
from app.tools.base import ToolResult, tool

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


class SearchArgs(BaseModel):
    query: str = Field(..., description="检索关键词")
    num_results: int = Field(5, ge=1, le=20, description="返回结果数量")


async def _search_duckduckgo(query: str, num: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS, follow_redirects=True) as client:
        resp = await client.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query, "kl": "cn-zh"},
        )
        resp.raise_for_status()
        html = resp.text

    # 轻量解析：抓 title + url + snippet
    rows = re.findall(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.S)

    results: list[dict[str, Any]] = []
    for i, (url, title) in enumerate(rows[:num]):
        snippet = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
        results.append(
            {
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": url,
                "snippet": snippet.strip(),
            }
        )
    return results


async def _search_bing_html(query: str, num: int) -> list[dict[str, Any]]:
    """抓取 www.bing.com/search 结果页并解析（免费、无 key、国内直连可用）。

    结果块结构：<li class="b_algo"> 内含 <h2><a href>标题</a></h2> 与 <p>摘要</p>。
    """
    import html as html_mod

    params = {"q": query, "count": min(num, 20), "setlang": "zh-hans"}
    async with httpx.AsyncClient(
        timeout=15, headers=_HEADERS, follow_redirects=True
    ) as client:
        resp = await client.get("https://www.bing.com/search", params=params)
        resp.raise_for_status()
        html_text = resp.text

    results: list[dict[str, Any]] = []
    for algo in re.findall(r'<li class="b_algo".*?</li>', html_text, re.S)[:num]:
        m = re.search(
            r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', algo, re.S
        )
        if not m:
            continue
        url = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        p = re.search(r"<p[^>]*>(.*?)</p>", algo, re.S)
        snippet = re.sub(r"<[^>]+>", "", p.group(1)).strip() if p else ""
        results.append(
            {
                "title": html_mod.unescape(title),
                "url": url,
                "snippet": html_mod.unescape(snippet),
            }
        )
    return results


async def _search_serpapi(query: str, num: int) -> list[dict[str, Any]]:
    s = get_settings().tools
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://serpapi.com/search.json",
            params={"q": query, "num": num, "api_key": s.serpapi_key},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic_results", [])
    ]


async def _search_bing(query: str, num: int) -> list[dict[str, Any]]:
    s = get_settings().tools
    async with httpx.AsyncClient(timeout=15, headers={"Ocp-Apim-Subscription-Key": s.bing_subscription_key}) as client:
        resp = await client.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": num},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": r.get("name", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
        for r in data.get("webPages", {}).get("value", [])
    ]


@tool(name="web_search", description="在网络上检索资料，返回标题/链接/摘要", params=SearchArgs)
async def web_search(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    provider = get_settings().tools.search_provider
    try:
        if provider == "bing_html":
            return await _search_bing_html(query, num_results)
        if provider == "duckduckgo":
            return await _search_duckduckgo(query, num_results)
        if provider == "serpapi":
            return await _search_serpapi(query, num_results)
        if provider == "bing":
            return await _search_bing(query, num_results)
        raise ToolError(f"不支持的搜索后端: {provider}")
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"检索失败: {e}")
