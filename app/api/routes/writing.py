"""论文写作工作流：大纲生成 → 分节草稿 → 参考文献格式化 → Markdown 导出。

学生科研产出闭环：在对话里查资料、溯源，在写作页把它变成论文草稿。
- 大纲：LLM 按论文类型生成结构化章节（json_mode 输出）；
- 草稿：按章节独立生成（避免单次超长截断），支持逐节生成/重写；
- 引用：原始引用列表按 APA / GB/T 7714 格式化；
- 导出：前端提交章节拼装为完整 Markdown 下载。
"""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from app.core.auth import User, get_current_user
from app.llm.base import ChatMessage, MessageRole
from app.llm.factory import get_llm

router = APIRouter(prefix="/writing", tags=["writing"])

PAPER_TYPES = ["综述", "研究论文", "实验报告", "学位论文"]

OUTLINE_PROMPT = (
    "你是学术写作导师。请为以下论文主题生成一份结构完整的中文论文大纲。\n"
    "论文类型：{paper_type}\n主题：{topic}\n"
    "要求：\n"
    "1. 输出 JSON 数组，每项 {{id: \"s1\", title: \"章节标题\", description: \"这一节写什么（50字内）\"}}；\n"
    "2. 包含引言、方法/主体、结论等完整章节，共 5-8 节；\n"
    "3. 只输出 JSON，不要任何前后缀或 markdown 代码块标记。"
)

DRAFT_PROMPT = (
    "你是学术写作助手。请为论文写「{section_title}」这一节的草稿。\n"
    "论文类型：{paper_type}\n主题：{topic}\n"
    "本节写作要点：{description}\n"
    "全文大纲：{outline}\n"
    "要求：用中文写 300-600 字，学术语气，Markdown 格式（可用小标题/列表/引用 [n]），"
    "只输出草稿正文，不要标题前缀。"
)

CITE_PROMPT = (
    "你是文献管理助手。请把下面的参考文献条目整理为 {style} 格式的规范引用列表。\n"
    "规则：每行一条，编号 [1][2]...，缺失字段（如页码）合理推断或省略，保持原始信息完整，"
    "中文文献保留中文，英文文献保留英文。只输出格式化后的列表。\n\n"
    "{refs}"
)


class OutlineRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    paper_type: str = Field("综述", description="综述/研究论文/实验报告/学位论文")


class OutlineSection(BaseModel):
    id: str
    title: str
    description: str


class DraftRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    paper_type: str = "综述"
    section: OutlineSection
    outline: list[OutlineSection] = Field(default_factory=list)


class CiteRequest(BaseModel):
    refs: list[str] = Field(..., min_length=1, max_length=50)
    style: str = Field("apa", description="apa / gbt7714")


def _parse_json_loose(text: str) -> list:
    """容错解析 LLM 输出：去 markdown 代码块/前后缀后取第一个 JSON 数组。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return []


async def _chat(prompt: str, json_mode: bool = False, max_tokens: int = 3000) -> str:
    from app.llm.gateway import get_llm_gateway
    from app.llm.router import TaskType

    resp = await get_llm_gateway().chat(
        [ChatMessage(role=MessageRole.USER, content=prompt)],
        json_mode=json_mode,
        max_tokens=max_tokens,
        task_type=TaskType.REASONING,
    )
    return (resp.content or "").strip()


async def _retrieve_kb_for(user: User, query: str, top_k: int = 4) -> str:
    """检索当前用户知识库，返回带来源标注的参考片段（失败/为空返回空串）。

    写作接 RAG：大纲与草稿都基于用户自己的资料生成，草稿中引用 [n] 可溯源。
    """
    try:
        from app.knowledge.user_kb import UserKnowledgeBase

        kb = UserKnowledgeBase(user.username)
        if kb.count() == 0:
            return ""
        hits = await kb.search(query, top_k=top_k)
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            src = (h.get("metadata") or {}).get("source") or h.get("source") or "知识库"
            text = (h.get("text") or h.get("content") or "")[:600]
            if text:
                parts.append(f"[{i}] 来源:{src}\n{text}")
        return "\n\n".join(parts)
    except Exception:  # noqa: BLE001
        return ""


@router.post("/outline")
async def generate_outline(req: OutlineRequest, user: User = Depends(get_current_user)) -> dict:
    """按主题 + 论文类型生成结构化大纲（自动检索知识库资料作为参考）。"""
    kb_refs = await _retrieve_kb_for(user, req.topic, top_k=4)
    ref_block = ""
    if kb_refs:
        ref_block = (
            "\n以下是用户知识库中与主题相关的参考资料，大纲应体现其中的研究内容：\n"
            f"{kb_refs}\n"
            "（大纲中可引用这些资料，如 [1][2]）"
        )
    prompt = OUTLINE_PROMPT.format(paper_type=req.paper_type, topic=req.topic) + ref_block
    text = await _chat(prompt, json_mode=True, max_tokens=2000)
    raw = _parse_json_loose(text)
    sections = []
    for i, s in enumerate(raw, 1):
        if isinstance(s, dict) and s.get("title"):
            sections.append(
                OutlineSection(
                    id=str(s.get("id") or f"s{i}"),
                    title=str(s["title"])[:100],
                    description=str(s.get("description") or "")[:120],
                )
            )
    if not sections:
        sections = [OutlineSection(id="s1", title="引言", description="研究背景与问题提出")]
    return {
        "topic": req.topic,
        "paper_type": req.paper_type,
        "sections": [s.model_dump() for s in sections],
        "kb_used": bool(kb_refs),
    }


@router.post("/draft")
async def generate_draft(req: DraftRequest, user: User = Depends(get_current_user)) -> dict:
    """按章节生成草稿（单节独立生成，可逐节重写；自动检索知识库资料并标注引用）。"""
    outline_text = "\n".join(f"- {s.title}：{s.description}" for s in req.outline) or req.section.title
    kb_refs = await _retrieve_kb_for(user, f"{req.topic} {req.section.title}", top_k=4)
    ref_block = ""
    if kb_refs:
        ref_block = (
            "\n以下是从用户知识库检索到的参考资料（务必基于这些资料写作，"
            "在正文中用 [1][2] 标注引用来源）：\n"
            f"{kb_refs}"
        )
    prompt = DRAFT_PROMPT.format(
        section_title=req.section.title,
        paper_type=req.paper_type,
        topic=req.topic,
        description=req.section.description,
        outline=outline_text,
    ) + ref_block
    content = await _chat(prompt, max_tokens=2500)
    return {
        "section_id": req.section.id,
        "title": req.section.title,
        "content": content,
        "kb_used": bool(kb_refs),
    }


@router.post("/cite")
async def format_citations(req: CiteRequest, _: User = Depends(get_current_user)) -> dict:
    """参考文献格式化为 APA 或 GB/T 7714。"""
    style = "APA 第7版" if req.style == "apa" else "GB/T 7714-2015"
    refs_text = "\n".join(f"- {r}" for r in req.refs)
    prompt = CITE_PROMPT.format(style=style, refs=refs_text)
    formatted = await _chat(prompt, max_tokens=2500)
    return {"style": req.style, "formatted": formatted}


class ExportRequest(BaseModel):
    topic: str
    paper_type: str = "综述"
    sections: list[dict] = Field(default_factory=list)  # [{title, content}]
    format: str = Field("md", pattern="^(md|docx|pdf)$")


@router.post("/export")
async def export_paper(req: ExportRequest, user: User = Depends(get_current_user)):
    """拼装完整论文并下载为 Markdown、Word 或 PDF。"""
    lines = [f"# {req.topic}", f"\n> 论文类型：{req.paper_type} · 由科研助手 Agent 辅助生成\n"]
    for s in req.sections:
        title = s.get("title") or ""
        content = (s.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"\n## {title}\n")
        lines.append(content)
    md = "\n".join(lines)
    from urllib.parse import quote

    extension = req.format
    safe_name = quote(f"{req.topic}.{extension}")
    disposition = f"attachment; filename=\"paper.{extension}\"; filename*=UTF-8''{safe_name}"
    if req.format == "docx":
        from app.services.document_export import render_docx

        return Response(
            render_docx(md),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": disposition},
        )
    if req.format == "pdf":
        from app.services.document_export import render_pdf

        return Response(
            render_pdf(md),
            media_type="application/pdf",
            headers={"Content-Disposition": disposition},
        )
    return PlainTextResponse(
        md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )
