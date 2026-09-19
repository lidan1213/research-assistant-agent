"""概念笔记路由：把科研中遇到的概念一键整理成规范笔记（带标签），存为 Markdown。

- POST /api/notes   生成并保存笔记（输入概念名，LLM 整理成结构化笔记）
- GET  /api/notes   列出全部笔记
- GET  /api/notes/{slug}  查看单篇
- DELETE /api/notes/{slug} 删除单篇

存储：data/notes/{slug}.md，frontmatter 含 title / tags / created / source。
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.auth import User, get_current_user
from app.core.exceptions import ForbiddenError
from app.llm.base import ChatMessage, MessageRole
from app.llm.factory import get_llm

router = APIRouter(prefix="/notes", tags=["notes"], dependencies=[Depends(get_current_user)])

# 笔记存储目录：支持环境变量覆盖（测试隔离用），默认 ./data/notes
NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./data/notes"))

# slug 白名单：中文字符 + 字母数字 + 连字符 + 下划线（防路径穿越）
_SLUG_RE = re.compile(r"^[\w\u4e00-\u9fff-]+$")


def _check_slug(slug: str) -> None:
    """校验 slug 格式，防路径穿越（GET/DELETE/PUT 共用）。"""
    if not slug or not _SLUG_RE.match(slug) or ".." in slug or "/" in slug or "\\" in slug:
        raise ForbiddenError("无效的笔记标识")


class NoteCreateRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=120, description="要整理的概念名（concept 模式）或对话主题（summary 模式）")
    note: str | None = Field(None, description="用户提供的补充笔记/资料（可选，没有则仅靠 LLM 知识）")
    kind: str = Field("concept", description="concept=概念笔记 | summary=对话总结笔记")
    session_id: str | None = Field(None, description="summary 模式：要总结的会话 id（默认当前会话）")
    to_kb: bool = Field(False, description="保存笔记后同时存入用户知识库（笔记变成可检索资料）")


class NoteItem(BaseModel):
    slug: str
    title: str
    tags: list[str]
    created: str
    source: str
    kb_saved: bool = False


NOTE_PROMPT = """你是一名科研助手。请把下面这个科研概念整理成一份规范的笔记，用 JSON 输出（不要输出其他内容）：

{{"title": "概念名", "definition": "一句话通俗定义（30字内）", "points": ["要点1", "要点2", "要点3"], "related": ["相关概念或文献，最多3个"], "tags": ["标签1", "标签2", "标签3"]}}

要求：
- definition 通俗易懂，30 字以内
- points 3 条，每条 40 字以内，讲清核心
- tags 3 个，简洁、可检索（如：深度学习、NLP、注意力机制）
- 若用户提供了补充笔记，优先参考其中的内容

概念：{concept}
用户补充笔记（可选）：{note}"""

SUMMARY_PROMPT = """你是一名科研助手。请把下面的对话记录总结成一份规范的笔记，用 JSON 输出（不要输出其他内容）：

{{"title": "对话主题（简短，如：RAG综述写作规划）", "definition": "本次对话解决了什么问题（30字内）", "points": ["结论或要点1", "要点2", "要点3"], "related": ["提到的关键概念/资料，最多3个"], "tags": ["标签1", "标签2", "标签3"]}}

要求：
- points 提取对话中的核心结论、规划或收获，3 条，每条 40 字以内
- tags 3 个，简洁、可检索

对话记录：
{transcript}"""


def _slugify(concept: str) -> str:
    """概念名 -> 安全文件名 slug（保留中文，转小写，空格/符号转 -）。"""
    s = unicodedata.normalize("NFKC", concept).strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or f"note-{uuid.uuid4().hex[:6]}"


def _parse_note_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON（容忍 markdown 代码块包裹）。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    # 提取第一个 { ... } 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):  # 必须是对象；裸字符串等一律回退
                return data
        except json.JSONDecodeError:
            pass
    # 回退：手搓最小结构
    return {"title": "未命名笔记", "definition": text[:100], "points": [], "related": [], "tags": []}


@router.post("", response_model=NoteItem)
async def create_note(
    req: NoteCreateRequest,
    user: User = Depends(get_current_user),
) -> NoteItem:
    """生成并保存笔记（concept=概念笔记 | summary=对话总结笔记）。"""
    from app.llm.gateway import get_llm_gateway
    from app.llm.router import TaskType

    gateway = get_llm_gateway()
    try:
        if req.kind == "summary":
            # 对话总结模式：读取会话历史（归属校验），生成总结 prompt
            sid = req.session_id or req.concept  # 前端可把 session_id 放 concept 字段简化
            prefix = f"{user.username}__"
            full_sid = sid if sid.startswith(prefix) else f"{prefix}{sid}"
            from app.agent.memory import SQLiteStore

            store = SQLiteStore(get_settings().memory.sqlite_path)
            try:
                msgs = await store.get(full_sid)
            finally:
                store.close()
            if not msgs:
                raise ForbiddenError("会话为空或不存在")
            # 只取 user/assistant 的正文，组装成可读记录
            transcript = "\n".join(
                f"{'用户' if m.role.value == 'user' else 'AI'}: {m.content[:200]}"
                for m in msgs
                if m.role.value in ("user", "assistant") and m.content
            )[:4000]
            prompt = SUMMARY_PROMPT.format(transcript=transcript or "（无内容）")
            title_hint = req.concept or "对话总结"
        else:
            prompt = NOTE_PROMPT.format(
                concept=req.concept,
                note=req.note or "（无）",
            )
            title_hint = req.concept
        resp = await gateway.chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            tools=None,
            task_type=TaskType.SUMMARIZATION if req.kind == "summary" else TaskType.REASONING,
        )
        data = _parse_note_json(resp.content or "")
    except Exception as e:  # noqa: BLE001
        raise ForbiddenError(f"笔记生成失败: {e}") from e

    title = (data.get("title") or title_hint).strip() or title_hint
    slug = _slugify(title)
    tags = [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()][:5]
    points = [str(p).strip() for p in (data.get("points") or []) if str(p).strip()][:5]
    related = [str(r).strip() for r in (data.get("related") or []) if str(r).strip()][:5]
    definition = (data.get("definition") or "").strip()

    created = date.today().isoformat()
    md = (
        "---\n"
        f"title: {title}\n"
        f"tags: [{', '.join(tags)}]\n"
        f"created: {created}\n"
        f"source: {user.username}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"> {definition}\n\n"
        "## 要点\n\n"
        + "\n".join(f"- {p}" for p in points)
        + "\n\n## 相关概念 / 文献\n\n"
        + "\n".join(f"- {r}" for r in related)
        + "\n\n---\n"
        f"*由 {user.username} 于 {datetime.now().strftime('%Y-%m-%d %H:%M')} 整理*\n"
    )

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTES_DIR / f"{slug}.md"
    # 同名已存在：追加时间戳避免覆盖
    if path.exists():
        slug = f"{slug}-{uuid.uuid4().hex[:4]}"
        path = NOTES_DIR / f"{slug}.md"
    path.write_text(md, encoding="utf-8")

    # 闭环：笔记一键入库（to_kb=True 时存为知识库文档，之后可被 RAG 检索）
    kb_saved = False
    if req.to_kb:
        try:
            from app.knowledge.user_kb import UserKnowledgeBase

            kb = UserKnowledgeBase(user.username)
            kb_text = f"# {title}\n\n{definition}\n\n## 要点\n" + "\n".join(
                f"- {p}" for p in points
            )
            res = await kb.add_document(f"note_{slug}.md", kb_text)
            kb_saved = not res.get("duplicate", True) or res.get("chunks", 0) > 0
            # 同步失效检索缓存 + 图谱增量抽取
            try:
                from app.cache.retrieval_cache import invalidate_user_kb

                await invalidate_user_kb(user.username)
            except Exception:  # noqa: BLE001
                pass
            try:
                import asyncio

                from app.knowledge.kg import sync_document_to_kg

                asyncio.create_task(
                    sync_document_to_kg(
                        user.username, f"note_{slug}.md", f"note_{slug}.md", kb_text
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            kb_saved = False

    return NoteItem(
        slug=slug,
        title=title,
        tags=tags,
        created=created,
        source=user.username,
        kb_saved=kb_saved,
    )


@router.get("")
async def list_notes(
    tag: str | None = Query(None, description="按标签过滤（精确匹配）"),
    user: User = Depends(get_current_user),
) -> dict:
    """列出全部笔记（admin 看全部；普通用户只看到自己创建的）。支持按标签过滤。"""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(NOTES_DIR.glob("*.md"), reverse=True):
        meta = _read_frontmatter(p)
        if user.role != "admin" and meta.get("source") != user.username:
            continue
        if tag and tag not in meta.get("tags", []):
            continue
        items.append(
            NoteItem(
                slug=p.stem,
                title=meta.get("title", p.stem),
                tags=meta.get("tags", []),
                created=meta.get("created", ""),
                source=meta.get("source", ""),
            )
        )
    return {"notes": items, "count": len(items), "tags": _all_tags(user)}


@router.get("/{slug}")
async def get_note(slug: str, user: User = Depends(get_current_user)) -> dict:
    """查看单篇笔记（含正文）。"""
    _check_slug(slug)
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise ForbiddenError("笔记不存在")
    meta = _read_frontmatter(path)
    if user.role != "admin" and meta.get("source") != user.username:
        raise ForbiddenError("无权查看他人的笔记")
    return {"slug": slug, "content": path.read_text(encoding="utf-8")}


@router.get("/{slug}/download")
async def download_note(slug: str, user: User = Depends(get_current_user)):
    """下载单篇笔记为 .md 文件（仅创建者或 admin）。"""
    _check_slug(slug)
    from fastapi.responses import FileResponse

    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise ForbiddenError("笔记不存在")
    meta = _read_frontmatter(path)
    if user.role != "admin" and meta.get("source") != user.username:
        raise ForbiddenError("无权下载他人的笔记")
    return FileResponse(
        path,
        media_type="text/markdown; charset=utf-8",
        filename=f"{slug}.md",
    )


class NoteUpdateRequest(BaseModel):
    content: str = Field(..., description="编辑后的完整 markdown 内容（含 frontmatter）")


@router.put("/{slug}")
async def update_note(
    slug: str, req: NoteUpdateRequest, user: User = Depends(get_current_user)
) -> NoteItem:
    """编辑笔记（仅创建者或 admin）：整体覆盖 markdown 内容，并同步 frontmatter 索引。"""
    _check_slug(slug)
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise ForbiddenError("笔记不存在")
    meta = _read_frontmatter(path)
    if user.role != "admin" and meta.get("source") != user.username:
        raise ForbiddenError("无权编辑他人的笔记")
    new_meta = _read_frontmatter_text(req.content)
    # 保留原创建者/创建时间，标题/标签取新内容
    path.write_text(req.content, encoding="utf-8")
    return NoteItem(
        slug=path.stem,
        title=new_meta.get("title") or path.stem,
        tags=new_meta.get("tags", []),
        created=new_meta.get("created") or meta.get("created") or "",
        source=new_meta.get("source") or meta.get("source") or user.username,
    )


@router.delete("/{slug}")
async def delete_note(slug: str, user: User = Depends(get_current_user)) -> dict:
    """删除笔记（仅创建者或 admin）。"""
    _check_slug(slug)
    path = NOTES_DIR / f"{slug}.md"
    if not path.exists():
        raise ForbiddenError("笔记不存在")
    meta = _read_frontmatter(path)
    if user.role != "admin" and meta.get("source") != user.username:
        raise ForbiddenError("无权删除他人的笔记")
    path.unlink()
    return {"deleted": slug}


def _read_frontmatter(path: Path) -> dict:
    """解析 markdown frontmatter（--- 块内 key: value，tags 为 [a, b] 列表）。"""
    meta: dict = {"title": path.stem, "tags": [], "created": "", "source": ""}
    try:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            block = text.split("---", 2)[1]
            for line in block.strip().splitlines():
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "tags":
                    meta["tags"] = [t.strip() for t in v.strip("[]").split(",") if t.strip()]
                else:
                    meta[k] = v
    except Exception:  # noqa: BLE001
        pass
    return meta


def _read_frontmatter_text(text: str) -> dict:
    """从 markdown 文本解析 frontmatter（不落盘）。"""
    meta: dict = {"title": "", "tags": [], "created": "", "source": ""}
    try:
        if text.startswith("---"):
            block = text.split("---", 2)[1]
            for line in block.strip().splitlines():
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "tags":
                    meta["tags"] = [t.strip() for t in v.strip("[]").split(",") if t.strip()]
                else:
                    meta[k] = v
    except Exception:  # noqa: BLE001
        pass
    return meta


def _all_tags(user: User) -> list[str]:
    """当前用户可见的全部标签（去重排序）。"""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    tags: set[str] = set()
    for p in NOTES_DIR.glob("*.md"):
        meta = _read_frontmatter(p)
        if user.role != "admin" and meta.get("source") != user.username:
            continue
        tags.update(meta.get("tags", []))
    return sorted(tags)
