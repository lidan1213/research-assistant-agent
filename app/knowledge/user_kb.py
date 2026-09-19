"""用户个人知识库：每个用户可拥有多个独立的知识库（默认库 kb_{username} + 命名库 kb_{username}__{库名}）。

- 与全局知识库（collection=knowledge）隔离：用户上传的文件只进自己的 collection。
- 多库隔离：用户可创建多个专属文献库（如"量子点文献"、"RAG 综述"），
  每个库独立 collection（kb_{username}__{库名}），对话时可选择使用哪个库或不使用知识库。
- 复用 RestChromaVectorStore + Retriever：分块、Ollama 嵌入、混合检索（向量+BM25）、重排全部复用。
- 上传支持 .txt/.md/.pdf：文本类直接读取，PDF 用 pdf_reader 的解析逻辑提取正文。
- 供路由与 knowledge_search 工具调用：通过 collection_name=f"kb_{username}__{库名}" 隔离。

权限模型：
- 登录用户：可上传/查看/删除/检索自己的知识库；
- admin：可额外查看所有用户库的元信息。

⚠️ ChromaDB 0.4.x collection 名仅允许 ASCII [a-zA-Z0-9_-]：
- 中文库名 → _ascii_slug() 哈希映射（显示名存 metadata.kb_display_name）
- 中文/非 ASCII 用户名 → _user_ascii_part() 用可逆 base64（kb__u_<b64>），ASCII 用户名保持原名（兼容旧数据）
"""
from __future__ import annotations

import base64
import asyncio
import hashlib
import os
import re
from typing import Optional

from app.config import get_settings
from app.core.logging import get_logger
from app.knowledge.chunking import Chunker
from app.knowledge.embeddings import EmbeddingModel
from app.knowledge.retriever import Retriever
from app.knowledge.stores.chroma_http import RestChromaVectorStore

logger = get_logger("user_kb")

COLLECTION_PREFIX = "kb_"
# collection 名中用户名与库名的分隔符：kb_{username}__{库名}
KB_NAME_SEP = "__"


def _collection_prefix() -> str:
    """ChromaDB collection 前缀：测试环境（CHROMA_COLLECTION_PREFIX=test_）下
    使用 test_ 前缀隔离，避免测试数据污染生产 collection。"""
    return os.environ.get("CHROMA_COLLECTION_PREFIX", COLLECTION_PREFIX)

# 允许的文件类型 -> 解析函数名
TEXT_EXTS = {".txt", ".md", ".text"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 扩展名 -> MIME（OCR 请求图片时用）
IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename or "")[1].lower() in IMAGE_EXTS

MAX_FILE_CHARS = 200_000  # 单文件入库字符上限（超长截断，防止超大文件拖垮嵌入）

# 单进程内按知识库串行化“检索/版本切换”，避免查询看到尚未激活的新版本。
_KB_UPDATE_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _kb_update_lock(username: str, kb_name: str | None) -> asyncio.Lock:
    return _KB_UPDATE_LOCKS.setdefault((username, kb_name or "__default__"), asyncio.Lock())

# 库名校验：允许中文/字母/数字/下划线/短横线，长度 1-30
import re as _re

KB_NAME_RE = _re.compile(r"^[\w\u4e00-\u9fa5-]{1,30}$")


def _ascii_slug(kb_name: str) -> str:
    """库名 → ChromaDB collection 兼容的 ASCII slug。

    ChromaDB 0.4.x collection 名仅允许 [a-zA-Z0-9_-]（3-63 字符），中文/unicode 会 500。
    纯 ASCII 合法名直接用；含中文的库名映射为 lib_{md5前8}（可复现，同名库 slug 稳定）。
    中文显示名存入 collection metadata（kb_display_name），由 list_user_kbs 反查。
    """
    if kb_name.isascii() and KB_NAME_RE.match(kb_name):
        return kb_name
    return "lib_" + hashlib.md5(kb_name.encode("utf-8")).hexdigest()[:8]


def _b64url(s: str) -> str:
    """URL 安全 base64（去 padding），用于把非 ASCII 用户名编码进 collection 名。"""
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def _user_ascii_part(username: str) -> str:
    """用户名 → collection 名中的 ASCII 段。

    - ASCII 用户名：原样返回（兼容旧数据：kb_student1 等）；
    - 非 ASCII（中文等）："_u_" + urlsafe base64（可逆，_u_ 前缀不会与 ASCII 用户名冲突，
      因为注册校验禁止用户名以下划线开头）。
    """
    if username.isascii():
        return username
    return "_u_" + _b64url(username)


def _user_from_ascii_part(part: str) -> str:
    """把 collection 名中的 ASCII 段还原为用户名（_user_ascii_part 的逆操作）。"""
    if part.startswith("_u_"):
        raw = part[3:]
        pad = "=" * (-len(raw) % 4)
        return base64.urlsafe_b64decode((raw + pad).encode("ascii")).decode("utf-8")
    return part


def normalize_kb_name(name: str) -> str:
    """校验并规范化库名。非法字符/超长/含分隔符 __ 时抛 ValueError。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("库名不能为空")
    if KB_NAME_SEP in name:
        raise ValueError("库名不能包含 '__'（保留分隔符）")
    if not KB_NAME_RE.match(name):
        raise ValueError("库名仅支持中文/字母/数字/下划线/短横线，1-30 个字符")
    return name


def collection_name_for(username: str, kb_name: str | None = None) -> str:
    """用户知识库的 ChromaDB collection 名。

    - 默认库（kb_name=None）：kb_{user_ascii}（与旧版单库兼容，历史数据不动）
    - 命名库：kb_{user_ascii}__{ascii_slug(kb_name)}（中文库名映射为 lib_xxx）
    其中 user_ascii：ASCII 用户名原样；非 ASCII 用户名 _u_<base64>（ChromaDB 名仅允许 ASCII）。
    """
    u = _user_ascii_part(username)
    if kb_name:
        return f"{_collection_prefix()}{u}{KB_NAME_SEP}{_ascii_slug(kb_name)}"
    return f"{_collection_prefix()}{u}"


def username_from_collection(collection: str) -> Optional[str]:
    """从 collection 名反解用户名（kb_xxx -> xxx，kb_xxx__yyy -> xxx）；非用户库返回 None。"""
    if collection.startswith(_collection_prefix()):
        rest = collection[len(_collection_prefix()):]
        return _user_from_ascii_part(rest.split(KB_NAME_SEP)[0])
    return None


def kb_name_from_collection(collection: str) -> Optional[str]:
    """从 collection 名反解库名：kb_xxx__yyy -> yyy；默认库（无 __）返回 None。

    注意：中文库名的 collection 是 lib_<hash>，无法直接反解出中文名——真实显示名
    由 list_user_kbs() 从 collection metadata（kb_display_name）读取。此函数仅用于
    无 metadata 时的兜底（返回 slug 本身）。
    """
    if collection.startswith(_collection_prefix()):
        rest = collection[len(_collection_prefix()):]
        if KB_NAME_SEP in rest:
            return rest.split(KB_NAME_SEP, 1)[1]
    return None


def extract_text(filename: str, data: bytes) -> str:
    """按扩展名提取文本：.txt/.md 直接读，.pdf 走解析。图片请走 ocr_image_text()。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="replace")
    if ext in PDF_EXTS:
        return _extract_pdf_text(data)
    return ""


async def ocr_image_text(filename: str, data: bytes) -> str:
    """图片 OCR：用主模型视觉能力提取图中文字（失败返回空串）。

    图内容以 base64 data URL 随消息发送（多模态格式），模型扮演 OCR 角色
    只输出文字本身。图片过大时降采样有损——上传前前端已压缩。
    """
    import base64

    try:
        from app.llm.base import ChatMessage, MessageRole
        from app.llm.factory import get_llm

        ext = os.path.splitext(filename)[1].lower()
        mime = IMAGE_MIME.get(ext, "image/png")
        img_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        from app.llm.gateway import get_llm_gateway
        from app.llm.router import TaskType

        resp = await get_llm_gateway().chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content="你是 OCR 引擎。请提取这张图片中的全部文字（含图表标题、坐标轴标签、表格内容），"
                    "按原顺序输出纯文本。如果图片没有文字，只输出「无文字」。不要添加任何解释或前后缀。",
                    images=[img_url],
                )
            ],
            task_type=TaskType.REASONING,  # OCR 视觉任务走主模型（多模态）
            max_tokens=2000,
        )
        text = (resp.content or "").strip()
        if text == "无文字":
            return ""
        return text
    except Exception as e:  # noqa: BLE001
        logger.warning(f"图片 OCR 失败（{filename}）: {e}")
        return ""


def _extract_pdf_text(data: bytes) -> str:
    """提取 PDF 正文（复用 pdf_reader 的解析依赖：pypdf 优先，PyPDF2 回退）。"""
    import io

    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(data))
    except ImportError:
        import PyPDF2  # type: ignore

        reader = PyPDF2.PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts)


async def _generate_doc_summary(text: str, max_len: int = 120) -> str:
    """上传文档自动摘要：用辅助模型生成"这篇讲什么"（默认 120 字内）。

    - 路由到辅助模型（ModelRouter），不占主模型额度；
    - 文本过短（<30 字）或 LLM 失败时返回空串（摘要缺失不影响入库）。
    """
    text = (text or "").strip()
    if len(text) < 30:
        return ""
    try:
        from app.llm.base import ChatMessage, MessageRole
        from app.llm.gateway import get_llm_gateway
        from app.llm.router import TaskType

        resp = await get_llm_gateway().chat(
            [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "你是文献摘要器。用一句话概括下面文档的核心内容"
                        f"（{max_len} 字以内），说明研究对象与主要结论。只输出摘要正文。"
                    ),
                ),
                ChatMessage(role=MessageRole.USER, content=text),
            ],
            tools=None,
            temperature=0.2,
            task_type=TaskType.SUMMARIZATION,
        )
        summary = (resp.content or "").strip()[:max_len]
        return summary
    except Exception as e:  # noqa: BLE001
        logger.debug("文档摘要生成失败（忽略）: %s", e)
        return ""


class UserKnowledgeBase:
    """单个用户的个人知识库（默认库 collection=kb_{username}，命名库 kb_{username}__{库名}）。"""

    def __init__(
        self,
        username: str,
        kb_name: str | None = None,
        *,
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> None:
        self.username = username
        self.kb_name = kb_name  # None=默认库
        self.collection = collection_name_for(username, kb_name)
        self._store = RestChromaVectorStore(collection_name=self.collection)
        self._retriever = Retriever(
            embedding=EmbeddingModel(),
            store=self._store,
            chunker=Chunker(chunk_size=chunk_size, overlap=overlap),
            use_bm25=True,
            use_hybrid=True,
        )

    # ---------- 写 ----------
    async def add_document(self, filename: str, text: str, metadata: dict | None = None) -> dict:
        """把一个文件全文入库（自动分块），返回 {chunks, duplicate}。

        duplicate=True 表示内容与已有文档完全相同（内容哈希去重），未重复入库。
        """
        async with _kb_update_lock(self.username, self.kb_name):
            return await self._add_document_version(filename, text, metadata)

    async def _add_document_version(self, filename: str, text: str, metadata: dict | None = None) -> dict:
        """Build a pending index version, then atomically activate it."""
        text = text.strip()
        if not text:
            return {"chunks": 0, "duplicate": False}
        if len(text) > MAX_FILE_CHARS:
            logger.warning("文件 %s 超过 %d 字符，截断入库", filename, MAX_FILE_CHARS)
            text = text[:MAX_FILE_CHARS]

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # 兼容版本注册表上线前的历史文档。
        for d in self._store.documents():
            meta = d.get("metadata") or {}
            # 优先哈希匹配（新数据）；旧数据无 content_hash 时回退全文比对
            if meta.get("content_hash") == content_hash or (
                (d.get("text") or "").strip() == text and text
            ):
                logger.info("文件 %s 内容与已有文档重复，跳过入库", filename)
                return {"chunks": 0, "duplicate": True}

        evidence_metadata = {
            key: value for key, value in (metadata or {}).items()
            if value not in (None, "") and key in {
                "source_type", "peer_reviewed", "publication_year", "sample_size",
                "dataset", "model_version", "retracted",
            }
        }
        from app.knowledge.versioning import version_store

        pending = version_store.begin(
            self.username, self.kb_name, filename, content_hash, evidence_metadata
        )
        if pending["duplicate"]:
            return {"chunks": 0, "duplicate": True,
                    "version_id": pending.get("active_version_id")}
        legacy_old_ids = {
            str((d.get("metadata") or {}).get("doc_id") or d.get("id"))
            for d in self._store.documents()
            if (d.get("metadata") or {}).get("source") == filename
        }
        doc = {
            "doc_id": pending["index_doc_id"],
            "source": filename,
            "text": text,
            "metadata": {
                "content_hash": content_hash,
                "logical_doc_id": pending["logical_doc_id"],
                "version_id": pending["version_id"],
                "version_number": pending["version_number"],
                **evidence_metadata,
            },
        }
        # 上传自动摘要：用辅助模型生成"这篇讲什么"（120 字内），存入 metadata 供列表/引用展示。
        # 失败静默（摘要缺失不影响入库）。
        doc["metadata"]["summary"] = await _generate_doc_summary(text[:3000])
        version_store.mark_indexing(pending["version_id"])
        try:
            count = self._retriever.ingest_documents([doc], clear=False)
            indexed = any(
                (d.get("metadata") or {}).get("version_id") == pending["version_id"]
                for d in self._store.documents()
            )
            if count <= 0 or not indexed:
                raise RuntimeError("新版本索引未写入，继续保留旧版本")
            activated = version_store.activate(pending["version_id"], count)
        except Exception as exc:
            self._retriever.delete_by_metadata("version_id", pending["version_id"])
            version_store.fail(pending["version_id"], str(exc))
            raise

        old_ids = set(legacy_old_ids)
        if activated.get("previous_index_doc_id"):
            old_ids.add(str(activated["previous_index_doc_id"]))
        old_ids.discard(doc["doc_id"])
        for old_id in old_ids:
            self._retriever.delete_by_metadata("doc_id", old_id)
        return {
            "doc_id": doc["doc_id"],
            "logical_doc_id": pending["logical_doc_id"],
            "version_id": pending["version_id"],
            "version_number": pending["version_number"],
            "kb_revision": activated["revision"],
            "chunks": count,
            "duplicate": False,
            "version_updated": bool(old_ids),
            "summary": doc["metadata"].get("summary", ""),
        }

    # ---------- 读 ----------
    async def search(self, query: str, top_k: int = 5, min_score: float = 0.35) -> list[dict]:
        """在用户知识库中语义检索。

        min_score：相似度下限（ChromaDB cosine 相似度 ∈ [0,1]）。nomic-embed-text 对
        完全无关的短文本基线约 0.30-0.35；跨语言（如英文查中文文档）的相关命中可能
        仅 0.40 左右，故阈值取 0.35 只过滤明显无关，避免误杀。
        """
        async with _kb_update_lock(self.username, self.kb_name):
            hits = await self._retriever.retrieve(query, top_k=top_k)
        if min_score > 0:
            hits = [h for h in hits if (h.get("score") or 0) >= min_score]
        return hits

    def count(self) -> int:
        return self._retriever.count()

    def documents(self) -> list[dict]:
        return self._store.documents()

    def ingest_documents(self, docs: list[dict]) -> int:
        return self._retriever.ingest_documents(docs, clear=False)

    def list_docs(self) -> list[dict]:
        """列出本库全部文档（doc_id + 来源文件名 + 文本预览 + 块数 + 摘要）。"""
        out: dict[str, dict] = {}
        for d in self._store.documents():
            meta = d.get("metadata") or {}
            doc_id = meta.get("doc_id") or d.get("id")
            source = meta.get("source") or meta.get("doc_id") or doc_id
            entry = out.setdefault(
                str(doc_id),
                {
                    "doc_id": str(doc_id),
                    "source": str(source),
                    "text": (d.get("text") or "")[:200],
                    "chunks": 0,
                    "summary": meta.get("summary") or "",
                    "metadata": {
                        key: meta[key] for key in (
                            "source_type", "peer_reviewed", "publication_year", "sample_size",
                            "dataset", "model_version", "retracted",
                        ) if key in meta
                    },
                },
            )
            if not entry["summary"]:
                entry["summary"] = meta.get("summary") or ""
            entry["chunks"] += 1  # 同 doc_id 的每个 chunk 计数
        return list(out.values())

    def delete_document(self, doc_id: str) -> bool:
        """按 doc_id 删除单个文档（含其全部分块），返回是否删除。"""
        deleted = self._retriever.delete_by_metadata("doc_id", doc_id)
        if deleted:
            from app.knowledge.versioning import version_store

            version_store.deactivate_index_document(self.username, self.kb_name, doc_id)
        return deleted

    def clear(self) -> None:
        self._store.clear()
        from app.knowledge.versioning import version_store

        version_store.bump_revision(self.username, self.kb_name)


def _fetch_all_collections() -> list[dict]:
    """从 ChromaDB 服务端拉取全部 collection（含 name + metadata + 文档数）。"""
    import httpx

    base = os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        return httpx.get(f"{base}/api/v1/collections", timeout=5).json()
    except Exception as e:  # noqa: BLE001
        logger.warning("列出知识库 collection 失败: %s", e)
        return []


def ensure_library(username: str, kb_name: str) -> str:
    """幂等创建命名库 collection，并把中文显示名写入 metadata（kb_display_name）。

    返回 collection 名。已存在则不动（保留原 metadata）。
    """
    import httpx

    collection = collection_name_for(username, kb_name)
    base = os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    for c in _fetch_all_collections():
        if c.get("name") == collection:
            return collection
    try:
        httpx.post(
            f"{base}/api/v1/collections",
            json={
                "name": collection,
                "metadata": {"hnsw:space": "cosine", "kb_display_name": kb_name},
            },
            timeout=10,
        ).raise_for_status()
        logger.info(f"已创建知识库 collection {collection}（显示名={kb_name}）")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"创建知识库 collection {collection} 失败: {e}")
    return collection


def list_user_kbs(username: str) -> list[dict]:
    """列出某用户的全部知识库（默认库 + 命名库），带真实文档数。

    返回: [{"name": 库名(None=默认库), "collection": 完整collection名, "documents": N, "is_default": bool}]
    命名库的中文显示名从 collection metadata（kb_display_name）读取；无 metadata 时
    兜底用 slug 反解（纯 ASCII 库名 slug=原名）。
    """
    out: dict[str, dict] = {}
    for c in _fetch_all_collections():
        cname = c.get("name", "")
        if not cname.startswith(_collection_prefix()):
            continue
        if username_from_collection(cname) != username:
            continue
        kb_name = kb_name_from_collection(cname)  # slug 或 None（默认库）
        if kb_name:
            meta = c.get("metadata") or {}
            kb_name = meta.get("kb_display_name") or kb_name  # 中文显示名优先
        try:
            n = UserKnowledgeBase(username, kb_name).count()
        except Exception:  # noqa: BLE001 服务异常时按 0 处理，不阻塞列表
            n = 0
        out[kb_name or ""] = {
            "name": kb_name,  # None=默认库
            "collection": cname,
            "documents": n,
            "is_default": kb_name is None,
        }
    return list(out.values())


def delete_user_kb(username: str, kb_name: str) -> bool:
    """删除某用户的指定知识库（含全部文档，collection 一并删除）。

    默认库（kb_name=None）也可删除——删除后为空库，重新上传即自动重建。
    """
    import httpx

    collection = collection_name_for(username, kb_name)
    base = os.environ.get("CHROMA_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        resp = httpx.delete(f"{base}/api/v1/collections/{collection}", timeout=10)
        if resp.status_code not in (200, 404):  # 404=已不存在，视为成功
            resp.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("删除知识库 %s/%s 失败: %s", username, kb_name, e)
        return False


def rename_user_kb(username: str, old_name: str, new_name: str) -> bool:
    """重命名知识库：把旧库全部 chunks 迁移到新 collection，再删除旧库。

    ChromaDB 0.4.x 不支持 collection 改名，故采用「迁移 + 删旧」方案。
    - 保留原文档的 doc_id / source / content_hash 元数据（引用溯源不受影响）；
    - 迁移失败时旧库保留，不产生半迁移状态（新库写入前先清空）。
    """
    old_kb = UserKnowledgeBase(username, old_name)
    chunks = old_kb.documents()
    if not chunks:
        # 空库：直接换名（创建新 collection，删旧）
        ensure_library(username, new_name)
        return delete_user_kb(username, old_name)

    # 先确保新 collection 存在且带中文显示名 metadata
    # （RestChromaVectorStore.clear() 重建时会丢 kb_display_name，故不能靠它建库）
    delete_user_kb(username, new_name)  # 清理同名残留（幂等，不存在即 404 视为成功）
    ensure_library(username, new_name)
    new_kb = UserKnowledgeBase(username, new_name)
    try:
        docs = []
        for c in chunks:
            meta = dict(c.get("metadata") or {})
            doc_id = meta.get("doc_id") or str(c.get("id", "")).split("#")[0]
            source = meta.get("source") or doc_id
            docs.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "text": c.get("text", ""),
                    "metadata": meta,
                }
            )
        # clear=False：collection 已由 ensure_library 建好（空），直接写入
        count = new_kb.ingest_documents(docs)
        if count <= 0:
            logger.warning("知识库 %s/%s 迁移后无内容，保留旧库", username, old_name)
            return False
        if not delete_user_kb(username, old_name):
            logger.warning("知识库 %s/%s 迁移成功但旧库删除失败", username, old_name)
            return False
        logger.info("知识库 %s/%s -> %s 迁移完成（%d chunks）", username, old_name, new_name, count)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("知识库 %s/%s 重命名失败: %s", username, old_name, e)
        return False


def list_user_collections() -> list[str]:
    """列出 ChromaDB 中全部用户知识库 collection 名（kb_*，兼容旧调用方）。"""
    return [c.get("name", "") for c in _fetch_all_collections() if c.get("name", "").startswith(_collection_prefix())]
