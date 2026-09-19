"""用户个人知识库路由：多库上传/查看/删除/检索/库管理（登录用户可用，仅限自己的库）。"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.auth import User, get_current_user
from app.core.exceptions import ForbiddenError
from app.knowledge.user_kb import (
    UserKnowledgeBase,
    collection_name_for,
    delete_user_kb,
    ensure_library,
    extract_text,
    is_image_file,
    list_user_kbs,
    normalize_kb_name,
    rename_user_kb,
)

router = APIRouter(prefix="/userkb", tags=["userkb"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 上传大小上限：10MB


async def check_upload_size(request: Request) -> None:
    """上传大小限制依赖：在 body 解析前按 Content-Length 拦截超大文件。

    必须放在依赖链（Depends）而不是路由函数体里——FastAPI 的 UploadFile
    参数会在函数体执行前读完整个 body，放在函数里就拦不住了。
    """
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise ForbiddenError(f"文件过大（> {MAX_UPLOAD_BYTES // 1024 // 1024}MB），请压缩后上传")


class DocItem(BaseModel):
    doc_id: str
    source: str
    text: str
    chunks: int = 0
    summary: str = ""
    metadata: dict = Field(default_factory=dict)


class DocListResponse(BaseModel):
    username: str
    kb_name: str | None = None
    total_chunks: int
    documents: list[DocItem]


class UploadResponse(BaseModel):
    username: str
    kb_name: str | None = None
    filename: str
    saved_to_kb: bool
    chunks: int
    duplicate: bool = False
    version_updated: bool = False
    version_id: str | None = None
    version_number: int | None = None
    kb_revision: int | None = None
    message: str


class SearchResponse(BaseModel):
    username: str
    kb_name: str | None = None
    hits: list[dict]


class LibraryItem(BaseModel):
    name: str | None  # None=默认库
    collection: str
    documents: int
    is_default: bool = False


class LibraryListResponse(BaseModel):
    username: str
    libraries: list[LibraryItem]


def _kb(user: User, kb_name: str | None = None) -> UserKnowledgeBase:
    if kb_name:
        kb_name = normalize_kb_name(kb_name)
    return UserKnowledgeBase(user.username, kb_name)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    save_to_kb: bool = Form(True, description="是否存入个人知识库；false 表示仅临时读取不存储"),
    kb_name: str | None = Form(None, description="目标库名；省略=默认库"),
    source_type: str | None = Form(None, description="来源类型：journal/conference/paper/review/blog 等"),
    peer_reviewed: bool | None = Form(None, description="是否经过同行评审"),
    publication_year: int | None = Form(None, ge=1900, le=2100, description="发表年份"),
    sample_size: int | None = Form(None, ge=1, description="研究样本量"),
    dataset: str | None = Form(None, description="数据集或实验对象"),
    model_version: str | None = Form(None, description="模型/方法版本"),
    user: User = Depends(get_current_user),
    _: None = Depends(check_upload_size),
) -> UploadResponse:
    """上传文件（.txt/.md/.pdf/图片）。save_to_kb=true 时存入当前用户的指定知识库（默认库）。

    图片（.jpg/.png/.webp 等）会先经视觉模型 OCR 提取文字，再把文字入库（图文混合检索）。
    """
    data = await file.read()
    filename = file.filename or "unnamed"
    if is_image_file(filename):
        from app.knowledge.user_kb import ocr_image_text

        text = await ocr_image_text(filename, data)
        if not text.strip():
            raise ForbiddenError(f"图片未提取到文字（OCR 失败或图内无文字）: {filename}")
    else:
        text = extract_text(filename, data)
        if not text.strip():
            raise ForbiddenError(f"不支持的文件类型或内容为空: {filename}（支持 .txt/.md/.pdf/图片）")

    if not save_to_kb:
        return UploadResponse(
            username=user.username,
            kb_name=kb_name,
            filename=filename,
            saved_to_kb=False,
            chunks=0,
            message=f"文件 {filename} 已读取（{len(text)} 字符），未存入知识库",
        )

    try:
        kb = _kb(user, kb_name)
        if kb_name:
            ensure_library(user.username, kb_name)  # 命名库先确保 collection + 中文显示名
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    evidence_metadata = {
        "source_type": source_type.strip().lower() if source_type else None,
        "peer_reviewed": peer_reviewed,
        "publication_year": publication_year,
        "sample_size": sample_size,
        "dataset": dataset,
        "model_version": model_version,
    }
    result = await kb.add_document(filename, text, evidence_metadata)
    # 知识库内容变更：失效该用户的 Redis 检索缓存，保证一致性（失败静默）
    try:
        from app.cache.retrieval_cache import invalidate_user_kb

        await invalidate_user_kb(user.username)
    except Exception:  # noqa: BLE001
        pass
    # 知识库内容变更：后台增量同步知识图谱（抽取该文档实体/关系，不阻塞上传响应）
    if result.get("chunks", 0) > 0 and not result.get("duplicate"):
        try:
            import asyncio

            from app.knowledge.kg import sync_document_to_kg

            asyncio.create_task(
                sync_document_to_kg(user.username, result.get("doc_id", ""), filename, text)
            )
        except Exception:  # noqa: BLE001
            pass
    lib_label = f"知识库「{kb_name}」" if kb_name else "个人知识库"
    if result["duplicate"]:
        return UploadResponse(
            username=user.username,
            kb_name=kb_name,
            filename=filename,
            saved_to_kb=True,
            chunks=0,
            duplicate=True,
            message=f"文件 {filename} 内容已存在，未重复入库（{lib_label}）",
        )
    return UploadResponse(
        username=user.username,
        kb_name=kb_name,
        filename=filename,
        saved_to_kb=True,
        chunks=result["chunks"],
        version_updated=result.get("version_updated", False),
        version_id=result.get("version_id"),
        version_number=result.get("version_number"),
        kb_revision=result.get("kb_revision"),
        message=(
            f"文件 {filename} 已更新为最新版本（{lib_label}，{result['chunks']} 个片段）"
            if result.get("version_updated")
            else f"文件 {filename} 已存入{lib_label}（{result['chunks']} 个片段）"
        ),
    )


@router.get("/versions")
async def list_document_versions(
    source: str = Query(..., description="文档文件名/source"),
    kb_name: str | None = None,
    user: User = Depends(get_current_user),
) -> dict:
    """查看文档版本状态、失败原因和当前知识库 revision。"""
    from app.knowledge.versioning import version_store

    return {
        "username": user.username,
        "kb_name": kb_name,
        "source": source,
        "kb_revision": version_store.revision(user.username, kb_name),
        "versions": version_store.versions(user.username, kb_name, source),
    }


@router.get("/documents", response_model=DocListResponse)
async def list_documents(
    kb_name: str | None = None,
    user: User = Depends(get_current_user),
) -> DocListResponse:
    """查看自己的指定知识库文档列表。"""
    try:
        kb = _kb(user, kb_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    return DocListResponse(
        username=user.username,
        kb_name=kb_name,
        total_chunks=kb.count(),
        documents=[DocItem(**d) for d in kb.list_docs()],
    )


@router.post("/search", response_model=SearchResponse)
async def search_kb(
    query: str = Form(...),
    top_k: int = Form(3),
    kb_name: str | None = Form(None, description="目标库名；省略=默认库"),
    user: User = Depends(get_current_user),
) -> SearchResponse:
    """在指定知识库中检索。"""
    try:
        kb = _kb(user, kb_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    hits = await kb.search(query, top_k=top_k)
    return SearchResponse(username=user.username, kb_name=kb_name, hits=hits)


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    kb_name: str | None = None,
    user: User = Depends(get_current_user),
):
    """删除指定知识库中的单个文档（含全部分块）。"""
    try:
        kb = _kb(user, kb_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    # 先查回 source（upload 时图谱以 filename 为来源），再删除文档
    src = doc_id
    for d in kb.list_docs():
        if str(d.get("doc_id")) == doc_id and d.get("source"):
            src = str(d["source"])
            break
    ok = kb.delete_document(doc_id)
    if not ok:
        raise ForbiddenError(f"删除失败或文档不存在: {doc_id}")
    try:
        from app.cache.retrieval_cache import invalidate_user_kb

        await invalidate_user_kb(user.username)
    except Exception:  # noqa: BLE001
        pass
    # 后台移除该文档在图谱中的实体/关系（共享实体保留）
    try:
        import asyncio

        from app.knowledge.kg import sync_document_to_kg

        asyncio.create_task(
            sync_document_to_kg(user.username, doc_id, src, "", remove=True)
        )
    except Exception:  # noqa: BLE001
        pass
    return {"username": user.username, "kb_name": kb_name, "deleted": doc_id, "total": kb.count()}


@router.delete("/clear")
async def clear_kb(
    kb_name: str | None = None,
    user: User = Depends(get_current_user),
):
    """清空指定知识库。"""
    try:
        kb = _kb(user, kb_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    kb.clear()
    try:
        from app.cache.retrieval_cache import invalidate_user_kb

        await invalidate_user_kb(user.username)
    except Exception:  # noqa: BLE001
        pass
    return {"username": user.username, "kb_name": kb_name, "cleared": True}


# ---------- 知识库管理（多库隔离） ----------

@router.get("/libraries", response_model=LibraryListResponse)
async def list_libraries(user: User = Depends(get_current_user)) -> LibraryListResponse:
    """列出我的全部知识库（默认库 + 命名库），带文档数。"""
    return LibraryListResponse(username=user.username, libraries=list_user_kbs(user.username))


@router.post("/libraries", response_model=LibraryItem)
async def create_library(
    name: str = Body(..., embed=True, description="新库名（中文/字母/数字/下划线/短横线，1-30 字符）"),
    user: User = Depends(get_current_user),
) -> LibraryItem:
    """创建命名知识库。collection 立即创建（幂等：首次上传时复用）。"""
    try:
        kb_name = normalize_kb_name(name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    # 名称冲突检查（同名库已存在则报错）
    for lib in list_user_kbs(user.username):
        if lib["name"] == kb_name:
            raise ForbiddenError(f"知识库「{kb_name}」已存在")
    # 创建 collection（带中文显示名 metadata）
    collection = ensure_library(user.username, kb_name)
    return LibraryItem(
        name=kb_name,
        collection=collection,
        documents=0,
        is_default=False,
    )


@router.delete("/libraries/{kb_name}")
async def delete_library(kb_name: str, user: User = Depends(get_current_user)):
    """删除命名知识库（含全部文档）。默认库删除=清空，重新上传即重建。"""
    try:
        kb_name = normalize_kb_name(kb_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    ok = delete_user_kb(user.username, kb_name)
    if not ok:
        raise ForbiddenError(f"删除知识库「{kb_name}」失败")
    return {"username": user.username, "deleted": kb_name}


class RenameRequest(BaseModel):
    new_name: str


@router.post("/libraries/{kb_name}/rename", response_model=LibraryItem)
async def rename_library(
    kb_name: str,
    req: RenameRequest,
    user: User = Depends(get_current_user),
) -> LibraryItem:
    """重命名知识库（数据迁移到新 collection，引用溯源不受影响）。"""
    try:
        old_name = normalize_kb_name(kb_name)
        new_name = normalize_kb_name(req.new_name)
    except ValueError as e:
        raise ForbiddenError(str(e)) from e
    if old_name == new_name:
        raise ForbiddenError("新旧库名相同")
    # 名称冲突检查
    for lib in list_user_kbs(user.username):
        if lib["name"] == new_name:
            raise ForbiddenError(f"知识库「{new_name}」已存在")
    ok = rename_user_kb(user.username, old_name, new_name)
    if not ok:
        raise ForbiddenError(f"重命名知识库「{old_name}」失败")
    return LibraryItem(
        name=new_name,
        collection=collection_name_for(user.username, new_name),
        documents=0,
        is_default=False,
    )
