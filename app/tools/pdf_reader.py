"""PDF 阅读工具：从本地路径或 URL 提取正文文本（用于文献精读）。

特性：
- 优先使用现代维护的 `pypdf`，回退到 `PyPDF2`；
- 支持本地路径（`path`）与 HTTP(S) 链接（`url`，例如 arXiv 的 pdf_url）两种来源；
- 处理加密/带密码的 PDF（尝试用空密码解密）；
- 返回结构化结果：来源、页数、是否截断、正文文本，便于 Agent 消费。
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

from app.tools.base import ToolResult, tool


class PdfArgs(BaseModel):
    path: str | None = Field(None, description="本地 PDF 路径（与 url 二选一）")
    url: str | None = Field(None, description="PDF 的 HTTP(S) 链接，例如 arXiv 的 pdf_url")
    max_chars: int = Field(6000, ge=100, le=40000, description="返回正文的最大字符数")
    download_dir: str | None = Field(
        None, description="若提供且传入 url，则把 PDF 缓存到此目录（否则用临时文件）"
    )

    @model_validator(mode="after")
    def _need_one_source(self) -> "PdfArgs":
        if not self.path and not self.url:
            raise ValueError("必须提供 path 或 url 之一")
        return self


def _load_pdf_lib():
    """优先 pypdf，回退 PyPDF2；返回 (名称, 模块)。"""
    try:
        import pypdf  # type: ignore

        return "pypdf", pypdf
    except ImportError:
        import PyPDF2  # type: ignore

        return "PyPDF2", PyPDF2


def _resolve_source(args: PdfArgs) -> tuple[str, bytes | None, str]:
    """返回 (来源描述, 已下载字节 or None, 本地可读路径)。"""
    if args.url:
        dl_dir = args.download_dir
        if dl_dir:
            os.makedirs(dl_dir, exist_ok=True)
            local = os.path.join(dl_dir, os.path.basename(args.url.split("?")[0]) or "paper.pdf")
            # 已经缓存过就直接复用
            if os.path.isfile(local):
                return local, None, local
            target = local
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            target = tmp.name
            tmp.close()
        try:
            with httpx.stream("GET", args.url, timeout=30, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:  # noqa: BLE001
            return args.url, None, ""
        return args.url, None, target
    return args.path, None, args.path


@tool(name="pdf_reader", description="读取本地或远程 PDF 并提取正文文本（用于文献精读）", params=PdfArgs)
async def pdf_reader(
    path: str | None = None,
    url: str | None = None,
    max_chars: int = 6000,
    download_dir: str | None = None,
) -> dict[str, Any] | ToolResult:
    args = PdfArgs(path=path, url=url, max_chars=max_chars, download_dir=download_dir)
    source, _raw, local = _resolve_source(args)
    if not local or not os.path.isfile(local):
        return ToolResult(success=False, error=f"无法获取 PDF：{source}")

    try:
        lib_name, pdf_lib = _load_pdf_lib()
        reader = pdf_lib.PdfReader(local)

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return ToolResult(success=False, error="PDF 已加密且无法用空密码解密")

        num_pages = len(reader.pages)
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                parts.append("")
        text = "\n\n".join(p.strip() for p in parts if p.strip())

        truncated = len(text) > max_chars
        return {
            "source": source,
            "backend": lib_name,
            "num_pages": num_pages,
            "truncated": truncated,
            "text": text[:max_chars],
        }
    except Exception as e:  # noqa: BLE001
        return ToolResult(success=False, error=f"PDF 解析失败: {e}")
    finally:
        # 用临时文件时才清理
        if local and args.url and not args.download_dir and os.path.isfile(local):
            try:
                os.remove(local)
            except OSError:
                pass
