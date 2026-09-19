"""判断恢复路径耗尽后应向用户索取哪类信息。"""
from __future__ import annotations


DOCUMENT_MARKERS = ("这篇", "该论文", "上传", "pdf", "文档", "文件", "附件", "全文")


def build_user_input_request(
    *, goal: str, step: str, failed_tool: str, available_tools: set[str]
) -> dict:
    text = f"{goal} {step}".lower()
    if any(marker in text for marker in DOCUMENT_MARKERS):
        return {
            "reason": "required_document_missing",
            "required_input": "document_or_reference",
            "message": "当前没有取得完成该步骤所需的文档。请上传相关 PDF/文件，或提供论文标题、链接及关键正文。",
        }
    if "web_search" not in available_tools:
        return {
            "reason": "web_search_disabled",
            "required_input": "enable_web_or_reference",
            "message": "当前知识库和论文检索未取得足够资料，且联网检索未启用。请允许联网检索，或提供相关资料、标题或链接。",
        }
    return {
        "reason": "insufficient_evidence",
        "required_input": "clarification_or_reference",
        "message": (
            f"已尝试可用检索路径，但仍缺少完成“{step}”的可靠证据。"
            "请补充更具体的关键词、资料范围、论文标题或相关文件。"
        ),
        "failed_tool": failed_tool,
    }

