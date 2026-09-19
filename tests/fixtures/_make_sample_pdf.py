"""生成测试用 PDF 夹具（仅本地一次性脚本，不进入运行时依赖）。

用法：python tests/fixtures/_make_sample_pdf.py
产物：tests/fixtures/sample_paper.pdf  （多页、含标题与正文，用于 pdf_reader 离线测试）
"""
from pathlib import Path

from fpdf import FPDF

TEXT = [
    ("Research Assistant Agent: A Survey", 16, True),
    ("Abstract", 12, True),
    (
        "Large language models have enabled autonomous research assistants that can "
        "plan, retrieve and synthesize scientific literature. This paper surveys the "
        "core components: retrieval-augmented generation, tool use, and multi-agent "
        "coordination.",
        11,
        False,
    ),
    ("1. Retrieval", 12, True),
    (
        "Retrieval-augmented generation (RAG) couples a vector index with a generative "
        "model. Hybrid retrieval combines dense and sparse signals via reciprocal rank "
        "fusion to improve recall on scientific corpora.",
        11,
        False,
    ),
    ("2. Multi-Agent Coordination", 12, True),
    (
        "A supervisor delegates sub-tasks to specialist agents. A shared blackboard lets "
        "experts exchange intermediate results, while chain passing forwards a primary "
        "product from one agent to the next.",
        11,
        False,
    ),
    ("3. Long-term Memory", 12, True),
    (
        "Conversation memory is persisted to SQLite so that findings survive restarts "
        "and can be recalled across sessions with full-text search.",
        11,
        False,
    ),
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    out = Path(__file__).resolve().parent / "sample_paper.pdf"
    for text, size, bold in TEXT:
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(0, 6, text)
        pdf.ln(1)
    pdf.output(str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
