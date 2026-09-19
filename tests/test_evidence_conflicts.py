import asyncio

from app.eval_citations import citation_support
from app.rag.evidence import analyze_evidence, annotate_evidence, extract_claim


def test_numeric_conflict_across_sources_is_flagged():
    hits = [
        {"source": "paper-a", "text": "钙钛矿电池效率达到 25.7%。"},
        {"source": "paper-b", "text": "钙钛矿电池效率达到 23.1%。"},
    ]
    report = annotate_evidence(hits, "钙钛矿电池效率")
    assert report["has_conflict"] is True
    assert report["policy"] == "disclose_both_sides"
    assert all(hit["evidence_conflict"] for hit in hits)


def test_same_source_chunks_are_not_independent_conflict():
    hits = [
        {"source": "paper-a", "text": "材料效率为 25.7%。"},
        {"source": "paper-a", "text": "材料效率在另一实验为 23.1%。"},
    ]
    assert analyze_evidence(hits, "材料效率")["has_conflict"] is False


def test_polarity_conflict_is_flagged():
    hits = [
        {"source": "a", "text": "该方法显著提高模型准确率和稳定性。"},
        {"source": "b", "text": "该方法未显著提高模型准确率和稳定性。"},
    ]
    assert analyze_evidence(hits, "模型准确率稳定性")["has_conflict"] is True


def test_chinese_need_vs_no_need_conflict_is_flagged():
    hits = [
        {"source": "a", "text": "联合索引覆盖查询时无须回表，可以直接读取索引。"},
        {"source": "b", "text": "普通联合索引不能覆盖查询，仍然必须回表。"},
    ]
    report = analyze_evidence(hits, "联合索引覆盖查询是否回表")
    assert report["has_conflict"] is True
    assert any("相反的技术论断" in reason for reason in report["conflicts"][0]["reasons"])


def test_different_datasets_are_contextual_difference_not_direct_conflict():
    hits = [
        {"source": "a", "text": "在 MS MARCO 数据集上，该方法显著提高准确率至 87%。"},
        {"source": "b", "text": "在 CMedQA 数据集上，该方法未提高准确率，仅为 72%。"},
    ]
    report = analyze_evidence(hits, "该方法准确率")
    assert report["has_conflict"] is False
    assert report["has_contextual_difference"] is True
    assert report["policy"] == "explain_condition_differences"
    assert any("数据集不同" in x for x in report["contextual_differences"][0]["condition_differences"])


def test_different_metrics_do_not_create_numeric_conflict():
    hits = [
        {"source": "a", "text": "该模型准确率为 80%。"},
        {"source": "b", "text": "该模型召回率为 90%。"},
    ]
    report = analyze_evidence(hits, "模型效果")
    assert report["has_conflict"] is False


def test_quality_metadata_allows_only_provisional_preference():
    hits = [
        {"source": "paper", "text": "该方法显著提高准确率。",
         "metadata": {"source_type": "journal", "peer_reviewed": True}},
        {"source": "blog", "text": "该方法未提高准确率。",
         "metadata": {"source_type": "blog", "retracted": True}},
    ]
    report = analyze_evidence(hits, "方法准确率")
    group = report["conflicts"][0]
    assert group["status"] == "provisional_preference"
    assert group["preferred_member"] == 1
    assert report["policy"] == "provisional_adjudication"


def test_claim_extraction_keeps_auditable_conditions():
    claim = extract_claim(
        {"source": "paper", "text": "2025 年在 MS MARCO 数据集上，样本量为 5000，模型 v2.1 准确率为 87%。"},
        "准确率",
    )
    assert claim["metric"] == "准确率"
    assert claim["conditions"] == {
        "dataset": "MS MARCO",
        "sample_size": 5000,
        "year": 2025,
        "version": "v2.1",
    }
    assert claim["quality"]["score"] > 0.5


def test_knowledge_search_exposes_contextual_difference_to_agent(monkeypatch):
    import importlib

    module = importlib.import_module("app.tools.knowledge_search")

    class StubRetriever:
        async def retrieve(self, query, top_k=3):
            return [
                {"text": "在 MS MARCO 数据集上，RAG 显著提高准确率至 87%。",
                 "score": 0.9, "metadata": {"source": "a.pdf"}},
                {"text": "在 CMedQA 数据集上，RAG 未提高准确率，仅为 72%。",
                 "score": 0.8, "metadata": {"source": "b.pdf"}},
            ]

    monkeypatch.setattr(module, "_global_retriever", StubRetriever())
    result = asyncio.run(module.knowledge_search.run(query="RAG准确率", top_k=3))
    assert result.success is True
    assert "[EVIDENCE_CONTEXT_DIFFERENCE]" in result.output
    assert "数据集不同" in result.output
    assert "证据质量:" in result.output


def test_citation_support_checks_claim_against_selected_source():
    report = citation_support(
        "钙钛矿效率达到 25.7% [1]。离子迁移造成稳定性问题 [2]。",
        ["钙钛矿电池效率达到 25.7%", "稳定性问题源于离子迁移"],
        threshold=0.3,
    )
    assert report["score"] == 1.0


def test_citation_support_rejects_unrelated_source():
    report = citation_support("钙钛矿效率达到 25.7% [1]。", ["机器学习训练数据"], threshold=0.3)
    assert report["score"] == 0.0
