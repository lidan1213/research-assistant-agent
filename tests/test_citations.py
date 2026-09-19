"""引用覆盖率与 Trace ID 单元测试。"""
from app.eval_citations import citation_coverage


def test_citation_coverage_counts_valid_refs():
    result = citation_coverage("量子点具有可调发光特性。[1] 其原因与量子限域有关。[2]", 2)
    assert result["citation_count"] == 2
    assert result["valid_citation_count"] == 2
    assert result["validity"] == 1.0
    assert result["citation_coverage"] > 0


def test_citation_coverage_marks_invalid_refs():
    result = citation_coverage("这是一个较长的事实陈述。[4]", 2)
    assert result["valid_citation_count"] == 0
    assert result["validity"] == 0.0


def test_citation_coverage_no_ref_is_safe():
    result = citation_coverage("没有引用的回答", 3)
    assert result["citation_count"] == 0
    assert result["citation_coverage"] == 0.0
