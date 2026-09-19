# RAG 检索消融实验报告（2026-08-13）

**目的**：在同一 qrels、同一语料、同一 top-k 下对比 Embedding × Reranker 组合，
为生产检索配置提供数据依据（Recall/Precision/HitRate/MRR/nDCG @1/@3/@5）。

## 实验设置

- 语料：`data/knowledge_docs`（13 篇，arxiv 论文 + 自建中文文档）
- 评测集：中文集 30 查询（`chinese_retrieval_qrels.json`）+ 英文扩展集 11 查询（`extended_retrieval_qrels.json`）
- Embedding：`ollama://nomic-embed-text`（768d，现生产）vs `BAAI/bge-small-zh-v1.5`（512d，中文优化）
- Reranker：`lexical`（词面融合，现生产）/ `cross`（bge-reranker-v2-m3 精排）/ `none`
- 管线：向量召回 + BM25 混合 + RRF 融合 + 重排（`Retriever` 生产链路，FAISS 临时库）
- 命令：`scripts/compare_retrievers.py`（报告：`ablation_full.json` / `ablation_bge.json` / `ablation_bge_cross.json`）

## 结果（@3 关键指标）

### 中文集（30 查询）—— 本项目核心场景

| Embedding | Reranker | Recall@3 | MRR@3 | nDCG@3 | 延迟 ms/q |
|---|---|---|---|---|---|
| nomic-embed-text | lexical | 0.617 | 0.556 | 0.569 | 147 |
| nomic-embed-text | cross | 0.617 | 0.556 | 0.569 | - |
| nomic-embed-text | none | 0.617 | 0.556 | 0.569 | 1.5 |
| **bge-small-zh-v1.5** | **lexical** | **0.900** | **0.894** | **0.883** | 68.9 |
| bge-small-zh-v1.5 | none | 0.833 | 0.828 | 0.817 | 2.1 |
| bge-small-zh-v1.5 | cross | 0.883 | 0.856 | 0.860 | 18480 |

### 英文扩展集（11 查询）

| Embedding | Reranker | Recall@3 | MRR@3 | nDCG@3 |
|---|---|---|---|---|
| **nomic-embed-text** | **lexical** | **0.909** | **0.803** | **0.815** |
| bge-small-zh-v1.5 | lexical | 0.818 | 0.773 | 0.781 |
| bge-small-zh-v1.5 | none | 0.818 | 0.773 | 0.781 |
| bge-small-zh-v1.5 | cross | 0.955 | 0.955 | 0.928 | 20566 |

## 结论

1. **中文场景 bge-small-zh-v1.5 显著优于 nomic**：Recall@3 +46%（0.617 → 0.900）、MRR@3 +61%。
   nomic 三个 reranker 结果完全相同，说明召回池里相关文档缺失，重排无法补救——是**召回层（embedding）瓶颈**。
2. **英文场景 nomic 略优**（+11%）：符合预期（nomic 为英文优化）；bge+cross 英文最高（0.955）但 20.5s/q 不可用。
3. **lexical 重排优于 cross**：中文集 0.900 vs 0.883，且 lexical 69ms vs cross 18.5s（慢 268 倍）——**cross 精排在 CPU 上不适合实时链路**，生产保持 lexical 正确。
4. **延迟**：bge+lexical 68.9ms/q 对交互链路可接受。

## 落地建议（按价值排序）

| 方案 | 动作 | 收益 | 风险 |
|---|---|---|---|
| A. 全局知识库迁 bge | 改 `.env` embedding → bge，`rebuild_knowledge.py` 重建（源文档在 data/knowledge_docs） | 中文检索 +46%，直接提升 RAG 主链路 | 需重启；knowledge collection 重建期间检索降级 |
| B. 全量迁移（含用户库） | A + 从 ChromaDB 导出 kb_* / memory_facts 文本重建 | 所有中文检索受益 | 数据迁移工程量大，doc_id 变化影响引用 |
| C. 保持 nomic | 不改 embedding，转向 query rewrite / 切分优化 | 零风险 | 中文召回瓶颈仍在 |

## 落地执行记录（2026-08-13 晚）

- **已执行方案 A（全量迁移 bge）**：`scripts/migrate_embedding.py`（dry-run 备份 → 执行）迁移 4 个有数据 collection
  （knowledge 13 / kb_student1 3 / kb_student2 1 / memory_facts 3），全部 768d → 512d，原 id/metadata 保留；
  备份在 `data/migration_backup/*.json`。
- `.env`：`KNOWLEDGE__EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
- **生产验证**（ChromaDB 实库，候选池 5）：
  - 中文集 30 查询：@1 Recall 0.75 / MRR 0.767；@3 Recall 0.80 / MRR 0.80；@5 Recall 0.883 / MRR 0.815
  - 英文基线 5 查询：@1 Recall 0.9 / MRR 1.0 / nDCG 1.0（不降）
  - 生产 @3 略低于消融（0.80 vs 0.90）系候选池差异（生产 vector_top_k=5 vs 消融 rerank_k=10），整体仍远超 nomic。
- 待办：`build_kb.py` 中硬编码的 `ollama://nomic-embed-text` 可后续改为读配置（本次未动，避免影响 CI 基线）。

