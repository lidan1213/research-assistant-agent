"""知识图谱（KG）模块：实体/关系抽取、SQLite 存储与查询。

设计：
- `KGStore`：每个用户独立 SQLite 库（data/kg_{username}.db），entities/relations 两张表，
  按 (name, type) 唯一去重实体；关系按 (source_id, target_id, relation) 去重。
- `extract_graph()`：用 LLM（json_mode）从文档文本抽取实体与三元组，
  输出 {"entities": [{"name","type","description"}], "relations": [{"source","relation","target"}]}。
- `build_from_docs()`：从用户知识库文档批量构建图谱（逐篇抽取，单篇失败不中断）。
- 查询能力：实体搜索、实体详情（相邻关系）、全图数据（供前端 SVG 可视化）。

用法：
    store = KGStore("student1")
    await store.build_from_docs(docs, llm=llm)     # docs: [{doc_id, source, text}]
    node = store.query_entity("钙钛矿")
    graph = store.graph_data()                      # {"nodes": [...], "edges": [...]}
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger("knowledge.kg")

# 实体类型白名单（中文科研场景），抽取时 LLM 从这些类型中选择
ENTITY_TYPES = [
    "方法", "技术", "材料", "模型", "数据集", "指标", "工具",
    "概念", "人物", "机构", "论文", "任务", "应用", "其他",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT '概念',
    description TEXT DEFAULT '',
    doc_source TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type ON entities(name, type);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    doc_source TEXT DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY(source_id) REFERENCES entities(id),
    FOREIGN KEY(target_id) REFERENCES entities(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_triple
    ON relations(source_id, target_id, relation);
"""

_EXTRACT_PROMPT = """你是科研知识抽取引擎。从下面的文档中抽取【重要】实体及其关系，输出 JSON。

规则：
1. entities: 抽取与主题相关的关键实体（方法、技术、材料、模型、数据集、指标、概念、人物、机构等）。
   - name: 实体名称（保持原文用词，中文文档用中文）
   - type: 从 [方法, 技术, 材料, 模型, 数据集, 指标, 工具, 概念, 人物, 机构, 论文, 任务, 应用, 其他] 中选择
   - description: 一句话说明（≤60字）
2. relations: 抽取实体间的明确关系（三元组），只保留文本中有依据的关系。
   - source: 主语实体名（必须出现在 entities 中）
   - relation: 关系动词短语，如 "提出" "用于" "优于" "属于" "依赖" "包含"
   - target: 宾语实体名（必须出现在 entities 中）
3. 数量：entities 5~15 个，relations 3~12 条；宁缺毋滥，不编造文档中没有的信息。
4. 输出必须是合法 JSON，不要输出任何其他文字：
{"entities":[{"name":"","type":"","description":""}],"relations":[{"source":"","relation":"","target":""}]}

文档内容：
"""


class KGStore:
    """用户级知识图谱存储（SQLite）。"""

    def __init__(self, username: str, data_dir: str | Path = "./data") -> None:
        safe = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "_", username or "default")
        self.username = safe
        self.path = Path(data_dir) / f"kg_{safe}.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---------- 实体 CRUD ----------
    def upsert_entity(
        self, name: str, type_: str, description: str = "", doc_source: str = ""
    ) -> int:
        """按 (name, type) 去重写入实体，返回 entity id。"""
        name = (name or "").strip()
        if not name or len(name) > 100:
            return 0
        type_ = type_.strip() or "概念"
        if type_ not in ENTITY_TYPES:
            type_ = "其他"
        now = time.time()
        cur = self._conn.execute(
            "SELECT id FROM entities WHERE name=? AND type=?",
            (name, type_),
        )
        row = cur.fetchone()
        if row:
            # 已有实体：补充描述（若为空）
            if description:
                self._conn.execute(
                    "UPDATE entities SET description=?, doc_source=? WHERE id=?",
                    (description[:200], (doc_source or "")[:200], row["id"]),
                )
            return int(row["id"])
        cur = self._conn.execute(
            "INSERT INTO entities(name, type, description, doc_source, created_at) VALUES(?,?,?,?,?)",
            (name, type_, (description or "")[:200], (doc_source or "")[:200], now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_relation(
        self, source_id: int, target_id: int, relation: str, doc_source: str = ""
    ) -> int | None:
        """按 (source_id, target_id, relation) 去重写入关系。"""
        relation = (relation or "").strip()
        if not relation or source_id <= 0 or target_id <= 0 or source_id == target_id:
            return None
        cur = self._conn.execute(
            "SELECT id FROM relations WHERE source_id=? AND target_id=? AND relation=?",
            (source_id, target_id, relation),
        )
        if cur.fetchone():
            return None
        cur = self._conn.execute(
            "INSERT INTO relations(source_id, target_id, relation, doc_source, created_at) VALUES(?,?,?,?,?)",
            (source_id, target_id, relation[:100], (doc_source or "")[:200], time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_triple(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        source_type: str = "概念",
        target_type: str = "概念",
        description: str = "",
        doc_source: str = "",
    ) -> int | None:
        """按三元组文本一次性写入（自动 upsert 两端实体）。"""
        s_id = self.upsert_entity(source, source_type, description, doc_source)
        if not s_id:
            return None
        t_id = self.upsert_entity(target, target_type, "", doc_source)
        if not t_id:
            return None
        return self.add_relation(s_id, t_id, relation, doc_source)

    # ---------- 查询 ----------
    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """按名称模糊搜索实体。"""
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT id, name, type, description, doc_source FROM entities "
            "WHERE name LIKE ? ORDER BY length(name) LIMIT ?",
            (like, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_entities(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, type, description, doc_source FROM entities "
            "ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def query_entity(self, name: str) -> dict | None:
        """查询实体详情 + 相邻关系（一跳邻居）。"""
        row = self._conn.execute(
            "SELECT id, name, type, description, doc_source FROM entities WHERE name=? "
            "ORDER BY id LIMIT 1",
            (name,),
        ).fetchone()
        if not row:
            return None
        node = dict(row)
        edges = self._conn.execute(
            "SELECT r.relation, e_out.name AS source, e_in.name AS target "
            "FROM relations r "
            "JOIN entities e_out ON e_out.id = r.source_id "
            "JOIN entities e_in ON e_in.id = r.target_id "
            "WHERE r.source_id=? OR r.target_id=?",
            (node["id"], node["id"]),
        ).fetchall()
        node["relations"] = [dict(e) for e in edges]
        return node

    def graph_data(self, limit_entities: int = 300) -> dict:
        """全图数据（节点 + 边），供前端可视化。"""
        nodes = [
            dict(r)
            for r in self._conn.execute(
                "SELECT id, name, type, description FROM entities ORDER BY id LIMIT ?",
                (limit_entities,),
            ).fetchall()
        ]
        edges = [
            dict(r)
            for r in self._conn.execute(
                "SELECT r.id, r.source_id, r.target_id, r.relation, "
                "e1.name AS source, e2.name AS target "
                "FROM relations r "
                "JOIN entities e1 ON e1.id = r.source_id "
                "JOIN entities e2 ON e2.id = r.target_id "
                "ORDER BY r.id"
            ).fetchall()
        ]
        return {"username": self.username, "nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        n_entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        n_relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        type_dist = {
            r["type"]: r["c"]
            for r in self._conn.execute(
                "SELECT type, COUNT(*) AS c FROM entities GROUP BY type ORDER BY c DESC"
            ).fetchall()
        }
        return {
            "username": self.username,
            "entities": n_entities,
            "relations": n_relations,
            "type_distribution": type_dist,
        }

    def clear(self) -> None:
        self._conn.execute("DELETE FROM relations")
        self._conn.execute("DELETE FROM entities")
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


# ---------- LLM 抽取 ----------
def _parse_extraction(text: str) -> dict:
    """从 LLM 输出中提取 JSON（容忍 ```json 围栏与前后杂讯）。"""
    if not text:
        return {"entities": [], "relations": []}
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    # 截取第一个 { 到最后一个 }
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # 最后一搏：按行解析 key: value
        return {"entities": [], "relations": []}
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    return {"entities": entities, "relations": relations}


async def extract_graph(text: str, llm: Any, max_chars: int = 6000) -> dict:
    """用 LLM 从文本抽取实体与关系（json_mode）。

    llm 为 None 时通过 LLMGateway 路由（ENTITY_EXTRACTION → 辅助模型）；
    失败时返回空结构，不向上抛异常（批量构建容错）。
    """
    if not text or not text.strip():
        return {"entities": [], "relations": []}
    from app.llm.base import ChatMessage, MessageRole

    truncated = text[:max_chars]
    try:
        if llm is None:
            from app.llm.gateway import get_llm_gateway
            from app.llm.router import TaskType

            resp = await get_llm_gateway().chat(
                [
                    ChatMessage(role=MessageRole.SYSTEM, content=_EXTRACT_PROMPT),
                    ChatMessage(role=MessageRole.USER, content=truncated),
                ],
                temperature=0.1,
                json_mode=True,
                task_type=TaskType.ENTITY_EXTRACTION,
            )
        else:
            resp = await llm.chat(
                [
                    ChatMessage(role=MessageRole.SYSTEM, content=_EXTRACT_PROMPT),
                    ChatMessage(role=MessageRole.USER, content=truncated),
                ],
                temperature=0.1,
                json_mode=True,
            )
        content = (resp.content or "").strip()
        return _parse_extraction(content)
    except Exception as e:  # noqa: BLE001
        logger.warning("KG 抽取失败（跳过该文档）: %s", e)
        return {"entities": [], "relations": []}


async def build_from_docs(store: KGStore, docs: list[dict], llm: Any) -> dict:
    """从文档列表批量构建图谱。

    docs: [{"doc_id", "source", "text"}]；单篇失败自动跳过，统计成功数。
    """
    stats = {"docs": 0, "entities": 0, "relations": 0, "failed": 0}
    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        extracted = await extract_graph(text, llm)
        ents = extracted.get("entities") or []
        rels = extracted.get("relations") or []
        if not ents and not rels:
            stats["failed"] += 1
            continue
        doc_source = d.get("source") or d.get("doc_id") or ""
        # 实体名 -> id 映射（同篇文档内引用）
        id_by_name: dict[str, int] = {}
        for e in ents:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            eid = store.upsert_entity(
                name, e.get("type", ""), e.get("description", ""), doc_source
            )
            if eid:
                id_by_name[name] = eid
                stats["entities"] += 1
        for r in rels:
            s = (r.get("source") or "").strip()
            t = (r.get("target") or "").strip()
            rel = (r.get("relation") or "").strip()
            if not s or not t or not rel:
                continue
            # 关系两端的实体可能不在 entities 列表（LLM 偶发），自动补建
            s_id = id_by_name.get(s) or store.upsert_entity(s, "概念", "", doc_source)
            t_id = id_by_name.get(t) or store.upsert_entity(t, "概念", "", doc_source)
            if s_id and t_id and store.add_relation(s_id, t_id, rel, doc_source):
                stats["relations"] += 1
        stats["docs"] += 1
    return stats


# ---------- 便捷入口 ----------
def get_kg_store(username: str, data_dir: str | Path = "./data") -> KGStore:
    """按用户名获取（并缓存）KGStore。"""
    import os

    env_dir = os.environ.get("KG_DATA_DIR")
    if env_dir:
        data_dir = env_dir
    if str(data_dir) == "./data":
        from app.config import get_settings

        data_dir = str(Path(get_settings().knowledge.knowledge_dir).parent)
    return KGStore(username, data_dir)


async def sync_document_to_kg(
    username: str,
    doc_id: str,
    source: str,
    text: str,
    llm: Any = None,
    *,
    remove: bool = False,
) -> dict:
    """知识库文档变更时的图谱增量同步（上传/删除调用）。

    - remove=True：删除该文档来源的全部实体与关系（doc_source 匹配）；
    - remove=False：抽取该文档的实体/关系并入库（同名实体去重合并）。

    异步后台执行，失败不影响知识库主流程。
    """
    try:
        store = get_kg_store(username)
        if remove:
            # 关系按文档来源删除；实体只删「该文档独占」的（共享实体保留，避免误删）
            deleted_r = store._conn.execute(
                "DELETE FROM relations WHERE doc_source=?", (source,)
            ).rowcount
            shared = {
                r["name"]
                for r in store._conn.execute(
                    "SELECT DISTINCT e.name FROM entities e "
                    "JOIN relations r ON r.source_id=e.id OR r.target_id=e.id"
                ).fetchall()
            }
            rows = store._conn.execute(
                "SELECT id, name FROM entities WHERE doc_source=?", (source,)
            ).fetchall()
            deleted_e = 0
            for row in rows:
                if row["name"] not in shared:
                    store._conn.execute("DELETE FROM entities WHERE id=?", (row["id"],))
                    deleted_e += 1
            store._conn.commit()
            return {"action": "remove", "entities": 0, "relations": 0,
                    "removed_entities": deleted_e, "removed_relations": deleted_r}
        if not text or not text.strip():
            return {"action": "skip", "reason": "empty"}
        # llm 为 None 时 extract_graph 内部走 LLMGateway（ENTITY_EXTRACTION → 辅助模型）
        extracted = await extract_graph(text, llm)
        ents = extracted.get("entities") or []
        rels = extracted.get("relations") or []
        id_by_name: dict[str, int] = {}
        n_e = n_r = 0
        for e in ents:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            eid = store.upsert_entity(name, e.get("type", ""), e.get("description", ""), source)
            if eid:
                id_by_name[name] = eid
                n_e += 1
        for r in rels:
            s, t, rel = (r.get("source") or "").strip(), (r.get("target") or "").strip(), (r.get("relation") or "").strip()
            if not s or not t or not rel:
                continue
            s_id = id_by_name.get(s) or store.upsert_entity(s, "概念", "", source)
            t_id = id_by_name.get(t) or store.upsert_entity(t, "概念", "", source)
            if s_id and t_id and store.add_relation(s_id, t_id, rel, source):
                n_r += 1
        return {"action": "upsert", "entities": n_e, "relations": n_r}
    except Exception as e:  # noqa: BLE001
        logger.warning("KG 增量同步失败 doc=%s: %s", doc_id, e)
        return {"action": "failed", "error": str(e)[:200]}
