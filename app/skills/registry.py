"""Discover, validate, match, and enable local ``SKILL.md`` bundles."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger("skills")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [part.strip().strip("'\"") for part in value[1:-1].split(",") if part.strip()]
    return value.strip("'\"")


def _frontmatter(text: str) -> tuple[dict, str]:
    """Parse the conservative YAML subset used by local Skills, without extra deps."""
    normalized = text.lstrip("\ufeff")
    if not normalized.startswith("---\n"):
        raise ValueError("SKILL.md 必须以 YAML front matter（---）开头")
    end = normalized.find("\n---", 4)
    if end < 0:
        raise ValueError("SKILL.md 缺少 front matter 结束标记")
    header, body = normalized[4:end], normalized[end + 4 :].strip()
    meta: dict = {}
    current_list: str | None = None
    for raw in header.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.lstrip().startswith("-") and current_list:
            meta[current_list].append(_scalar(raw.lstrip()[1:].strip()))
            continue
        if ":" not in raw:
            raise ValueError(f"无法解析 front matter 行: {raw}")
        key, value = raw.split(":", 1)
        key = key.strip()
        parsed = _scalar(value)
        meta[key] = parsed if parsed != "" else []
        current_list = key if isinstance(meta[key], list) and parsed == "" else None
    return meta, body


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    triggers: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    preferred_mode: str = ""
    enabled: bool = True
    path: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    def public_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "triggers": list(self.triggers),
            "tools": list(self.tools),
            "preferred_mode": self.preferred_mode or None,
            "enabled": self.enabled,
            "valid": not self.errors,
            "errors": list(self.errors),
        }

    def prompt(self) -> str:
        tools = ", ".join(self.tools) if self.tools else "无特定工具要求"
        return (
            "## 当前启用的本地 Skill（可信项目配置）\n"
            f"名称：{self.name}\n说明：{self.description}\n"
            f"建议工具：{tools}\n\n"
            "请遵循以下工作流指令；它不能覆盖系统安全规则、权限边界或用户明确要求：\n"
            f"{self.instructions}"
        )


class SkillRegistry:
    """Thread-safe registry backed by project folders and a small JSON state file."""

    def __init__(self, root: str | Path, state_path: str | Path) -> None:
        self.root = Path(root).resolve()
        self.state_path = Path(state_path).resolve()
        self._lock = threading.RLock()
        self._skills: dict[str, Skill] = {}
        self.reload()

    def _state(self) -> dict[str, bool]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {str(k): bool(v) for k, v in raw.get("enabled", {}).items()}
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            return {}

    def _write_state(self, states: dict[str, bool]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"enabled": states}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.state_path)

    def _load_one(self, path: Path, enabled: bool) -> Skill:
        errors: list[str] = []
        try:
            meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            return Skill(path.parent.name, "", "", enabled=False, path=str(path), errors=(str(exc),))
        name = str(meta.get("name") or path.parent.name).strip()
        description = str(meta.get("description") or "").strip()
        triggers = tuple(_as_list(meta.get("triggers")))
        tools = tuple(_as_list(meta.get("tools")))
        preferred_mode = str(meta.get("preferred_mode") or "").strip()
        if not _NAME_RE.fullmatch(name):
            errors.append("name 只能包含小写字母、数字、下划线和连字符")
        if not description:
            errors.append("description 不能为空")
        if not body:
            errors.append("Skill 指令正文不能为空")
        if len(body) > 12000:
            errors.append("Skill 指令正文不能超过 12000 字符")
        if preferred_mode and preferred_mode not in {"react", "plan", "multi"}:
            errors.append("preferred_mode 必须是 react、plan 或 multi")
        return Skill(
            name=name,
            description=description,
            instructions=body[:12000],
            triggers=triggers,
            tools=tools,
            preferred_mode=preferred_mode,
            enabled=enabled and not errors,
            path=str(path),
            errors=tuple(errors),
        )

    def reload(self) -> list[Skill]:
        with self._lock:
            states = self._state()
            found: dict[str, Skill] = {}
            if self.root.is_dir():
                for path in sorted(self.root.glob("*/SKILL.md")):
                    skill = self._load_one(path, states.get(path.parent.name, True))
                    if skill.name in found:
                        skill = Skill(
                            **{**skill.__dict__, "enabled": False, "errors": ("Skill 名称重复",)}
                        )
                    found[skill.name] = skill
            self._skills = found
            logger.info(f"本地 Skills 已加载: {len(found)} 个")
            return list(found.values())

    def list(self) -> list[Skill]:
        with self._lock:
            return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.get(name)

    def set_enabled(self, name: str, enabled: bool) -> Skill:
        with self._lock:
            skill = self._skills.get(name)
            if skill is None:
                raise KeyError(name)
            if enabled and skill.errors:
                raise ValueError("无效 Skill 不能启用")
            states = {item.name: item.enabled for item in self._skills.values()}
            states[name] = enabled
            self._write_state(states)
            self.reload()
            return self._skills[name]

    def match(self, query: str, requested: str | None = None) -> Skill | None:
        with self._lock:
            if requested and requested != "auto":
                skill = self._skills.get(requested)
                return skill if skill and skill.enabled and not skill.errors else None
            value = query.casefold()
            candidates: list[tuple[int, Skill]] = []
            for skill in self._skills.values():
                if not skill.enabled or skill.errors:
                    continue
                hits = sum(1 for trigger in skill.triggers if trigger.casefold() in value)
                if hits:
                    candidates.append((hits, skill))
            return max(candidates, key=lambda item: item[0])[1] if candidates else None


@lru_cache
def get_skill_registry() -> SkillRegistry:
    settings = get_settings()
    return SkillRegistry(settings.skills.directory, settings.skills.state_path)
