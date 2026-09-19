"""Local Skill discovery and administration API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import User, get_current_user, require_admin
from app.skills.registry import get_skill_registry

router = APIRouter(prefix="/skills", tags=["skills"])


class SkillStateRequest(BaseModel):
    enabled: bool


class SkillMatchRequest(BaseModel):
    query: str
    skill: str = "auto"


@router.get("")
async def list_skills(_: User = Depends(get_current_user)) -> dict:
    skills = get_skill_registry().list()
    return {"skills": [skill.public_dict() for skill in skills], "count": len(skills)}


@router.post("/match")
async def match_skill(req: SkillMatchRequest, _: User = Depends(get_current_user)) -> dict:
    skill = get_skill_registry().match(req.query, req.skill)
    return {"matched": skill.public_dict() if skill else None}


@router.post("/reload")
async def reload_skills(_: User = Depends(require_admin)) -> dict:
    skills = get_skill_registry().reload()
    return {"skills": [skill.public_dict() for skill in skills], "count": len(skills)}


@router.put("/{name}/enabled")
async def set_skill_enabled(
    name: str, req: SkillStateRequest, _: User = Depends(require_admin)
) -> dict:
    try:
        skill = get_skill_registry().set_enabled(name, req.enabled)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Skill 不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return skill.public_dict()
