"""认证路由：注册 / 登录 / 当前用户信息。"""
from __future__ import annotations

import re
import threading
import time

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.core.auth import ROLE_STUDENT, UserStore, get_current_user, issue_token
from app.core.exceptions import AuthError

router = APIRouter(tags=["auth"])

# 用户名规则（账号不限，宽松校验）：去首尾空白后 1-64 字符，禁止内部空白/控制字符，
# 禁止 "__"（会话前缀分隔符保留），避免干扰 session_id 规范化和 URL 编码
USERNAME_RE = re.compile(r"^[^\s_][^\s]{0,62}[^\s_]$|^[^\s_]$")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="账号（不限格式）")
    password: str = Field(..., min_length=1, max_length=128, description="密码")
    password2: str = Field(..., min_length=1, max_length=128, description="确认密码（需与密码一致）")


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str


class MeResponse(BaseModel):
    username: str
    role: str


def _validate_username(username: str) -> str:
    """校验并规范化用户名：去首尾空白；空/含内部空白/含 __ 均拒绝。"""
    username = (username or "").strip()
    if not username:
        raise AuthError("账号不能为空")
    if "__" in username:
        raise AuthError("账号不能包含 '__'")
    if not USERNAME_RE.match(username) or any(ord(ch) < 32 for ch in username):
        raise AuthError("账号包含非法字符（不能含空格或控制字符）")
    return username


@router.post("/auth/register", response_model=LoginResponse)
async def register(req: RegisterRequest) -> LoginResponse:
    """注册新账号（角色：学生）。成功即返回 token，可直接进入系统。"""
    username = _validate_username(req.username)
    if len(username) > 64:
        raise AuthError("账号过长（上限 64 字符）")
    if len(req.password) < 8 or not re.search(r"[A-Za-z]", req.password) or not re.search(r"\d", req.password):
        raise AuthError("密码至少 8 位，且需同时包含字母和数字")
    if req.password != req.password2:
        raise AuthError("两次输入的密码不一致")
    store = UserStore()
    ok = store.create_user(username, req.password, role=ROLE_STUDENT)
    if not ok:
        raise AuthError(f"账号「{username}」已存在，请直接登录或换一个账号")
    user = store.get(username)
    return LoginResponse(token=issue_token(user), username=user.username, role=user.role)


# ---------- 登录失败限流（账号|IP 维度，连续 5 次失败锁定 15 分钟） ----------
_MAX_FAILS = 5
_LOCK_SECONDS = 15 * 60
_login_fails: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _fail_key(username: str, request: Request) -> str:
    ip = request.client.host if request.client else "?"
    return f"{username}|{ip}"


def _check_locked(key: str) -> int:
    """返回剩余锁定秒数（0=未锁定）。"""
    with _login_lock:
        now = time.time()
        ts = [t for t in _login_fails.get(key, []) if now - t < _LOCK_SECONDS]
        _login_fails[key] = ts
        if len(ts) >= _MAX_FAILS:
            return int(_LOCK_SECONDS - (now - ts[0]))
    return 0


def _record_fail(key: str) -> None:
    with _login_lock:
        _login_fails.setdefault(key, []).append(time.time())
        _login_fails[key] = _login_fails[key][-_MAX_FAILS:]


def _reset_fails(key: str) -> None:
    with _login_lock:
        _login_fails.pop(key, None)


@router.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request) -> LoginResponse:
    key = _fail_key(req.username, request)
    remaining = _check_locked(key)
    if remaining:
        minutes = max(1, remaining // 60 + 1)
        raise AuthError(f"登录尝试次数过多，账号已临时锁定，请约 {minutes} 分钟后再试")
    user = UserStore().verify(req.username, req.password)
    if user is None:
        _record_fail(key)
        raise AuthError("用户名或密码错误")
    _reset_fails(key)
    return LoginResponse(token=issue_token(user), username=user.username, role=user.role)


@router.get("/auth/me", response_model=MeResponse)
async def me(user=Depends(get_current_user)) -> MeResponse:
    return MeResponse(username=user.username, role=user.role)
