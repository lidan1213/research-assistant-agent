"""用户认证与角色权限：SQLite 用户表 + HMAC 签名 Token（零第三方依赖）。

设计要点：
- 用户存储：独立 SQLite 文件（data/users.db），表 users(username PK, password_hash, salt, role, created_at)。
- 密码：sha256(salt + password) 加盐哈希，不存明文。
- Token：HMAC-SHA256 签名（payload = base64(username).base64(expiry)），24h 过期，
  签名密钥取 settings.auth_secret（.env 可覆盖），标准库实现，无需 JWT 依赖。
- 角色：admin（管理员，可管理知识库/评测）/ student（学生，仅对话）。
- 依赖注入：get_current_user（任意登录用户）/ require_admin（仅管理员）。
- 预置账号：admin/123456（管理员）、student/123456（学生），首次启动自动创建。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.core.exceptions import AgentError, AuthError, ForbiddenError

_bearer = HTTPBearer(auto_error=False)

TOKEN_TTL = 24 * 3600  # 24 小时

ROLE_ADMIN = "admin"
ROLE_STUDENT = "student"

# 预置账号（首次启动 seed；生产环境请通过 .env 覆盖密码）
DEFAULT_USERS = [
    {"username": "admin", "password": "123456", "role": ROLE_ADMIN},
    {"username": "student", "password": "123456", "role": ROLE_STUDENT},
    {"username": "student1", "password": "123456", "role": ROLE_STUDENT},
    {"username": "student2", "password": "123456", "role": ROLE_STUDENT},
    {"username": "student3", "password": "123456", "role": ROLE_STUDENT},
    {"username": "student4", "password": "123456", "role": ROLE_STUDENT},
]


@dataclass
class User:
    username: str
    role: str


# ---------------------------------------------------------------------------
# 用户存储（SQLite）
# ---------------------------------------------------------------------------
def _users_db_path() -> str:
    return os.environ.get(
        "AUTH_USERS_DB", os.path.join(os.path.dirname(get_settings().memory.longterm_path), "users.db")
    )


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


class UserStore:
    """用户表读写（线程安全：每个操作短连接 + WAL）。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _users_db_path()
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)
        self._init_schema()
        self._seed_defaults()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users("
                "username TEXT PRIMARY KEY, password_hash TEXT, salt TEXT, "
                "role TEXT NOT NULL, created_at TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    def _seed_defaults(self) -> None:
        """首次启动插入预置账号（已存在则跳过）。"""
        conn = self._conn()
        try:
            for u in DEFAULT_USERS:
                exists = conn.execute(
                    "SELECT 1 FROM users WHERE username=?", (u["username"],)
                ).fetchone()
                if exists:
                    continue
                salt = hashlib.sha256(os.urandom(16)).hexdigest()
                conn.execute(
                    "INSERT INTO users(username, password_hash, salt, role, created_at) "
                    "VALUES (?,?,?,?,datetime('now'))",
                    (u["username"], _hash_password(u["password"], salt), salt, u["role"]),
                )
            conn.commit()
        finally:
            conn.close()

    def verify(self, username: str, password: str) -> Optional[User]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT username, password_hash, salt, role FROM users WHERE username=?",
                (username,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        _, pw_hash, salt, role = row
        if _hash_password(password, salt) != pw_hash:
            return None
        return User(username=row[0], role=role)

    def reset_password(self, username: str, new_password: str) -> bool:
        """重置用户密码（管理员用）。返回是否成功（用户不存在返回 False）。"""
        if not new_password or len(new_password) < 4:
            raise ValueError("新密码长度至少 4 位")
        conn = self._conn()
        try:
            cur = conn.execute("SELECT 1 FROM users WHERE username=?", (username,))
            if cur.fetchone() is None:
                return False
            salt = hashlib.sha256(os.urandom(16)).hexdigest()
            conn.execute(
                "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                (_hash_password(new_password, salt), salt, username),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get(self, username: str) -> Optional[User]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT username, role FROM users WHERE username=?", (username,)
            ).fetchone()
        finally:
            conn.close()
        return User(username=row[0], role=row[1]) if row else None

    def create_user(self, username: str, password: str, role: str = ROLE_STUDENT) -> bool:
        """注册新用户（角色默认学生）。用户名已存在返回 False，成功返回 True。"""
        conn = self._conn()
        try:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username=?", (username,)
            ).fetchone()
            if exists:
                return False
            salt = hashlib.sha256(os.urandom(16)).hexdigest()
            conn.execute(
                "INSERT INTO users(username, password_hash, salt, role, created_at) "
                "VALUES (?,?,?,?,datetime('now'))",
                (username, _hash_password(password, salt), salt, role),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def list_by_role(self, role: str) -> list[User]:
        """按角色列出用户（如管理员查看全部学生）。"""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT username, role FROM users WHERE role=? ORDER BY username", (role,)
            ).fetchall()
        finally:
            conn.close()
        return [User(username=r[0], role=r[1]) for r in rows]


# ---------------------------------------------------------------------------
# HMAC Token 签发 / 校验
# ---------------------------------------------------------------------------
def _secret() -> str:
    return get_settings().auth_secret


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")


def _unb64(s: str) -> str:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")


def issue_token(user: User, ttl: int = TOKEN_TTL) -> str:
    """签发 token：base64(username).base64(expiry).hmac_signature"""
    payload = f"{_b64(user.username)}.{_b64(str(int(time.time()) + ttl))}"
    sig = hmac.new(_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[User]:
    """校验 token，返回用户；无效/过期返回 None。"""
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        b64_user, b64_exp = payload.split(".")
        if int(_unb64(b64_exp)) < time.time():
            return None
        username = _unb64(b64_user)
    except Exception:  # noqa: BLE001
        return None
    return UserStore().get(username)


# ---------------------------------------------------------------------------
# FastAPI 依赖
# ---------------------------------------------------------------------------
def _extract_token(request: Request, creds: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    """优先取 Authorization: Bearer；WS 场景可退化为 ?token= 查询参数。"""
    if creds is not None:
        return creds.credentials
    token = request.query_params.get("token")
    return token


async def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> User:
    """任意已登录用户（对话等通用功能）。"""
    token = _extract_token(request, creds)
    user = verify_token(token) if token else None
    if user is None:
        raise AuthError("未登录或登录已过期")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """仅管理员（知识库管理 / 评测等功能）。"""
    if user.role != ROLE_ADMIN:
        raise ForbiddenError("需要管理员权限")
    return user


def parse_token_from_query(query: str) -> Optional[str]:
    """WebSocket 无法带 Authorization header，从 URL query 解析 token。"""
    if not query:
        return None
    for part in query.split("&"):
        if part.startswith("token="):
            return part[len("token="):]
    return None
