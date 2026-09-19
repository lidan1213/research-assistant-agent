"""一键备份：打包用户数据（会话记忆/追踪/笔记/知识库/用户库）为时间戳 zip。

用法：
    python scripts/backup.py                    # 备份到 backups/，保留最近 14 份
    python scripts/backup.py --keep 7           # 只保留最近 7 份
    python scripts/backup.py --out D:/backup    # 指定输出目录
    python scripts/backup.py --with-env         # 额外包含 .env（含 API Key，注意保管）

备份内容（data/ 下）：
    memory.db  traces.db  longterm.db  users.db     会话记忆/追踪/长期记忆/账号
    notes/     knowledge/ knowledge_chroma/ knowledge_docs/  papers/   笔记与知识库
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_KEEP = 14

# 需要备份的 data/ 子项（目录或文件）；logs/ 与临时文件排除
BACKUP_ITEMS = [
    "memory.db",
    "traces.db",
    "longterm.db",
    "users.db",
    "notes",
    "knowledge",
    "knowledge_chroma",
    "knowledge_docs",
    "papers",
]

# SQLite WAL/SHM 附属文件：单独备份，避免 zip 冲突
WAL_SUFFIXES = ("-wal", "-shm")


def _collect(path: str, arc_base: str, out: list[tuple[str, str]]) -> None:
    """递归收集 (磁盘路径, 归档内路径)。"""
    if os.path.isfile(path):
        out.append((path, arc_base))
        return
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name == "__pycache__" or name.endswith(WAL_SUFFIXES):
                continue
            _collect(os.path.join(path, name), f"{arc_base}/{name}", out)


def backup(data_dir: str, out_dir: str, with_env: bool = False) -> str:
    """执行备份，返回 zip 路径。"""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(out_dir, f"research_assistant_{stamp}.zip")

    files: list[tuple[str, str]] = []
    missing: list[str] = []
    for item in BACKUP_ITEMS:
        src = os.path.join(data_dir, item)
        if os.path.exists(src):
            _collect(src, f"data/{item}", files)
        else:
            missing.append(item)
    if with_env:
        env_path = os.path.join(ROOT, ".env")
        if os.path.exists(env_path):
            files.append((env_path, ".env"))

    if not files:
        print(f"[backup] data/ 下没有可备份的数据（缺失: {missing}），跳过。")
        return ""

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for disk_path, arc_path in files:
            try:
                zf.write(disk_path, arc_path)
            except OSError as e:
                print(f"[backup] 跳过 {arc_path}: {e}")

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(
        f"[backup] ✅ 已备份 {len(files)} 个文件 -> {zip_path} "
        f"({size_mb:.2f} MB)"
    )
    if missing:
        print(f"[backup] 提示：以下目录不存在，已跳过 -> {missing}")
    return zip_path


def prune(out_dir: str, keep: int) -> int:
    """只保留最近 keep 份备份，删除更旧的。返回删除数。"""
    if keep <= 0:
        return 0
    backups = sorted(
        f for f in os.listdir(out_dir) if f.startswith("research_assistant_") and f.endswith(".zip")
    )
    removed = 0
    for old in backups[:-keep]:
        os.remove(os.path.join(out_dir, old))
        removed += 1
    if removed:
        print(f"[backup] 已清理 {removed} 份旧备份（保留最近 {keep} 份）")
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="科研助手数据备份")
    ap.add_argument("--out", default=os.path.join(ROOT, "backups"), help="备份输出目录")
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=f"保留最近 N 份（默认 {DEFAULT_KEEP}）")
    ap.add_argument("--with-env", action="store_true", help="同时备份 .env（含 API Key）")
    args = ap.parse_args()

    zip_path = backup(DATA_DIR, args.out, with_env=args.with_env)
    prune(args.out, args.keep)
    if not zip_path:
        sys.exit(1)


if __name__ == "__main__":
    main()
