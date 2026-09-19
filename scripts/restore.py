"""从备份 zip 恢复数据（backup.py 的逆操作）。

用法：
    python scripts/restore.py --backup backups/research_assistant_20260808_232432.zip
    python scripts/restore.py --backup xxx.zip --yes     # 跳过确认
    python scripts/restore.py --backup xxx.zip --dry-run # 只列出将恢复的文件

安全措施：
- 恢复前自动把现有 data/ 备份为 backups/pre_restore_<时间戳>.zip（可回滚）；
- 只接受 zip 内 data/ 前缀的成员，且拒绝含 ".." 的路径（防 zip-slip）。
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
BACKUP_DIR = os.path.join(ROOT, "backups")


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """只允许 data/ 前缀、无 .. 与绝对路径的成员。"""
    out = []
    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        if not name.startswith("data/"):
            print(f"[restore] 跳过非数据成员（不在 data/ 下）: {name}")
            continue
        if ".." in name or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
            print(f"[restore] ✖ 拒绝不安全路径: {name}")
            continue
        out.append(info)
    return out


def _pre_backup() -> str | None:
    """恢复前自动备份现有 data/，返回备份路径（无数据则 None）。"""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from backup import backup  # 复用 backup 逻辑

    zip_path = backup(DATA_DIR, BACKUP_DIR)
    if not zip_path:
        return None
    pre = os.path.join(BACKUP_DIR, f"pre_restore_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    os.replace(zip_path, pre)
    return pre


def restore(zip_path: str, yes: bool = False, dry_run: bool = False) -> int:
    if not os.path.exists(zip_path):
        print(f"[restore] ✖ 备份文件不存在: {zip_path}")
        return 1
    with zipfile.ZipFile(zip_path) as zf:
        members = _safe_members(zf)
        if not members:
            print("[restore] ✖ zip 中没有可恢复的 data/ 数据")
            return 1
        total_mb = sum(m.file_size for m in members) / 1024 / 1024
        print(f"[restore] 将恢复 {len(members)} 个文件（共 {total_mb:.2f} MB）:")
        for m in members[:10]:
            print(f"  - {m.filename}")
        if len(members) > 10:
            print(f"  … 等 {len(members) - 10} 个文件")

        if dry_run:
            print("[restore] dry-run：未执行任何写入")
            return 0
        if not yes:
            ans = input("确认恢复？将覆盖 data/ 下同名文件（恢复前自动备份） [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("[restore] 已取消")
                return 1

        pre = _pre_backup()
        if pre:
            print(f"[restore] 现有 data/ 已备份到: {pre}")
        else:
            print("[restore] data/ 当前无数据，直接恢复")

        try:
            for m in members:
                zf.extract(m, ROOT)
            print(f"[restore] ✅ 已恢复 {len(members)} 个文件到 {DATA_DIR}")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"[restore] ✖ 恢复失败: {e}（可用 pre_restore 备份回滚）")
            return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="从备份 zip 恢复数据")
    ap.add_argument("--backup", required=True, help="备份 zip 路径")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    ap.add_argument("--dry-run", action="store_true", help="只列出将恢复的文件")
    args = ap.parse_args()
    sys.exit(restore(args.backup, yes=args.yes, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
