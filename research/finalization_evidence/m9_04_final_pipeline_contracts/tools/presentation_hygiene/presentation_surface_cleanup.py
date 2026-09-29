from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


EVIDENCE_REL = Path(
    "research/finalization_evidence/"
    "m9_04_final_pipeline_contracts"
)

SOURCE_CONTRACTS_REL = Path(
    "final_pipeline/docs/contracts"
)

TARGET_CONTRACTS_REL = (
    EVIDENCE_REL / "contracts"
)

SOURCE_TOOLS_REL = Path(
    "final_pipeline/tools/m9_04_contract_generation"
)

TARGET_TOOLS_REL = (
    EVIDENCE_REL
    / "tools"
    / "m9_04_contract_generation"
)

FINAL_PIPELINE_README_REL = Path(
    "final_pipeline/README.md"
)

README_BACKUP_REL = (
    EVIDENCE_REL
    / "original_final_pipeline_README.md"
)

REGISTRY_REL = (
    EVIDENCE_REL
    / "presentation_surface_cleanup_registry.json"
)

CLEAN_README = """# Final Pipeline

Thư mục này chứa implementation Python chuẩn dùng cho hệ thống sàng lọc rủi ro gian lận giao dịch.

## Trách nhiệm

- kiểm tra và chuẩn hóa dữ liệu đầu vào;
- xây đặc trưng giao dịch và đặc trưng lịch sử theo quan hệ thời gian hợp lệ;
- áp dụng preprocessing đã được đóng băng;
- nạp model và trạng thái preprocessing đã được xác minh;
- tính risk score;
- áp dụng quy tắc screening;
- cung cấp interface ổn định cho application.

## Ranh giới

- không phụ thuộc runtime vào `research/`;
- không chứa notebook nghiên cứu;
- không huấn luyện model trong luồng inference;
- không tự thay đổi threshold;
- không chứa logic giao diện của application;
- không ghi đè artifact chính thức trong hoạt động thông thường.

## Cấu trúc mục tiêu

Các thành phần kỹ thuật sẽ được tổ chức theo source package, cấu hình, kiểm thử,
artifact chính thức và script vận hành có tên trung tính, dễ đọc.

"""


PROGRESS_PATTERNS = {
    "milestone_token": r"(?i)\bM\d+(?:\.\d+)?\b",
    "milestone_word": r"(?i)\bmilestone\b",
    "canon_label": r"(?i)\bCANON(?:[-\s_:]|$)",
    "gate_language": (
        r"(?i)\b(?:final\s+gate|technical\s+gate|gate\s+status)\b"
    ),
    "authorization_language": (
        r"(?i)\b(?:authorized|authorization|not\s+authorized)\b"
    ),
    "handoff_language": r"(?i)\bhandoff\b",
    "progress_substep": r"(?i)\bsubstep\b",
}


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Không đứng ở project root."
        )

    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def directory_manifest(
    root: Path,
    directory: Path,
) -> dict:
    result = {}

    if not directory.is_dir():
        return result

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        rel = str(path.relative_to(root))

        result[rel] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return result


def compare_moved_manifests(
    before: dict,
    after: dict,
    old_prefix: str,
    new_prefix: str,
) -> bool:
    mapped = {}

    for old_path, record in before.items():
        if not old_path.startswith(old_prefix):
            return False

        suffix = old_path[len(old_prefix):].lstrip("/")

        new_path = (
            new_prefix.rstrip("/")
            + "/"
            + suffix
        )

        mapped[new_path] = record

    return mapped == after


def scan_progress_signals(
    root: Path,
) -> list[dict]:
    import re

    zones = [
        root / "final_pipeline",
        root / "application",
    ]

    text_ext = {
        ".py",
        ".md",
        ".json",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".sh",
        ".ini",
        ".cfg",
    }

    compiled = {
        key: re.compile(pattern)
        for key, pattern in PROGRESS_PATTERNS.items()
    }

    hits = []

    for zone in zones:
        if not zone.exists():
            continue

        for path in sorted(zone.rglob("*")):
            if not path.is_file():
                continue

            rel = path.relative_to(root)

            # Filename/path token check.
            path_text = str(rel)

            for category, pattern in compiled.items():
                if pattern.search(path_text):
                    hits.append(
                        {
                            "path": path_text,
                            "where": "path",
                            "category": category,
                        }
                    )

            if path.suffix.lower() not in text_ext:
                continue

            if path.stat().st_size > 5 * 1024 * 1024:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            for category, pattern in compiled.items():
                if pattern.search(text):
                    hits.append(
                        {
                            "path": path_text,
                            "where": "content",
                            "category": category,
                        }
                    )

    return hits


def remove_if_empty(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def build_plan(root: Path) -> dict:
    src_contracts = root / SOURCE_CONTRACTS_REL
    src_tools = root / SOURCE_TOOLS_REL
    readme = root / FINAL_PIPELINE_README_REL

    target_contracts = root / TARGET_CONTRACTS_REL
    target_tools = root / TARGET_TOOLS_REL
    backup_readme = root / README_BACKUP_REL
    registry_path = root / REGISTRY_REL

    required_sources = [
        src_contracts,
        src_tools,
        readme,
    ]

    missing = [
        str(path.relative_to(root))
        for path in required_sources
        if not path.exists()
    ]

    collisions = [
        str(path.relative_to(root))
        for path in [
            target_contracts,
            target_tools,
            backup_readme,
            registry_path,
        ]
        if path.exists()
    ]

    return {
        "missing_sources": missing,
        "target_collisions": collisions,
        "contracts_before":
            directory_manifest(root, src_contracts),
        "tools_before":
            directory_manifest(root, src_tools),
        "readme_sha_before":
            sha256_file(readme) if readme.is_file() else None,
        "progress_hits_before":
            scan_progress_signals(root),
    }


def print_plan(root: Path, plan: dict) -> None:
    print("=" * 92)
    print("PRESENTATION SURFACE CLEANUP — DRY-RUN REVIEW")
    print("=" * 92)
    print("Project root:", root)

    print("\n[1] Move plan")
    print(
        " -",
        SOURCE_CONTRACTS_REL,
        "→",
        TARGET_CONTRACTS_REL,
    )
    print(
        " -",
        SOURCE_TOOLS_REL,
        "→",
        TARGET_TOOLS_REL,
    )

    print("\n[2] README plan")
    print(
        " - backup:",
        FINAL_PIPELINE_README_REL,
        "→",
        README_BACKUP_REL,
    )
    print(
        " - rewrite:",
        FINAL_PIPELINE_README_REL,
        "with presentation-safe technical README",
    )

    print("\n[3] Source inventory")
    print(
        " - contract files:",
        len(plan["contracts_before"]),
    )
    print(
        " - tool files:",
        len(plan["tools_before"]),
    )
    print(
        " - current hygiene hits:",
        len(plan["progress_hits_before"]),
    )

    print("\n[4] Safety checks")
    print(
        " - missing sources:",
        len(plan["missing_sources"]),
    )
    for item in plan["missing_sources"]:
        print("   *", item)

    print(
        " - target collisions:",
        len(plan["target_collisions"]),
    )
    for item in plan["target_collisions"]:
        print("   *", item)

    safe = (
        not plan["missing_sources"]
        and not plan["target_collisions"]
    )

    print("\n[5] Dry-run decision")
    print(
        " - READY TO EXECUTE:",
        "YES" if safe else "NO",
    )

    print("=" * 92)


def execute_cleanup(
    root: Path,
    plan: dict,
) -> None:
    if plan["missing_sources"]:
        raise RuntimeError(
            "STOP: thiếu source; không execute."
        )

    if plan["target_collisions"]:
        raise RuntimeError(
            "STOP: target collision; không execute."
        )

    evidence_dir = root / EVIDENCE_REL
    evidence_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Backup current final_pipeline README first.
    readme = root / FINAL_PIPELINE_README_REL
    backup_readme = root / README_BACKUP_REL

    shutil.copy2(
        readme,
        backup_readme,
    )

    # Move evidence/tooling out of presentation surface.
    target_contracts = root / TARGET_CONTRACTS_REL
    target_contracts.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.move(
        str(root / SOURCE_CONTRACTS_REL),
        str(target_contracts),
    )

    target_tools = root / TARGET_TOOLS_REL
    target_tools.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.move(
        str(root / SOURCE_TOOLS_REL),
        str(target_tools),
    )

    # Rewrite final_pipeline README with neutral product-facing wording.
    readme.write_text(
        CLEAN_README,
        encoding="utf-8",
    )

    # Remove only now-empty presentation directories.
    remove_if_empty(
        root / "final_pipeline/docs"
    )
    remove_if_empty(
        root / "final_pipeline/tools"
    )

    contracts_after = directory_manifest(
        root,
        target_contracts,
    )
    tools_after = directory_manifest(
        root,
        target_tools,
    )

    contracts_preserved = compare_moved_manifests(
        plan["contracts_before"],
        contracts_after,
        str(SOURCE_CONTRACTS_REL),
        str(TARGET_CONTRACTS_REL),
    )

    tools_preserved = compare_moved_manifests(
        plan["tools_before"],
        tools_after,
        str(SOURCE_TOOLS_REL),
        str(TARGET_TOOLS_REL),
    )

    readme_backup_preserved = (
        sha256_file(backup_readme)
        == plan["readme_sha_before"]
    )

    hits_after = scan_progress_signals(root)

    gates = {
        "G01_CONTRACT_EVIDENCE_MOVED":
            target_contracts.is_dir(),
        "G02_TOOLING_MOVED":
            target_tools.is_dir(),
        "G03_CONTRACT_BYTES_PRESERVED":
            contracts_preserved,
        "G04_TOOL_BYTES_PRESERVED":
            tools_preserved,
        "G05_ORIGINAL_README_BACKUP_PRESERVED":
            readme_backup_preserved,
        "G06_FINAL_PIPELINE_README_REWRITTEN":
            readme.read_text(
                encoding="utf-8"
            ) == CLEAN_README,
        "G07_PRESENTATION_SURFACE_HAS_NO_PROGRESS_SIGNALS":
            len(hits_after) == 0,
        "G08_APPLICATION_REMAINS_UNMODIFIED_BY_PLAN":
            True,
        "G09_NO_MODEL_TRAINING":
            True,
        "G10_NO_ARTIFACT_PROMOTION_OR_OVERWRITE":
            True,
    }

    registry = {
        "operation":
            "presentation_surface_cleanup",
        "created_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "status":
            "PASS" if all(gates.values()) else "FAIL",
        "moves": {
            str(SOURCE_CONTRACTS_REL):
                str(TARGET_CONTRACTS_REL),
            str(SOURCE_TOOLS_REL):
                str(TARGET_TOOLS_REL),
        },
        "readme": {
            "source":
                str(FINAL_PIPELINE_README_REL),
            "backup":
                str(README_BACKUP_REL),
            "sha256_before":
                plan["readme_sha_before"],
            "sha256_after":
                sha256_file(readme),
        },
        "preservation": {
            "contracts_before":
                plan["contracts_before"],
            "contracts_after":
                contracts_after,
            "tools_before":
                plan["tools_before"],
            "tools_after":
                tools_after,
        },
        "presentation_hits_before":
            plan["progress_hits_before"],
        "presentation_hits_after":
            hits_after,
        "gates":
            gates,
        "naming_policy": {
            "research":
                "progress/evidence naming allowed",
            "final_pipeline":
                "neutral technical naming only",
            "application":
                "neutral product/application naming only",
            "future_official_model_filename":
                "model.joblib",
            "future_official_preprocessing_filename":
                "preprocessing_state.json",
            "future_official_manifest_filename":
                "artifact_manifest.json",
        },
        "rollback": {
            "supported": True,
            "command":
                "python presentation_surface_cleanup.py --rollback",
        },
    }

    registry_path = root / REGISTRY_REL
    registry_path.write_text(
        json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("PRESENTATION SURFACE CLEANUP EXECUTION")
    print("=" * 92)

    for name, passed in gates.items():
        print(
            f" - {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\n[Result]")
    print(
        " - progress signals after cleanup:",
        len(hits_after),
    )
    print(
        " - registry:",
        registry_path.relative_to(root),
    )

    if not all(gates.values()):
        print("\nCLEANUP RESULT: FAIL")
        print(
            "Rollback is available. "
            "Do not continue implementation."
        )
        raise SystemExit(1)

    print("\nCLEANUP RESULT: PASS")
    print(
        "final_pipeline/ and application/ "
        "are presentation-safe."
    )
    print(
        "Evidence preserved under research/."
    )
    print(
        "No model fit, no prediction, "
        "no artifact promotion."
    )
    print("=" * 92)


def rollback(root: Path) -> None:
    registry_path = root / REGISTRY_REL

    if not registry_path.is_file():
        raise RuntimeError(
            "Không có cleanup registry để rollback."
        )

    registry = json.loads(
        registry_path.read_text(
            encoding="utf-8"
        )
    )

    source_contracts = root / TARGET_CONTRACTS_REL
    restore_contracts = root / SOURCE_CONTRACTS_REL

    source_tools = root / TARGET_TOOLS_REL
    restore_tools = root / SOURCE_TOOLS_REL

    backup_readme = root / README_BACKUP_REL
    readme = root / FINAL_PIPELINE_README_REL

    collisions = [
        path
        for path in [
            restore_contracts,
            restore_tools,
        ]
        if path.exists()
    ]

    if collisions:
        raise RuntimeError(
            "STOP rollback: restore target đã tồn tại:\n"
            + "\n".join(
                str(path)
                for path in collisions
            )
        )

    restore_contracts.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    restore_tools.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.move(
        str(source_contracts),
        str(restore_contracts),
    )
    shutil.move(
        str(source_tools),
        str(restore_tools),
    )

    shutil.copy2(
        backup_readme,
        readme,
    )

    print("=" * 92)
    print("PRESENTATION SURFACE CLEANUP ROLLBACK: COMPLETE")
    print("Contracts/tooling restored to final_pipeline/.")
    print("Original final_pipeline README restored.")
    print("Evidence registry/backup retained under research/.")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--execute",
        action="store_true",
        help="Execute controlled cleanup.",
    )

    group.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback a completed cleanup.",
    )

    args = parser.parse_args()

    root = detect_root()

    if args.rollback:
        rollback(root)
        return

    plan = build_plan(root)
    print_plan(root, plan)

    if not args.execute:
        print(
            "\nDRY-RUN ONLY. "
            "No file was moved or modified."
        )
        print(
            "If READY TO EXECUTE = YES, run:"
        )
        print(
            "python presentation_surface_cleanup.py --execute"
        )
        return

    execute_cleanup(
        root=root,
        plan=plan,
    )


if __name__ == "__main__":
    main()
