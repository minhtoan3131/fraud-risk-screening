from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


TEXT_EXTENSIONS = {
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

SCAN_ZONES = (
    "final_pipeline",
    "application",
)

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
}

MAX_TEXT_BYTES = 5 * 1024 * 1024

PATTERNS = {
    "milestone_token":
        re.compile(r"(?i)\bM\d+(?:\.\d+)?\b"),
    "milestone_word":
        re.compile(r"(?i)\bmilestone\b"),
    "canon_label":
        re.compile(r"(?i)\bCANON(?:[-\s_:]|$)"),
    "gate_language":
        re.compile(
            r"(?i)\b(?:final\s+gate|technical\s+gate|gate\s+status)\b"
        ),
    "authorization_language":
        re.compile(
            r"(?i)\b(?:authorized|authorization|not\s+authorized)\b"
        ),
    "handoff_language":
        re.compile(r"(?i)\bhandoff\b"),
    "progress_substep":
        re.compile(r"(?i)\bsubstep\b"),
    "progress_registry_name":
        re.compile(
            r"(?i)\b(?:m\d+[_\-.]|milestone[_\-.]).*registry\b"
        ),
}

FILENAME_PATTERNS = {
    "filename_milestone":
        re.compile(r"(?i)(?:^|[_\-. ])M\d+(?:[_\-.]\d+)?(?:[_\-. ]|$)"),
    "filename_canon":
        re.compile(r"(?i)CANON"),
    "filename_milestone_word":
        re.compile(r"(?i)milestone"),
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
            "Không đứng ở project root có research/, "
            "final_pipeline/, application/."
        )

    return root


def excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def compact(text: str, start: int, end: int) -> str:
    left = max(0, start - 80)
    right = min(len(text), end + 140)
    value = text[left:right].replace("\n", "\\n")
    if len(value) > 360:
        value = value[:357] + "..."
    return value


def main() -> None:
    root = detect_root()

    print("=" * 88)
    print("PRESENTATION SURFACE HYGIENE AUDIT")
    print("=" * 88)
    print("Project root:", root)
    print(
        "Mode: READ ONLY — NO MOVE / NO DELETE / "
        "NO SOURCE MODIFICATION"
    )
    print("Zones:")
    for zone in SCAN_ZONES:
        print(" -", zone + "/")

    filename_hits = []
    content_hits = []
    scanned_text_files = 0
    skipped_large_files = 0
    read_errors = []

    for zone_name in SCAN_ZONES:
        zone = root / zone_name

        for path in sorted(zone.rglob("*")):
            if excluded(path):
                continue

            rel = path.relative_to(root)

            # ----------------------------------------------------------
            # Filename / directory-name signals
            # ----------------------------------------------------------
            parts_to_check = path.relative_to(zone).parts
            for part in parts_to_check:
                for category, pattern in FILENAME_PATTERNS.items():
                    if pattern.search(part):
                        filename_hits.append(
                            {
                                "zone": zone_name,
                                "path": str(rel),
                                "component": part,
                                "category": category,
                            }
                        )

            if not path.is_file():
                continue

            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            try:
                size = path.stat().st_size
            except OSError as exc:
                read_errors.append((str(rel), repr(exc)))
                continue

            if size > MAX_TEXT_BYTES:
                skipped_large_files += 1
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as exc:
                read_errors.append((str(rel), repr(exc)))
                continue

            scanned_text_files += 1

            for category, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    content_hits.append(
                        {
                            "zone": zone_name,
                            "path": str(rel),
                            "category": category,
                            "match": match.group(0),
                            "context": compact(
                                text,
                                match.start(),
                                match.end(),
                            ),
                        }
                    )

    file_set = sorted(
        {
            item["path"]
            for item in filename_hits + content_hits
        }
    )

    category_counts = Counter(
        item["category"]
        for item in filename_hits + content_hits
    )

    print("\n[1] Scan summary")
    print(" - text files scanned:", scanned_text_files)
    print(" - large text files skipped:", skipped_large_files)
    print(" - read errors:", len(read_errors))
    print(" - flagged unique paths:", len(file_set))
    print(
        " - total filename/content signals:",
        len(filename_hits) + len(content_hits),
    )

    print("\n[2] Category counts")
    if category_counts:
        for category, count in sorted(category_counts.items()):
            print(f" - {category}: {count}")
    else:
        print(" - none")

    print("\n[3] Flagged paths")
    if file_set:
        for path in file_set:
            print(" -", path)
    else:
        print(" - none")

    print("\n[4] Filename/directory signals")
    if filename_hits:
        for item in filename_hits:
            print(
                f" - {item['path']} | "
                f"{item['category']} | "
                f"component={item['component']!r}"
            )
    else:
        print(" - none")

    print("\n[5] Content signals")
    if content_hits:
        for item in content_hits[:120]:
            print(
                f" - {item['path']} | "
                f"{item['category']} | "
                f"match={item['match']!r}"
            )
            print("   context:", item["context"])

        if len(content_hits) > 120:
            print(
                " - ... truncated:",
                len(content_hits) - 120,
                "additional content signals",
            )
    else:
        print(" - none")

    if read_errors:
        print("\n[6] Read errors")
        for path, error in read_errors:
            print(" -", path, "|", error)

    print("\n" + "=" * 88)

    if file_set:
        print("PRESENTATION SURFACE HYGIENE AUDIT: REVIEW REQUIRED")
        print(
            "Progress/governance traces exist under "
            "final_pipeline/ or application/."
        )
        print(
            "NEXT: review output and prepare controlled relocation/"
            "sanitization plan."
        )
    else:
        print("PRESENTATION SURFACE HYGIENE AUDIT: PASS")
        print(
            "No milestone/progress signals detected in presentation zones."
        )

    print("No files modified.")
    print("=" * 88)


if __name__ == "__main__":
    main()
