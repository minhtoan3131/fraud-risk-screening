"""Filesystem policy for rebuild outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fraud_screening.errors import (
    ArtifactCompatibilityError,
)


_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"
)

_FORBIDDEN_COMPONENTS = {
    "official",
    "artifacts",
    "research",
}


@dataclass(frozen=True, slots=True)
class RebuildOutputPaths:
    """Resolved paths for one isolated rebuild run."""

    run_id: str
    run_dir: Path
    model_path: Path
    preprocessing_path: Path
    manifest_path: Path

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.model_path,
            self.preprocessing_path,
            self.manifest_path,
        )


def validate_run_id(run_id: str) -> str:
    """Validate a human-readable, filesystem-safe rebuild run id."""

    if not isinstance(run_id, str):
        raise ValueError(
            "run_id must be a string."
        )

    normalized = run_id.strip()

    if not normalized:
        raise ValueError(
            "run_id must not be empty."
        )

    if not _RUN_ID_PATTERN.fullmatch(
        normalized
    ):
        raise ValueError(
            "run_id may contain only letters, numbers, '.', '_' and '-'."
        )

    lowered_parts = {
        part.lower()
        for part in re.split(
            r"[._-]+",
            normalized,
        )
        if part
    }

    if lowered_parts & _FORBIDDEN_COMPONENTS:
        raise ValueError(
            "run_id contains a reserved path term."
        )

    return normalized


def _ensure_under(
    candidate: Path,
    parent: Path,
) -> None:
    try:
        candidate.relative_to(
            parent
        )
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            "Rebuild output escaped the allowed rebuild root."
        ) from exc


def create_rebuild_output_paths(
    project_root: str | Path,
    run_id: str,
    *,
    create: bool = False,
) -> RebuildOutputPaths:
    """Resolve a unique rebuild directory without touching official artifacts."""

    root = Path(
        project_root
    ).resolve()

    validated_run_id = validate_run_id(
        run_id
    )

    rebuild_root = (
        root
        / "final_pipeline"
        / "outputs"
        / "rebuilds"
    ).resolve()

    official_root = (
        root
        / "final_pipeline"
        / "artifacts"
        / "official"
    ).resolve()

    run_dir = (
        rebuild_root
        / validated_run_id
    ).resolve()

    _ensure_under(
        run_dir,
        rebuild_root,
    )

    if (
        run_dir == official_root
        or official_root in run_dir.parents
    ):
        raise ArtifactCompatibilityError(
            "Rebuild output may not target official artifacts."
        )

    if run_dir.exists():
        raise FileExistsError(
            f"Rebuild run already exists: {run_dir}"
        )

    paths = RebuildOutputPaths(
        run_id=validated_run_id,
        run_dir=run_dir,
        model_path=run_dir / "model.joblib",
        preprocessing_path=
            run_dir / "preprocessing_state.json",
        manifest_path=
            run_dir / "rebuild_manifest.json",
    )

    if create:
        rebuild_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        run_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

    return paths
