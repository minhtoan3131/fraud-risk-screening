"""Runtime path configuration for the application."""

from __future__ import annotations

import os
from pathlib import Path


OFFICIAL_ROOT_ENV = "FRAUD_SCREENING_OFFICIAL_ROOT"


def resolve_official_root(
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        return Path(
            explicit_path
        ).expanduser().resolve()

    env_value = os.environ.get(
        OFFICIAL_ROOT_ENV
    )

    if env_value:
        return Path(
            env_value
        ).expanduser().resolve()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    return (
        project_root
        / "final_pipeline"
        / "artifacts"
        / "official"
    ).resolve()
