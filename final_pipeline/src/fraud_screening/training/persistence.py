"""Guarded persistence for newly learned rebuild artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import joblib

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
)
from fraud_screening.training.outputs import (
    RebuildOutputPaths,
    create_rebuild_output_paths,
)
from fraud_screening.training.preprocessing_fit import (
    RebuildPreprocessingFit,
)
from fraud_screening.training.trainer import (
    RebuildModelFit,
)


@dataclass(frozen=True, slots=True)
class RebuildArtifactSet:
    """Persisted files and fingerprints for one isolated rebuild run."""

    paths: RebuildOutputPaths
    model_sha256: str
    preprocessing_sha256: str
    manifest: dict


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _guard_fit_status(
    model_fit: RebuildModelFit,
    preprocessing_fit: RebuildPreprocessingFit,
) -> None:
    if (
        model_fit.artifact_identity_status
        != "NEW_REBUILD_MODEL"
    ):
        raise ValueError(
            "Model fit is not marked as a new rebuild artifact."
        )

    if (
        model_fit.evaluation_status
        != "NOT_EVALUATED_AS_OFFICIAL"
    ):
        raise ValueError(
            "Rebuild model may not claim official evaluation status."
        )

    if (
        preprocessing_fit.state.get(
            "rebuild_validation_status"
        )
        != "NOT_COMPARED_TO_OFFICIAL_ARTIFACT"
    ):
        raise ValueError(
            "Rebuild preprocessing may not claim official equivalence."
        )

    if (
        preprocessing_fit.state.get(
            "feature_count"
        )
        != 47
    ):
        raise ValueError(
            "Rebuild preprocessing feature count must be 47."
        )

    if (
        model_fit.feature_count
        != 47
    ):
        raise ValueError(
            "Rebuild model feature count must be 47."
        )

    if (
        model_fit.classes
        != (0, 1)
    ):
        raise ValueError(
            "Rebuild model classes must be exactly (0, 1)."
        )

    actual_params = (
        model_fit.estimator
        .get_params(
            deep=False
        )
    )

    for key, expected in (
        EXPECTED_MODEL_PARAMETERS.items()
    ):
        if actual_params[key] != expected:
            raise ValueError(
                f"Rebuild model parameter mismatch: {key}."
            )


def _build_manifest(
    *,
    run_id: str,
    paths: RebuildOutputPaths,
    model_fit: RebuildModelFit,
    preprocessing_fit: RebuildPreprocessingFit,
    model_sha256: str,
    preprocessing_sha256: str,
) -> dict:
    return {
        "manifest_version":
            "rebuild-1.0",
        "run_id":
            run_id,
        "artifact_identity":
            "NEW_REBUILD_ARTIFACT_SET",
        "official_artifact_replacement":
            False,
        "official_metrics_inherited":
            False,
        "model": {
            "file":
                paths.model_path.name,
            "sha256":
                model_sha256,
            "model_id":
                model_fit.model_id,
            "artifact_identity_status":
                model_fit.artifact_identity_status,
            "evaluation_status":
                model_fit.evaluation_status,
            "training_rows":
                model_fit.training_rows,
            "fraud_rows":
                model_fit.fraud_rows,
            "feature_count":
                model_fit.feature_count,
            "classes":
                list(
                    model_fit.classes
                ),
            "parameters": {
                key:
                    model_fit.estimator
                    .get_params(
                        deep=False
                    )[key]
                for key
                in EXPECTED_MODEL_PARAMETERS
            },
        },
        "preprocessing": {
            "file":
                paths.preprocessing_path.name,
            "sha256":
                preprocessing_sha256,
            "strategy":
                preprocessing_fit.state[
                    "strategy"
                ],
            "fit_source":
                preprocessing_fit.state[
                    "fit_source"
                ],
            "fit_row_count":
                preprocessing_fit.state[
                    "fit_row_count"
                ],
            "fit_min_timestamp":
                preprocessing_fit.state[
                    "fit_min_timestamp"
                ],
            "fit_max_timestamp":
                preprocessing_fit.state[
                    "fit_max_timestamp"
                ],
            "feature_count":
                preprocessing_fit.state[
                    "feature_count"
                ],
            "rebuild_validation_status":
                preprocessing_fit.state[
                    "rebuild_validation_status"
                ],
        },
        "safety": {
            "official_path_written":
                False,
            "research_path_written":
                False,
            "silent_overwrite":
                False,
        },
    }


def persist_rebuild_artifacts(
    project_root: str | Path,
    run_id: str,
    *,
    model_fit: RebuildModelFit,
    preprocessing_fit: RebuildPreprocessingFit,
) -> RebuildArtifactSet:
    """Persist one rebuild artifact set under a new unique run directory."""

    _guard_fit_status(
        model_fit,
        preprocessing_fit,
    )

    paths = create_rebuild_output_paths(
        project_root,
        run_id,
        create=True,
    )

    try:
        joblib.dump(
            model_fit.estimator,
            paths.model_path,
        )

        paths.preprocessing_path.write_text(
            json.dumps(
                preprocessing_fit.state,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        model_sha256 = _sha256_file(
            paths.model_path
        )

        preprocessing_sha256 = (
            _sha256_file(
                paths.preprocessing_path
            )
        )

        manifest = _build_manifest(
            run_id=paths.run_id,
            paths=paths,
            model_fit=model_fit,
            preprocessing_fit=
                preprocessing_fit,
            model_sha256=
                model_sha256,
            preprocessing_sha256=
                preprocessing_sha256,
        )

        paths.manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        loaded_manifest = json.loads(
            paths.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if loaded_manifest != manifest:
            raise RuntimeError(
                "Rebuild manifest round-trip mismatch."
            )

        if (
            _sha256_file(
                paths.model_path
            )
            != model_sha256
        ):
            raise RuntimeError(
                "Rebuild model fingerprint changed after persistence."
            )

        if (
            _sha256_file(
                paths.preprocessing_path
            )
            != preprocessing_sha256
        ):
            raise RuntimeError(
                "Rebuild preprocessing fingerprint changed after persistence."
            )

        expected_files = {
            paths.model_path.name,
            paths.preprocessing_path.name,
            paths.manifest_path.name,
        }

        actual_files = {
            path.name
            for path
            in paths.run_dir.iterdir()
            if path.is_file()
        }

        if actual_files != expected_files:
            raise RuntimeError(
                "Unexpected file set in rebuild run directory."
            )

    except Exception:
        shutil.rmtree(
            paths.run_dir,
            ignore_errors=True,
        )
        raise

    return RebuildArtifactSet(
        paths=paths,
        model_sha256=model_sha256,
        preprocessing_sha256=
            preprocessing_sha256,
        manifest=manifest,
    )
