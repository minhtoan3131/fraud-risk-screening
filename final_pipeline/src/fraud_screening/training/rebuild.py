"""Orchestration of canonical rebuild training from a validated bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scipy import sparse

from fraud_screening.training.bundle import (
    CANONICAL_W_SHORT_PROFILE,
    TrainingBundleProfile,
    load_training_bundle,
)
from fraud_screening.training.outputs import (
    create_rebuild_output_paths,
)
from fraud_screening.training.persistence import (
    RebuildArtifactSet,
    persist_rebuild_artifacts,
)
from fraud_screening.training.preprocessing_fit import (
    RebuildPreprocessingFitter,
)
from fraud_screening.training.trainer import (
    RebuildModelFit,
    fit_rebuild_model,
)


@dataclass(frozen=True, slots=True)
class RebuildRunResult:
    """Finished rebuild result written to one isolated run directory."""

    artifact_set: RebuildArtifactSet
    model_fit: RebuildModelFit


def run_rebuild_training(
    project_root: str | Path,
    bundle_dir: str | Path,
    run_id: str,
    *,
    batch_size: int = 50_000,
    profile: TrainingBundleProfile = CANONICAL_W_SHORT_PROFILE,
) -> RebuildRunResult:
    """Fit preprocessing + model and persist a new rebuild artifact set."""

    root = Path(
        project_root
    ).resolve()

    # Collision/path guard before creating learned state.
    create_rebuild_output_paths(
        root,
        run_id,
        create=False,
    )

    bundle = load_training_bundle(
        bundle_dir,
        profile=profile,
        project_root=root,
    )

    fitter = (
        RebuildPreprocessingFitter()
    )

    for rows, timestamps in (
        bundle.iter_semantic_batches(
            batch_size
        )
    ):
        fitter.partial_fit(
            rows,
            timestamps,
        )

    preprocessing_fit = (
        fitter.finalize()
    )

    if (
        preprocessing_fit.state[
            "fit_row_count"
        ]
        != bundle.row_count
    ):
        raise RuntimeError(
            "Rebuild preprocessing row count does not match bundle."
        )

    matrix_parts = []

    for rows, _ in (
        bundle.iter_semantic_batches(
            batch_size
        )
    ):
        matrix_parts.append(
            preprocessing_fit
            .preprocessor
            .transform(
                rows
            )
        )

    matrix = sparse.vstack(
        matrix_parts,
        format="csr",
    )

    del matrix_parts

    model_fit = fit_rebuild_model(
        matrix,
        bundle.target,
    )

    if (
        model_fit.training_rows
        != bundle.row_count
    ):
        raise RuntimeError(
            "Rebuild model row count does not match bundle."
        )

    if (
        model_fit.fraud_rows
        != bundle.fraud_rows
    ):
        raise RuntimeError(
            "Rebuild model fraud count does not match bundle."
        )

    artifact_set = (
        persist_rebuild_artifacts(
            root,
            run_id,
            model_fit=
                model_fit,
            preprocessing_fit=
                preprocessing_fit,
        )
    )

    return RebuildRunResult(
        artifact_set=
            artifact_set,
        model_fit=
            model_fit,
    )
