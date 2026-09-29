from __future__ import annotations

import argparse
from pathlib import Path

INIT_REL = Path(
    "final_pipeline/src/fraud_screening/training/__init__.py"
)

EXPECTED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n    RebuildPreprocessingFitter,\n    RebuildPreprocessor,\n    fit_rebuild_preprocessor,\n)\nfrom fraud_screening.training.trainer import (\n    RebuildModelFit,\n    RebuildTrainingError,\n    fit_rebuild_model,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildModelFit",\n    "RebuildOutputPaths",\n    "RebuildPreprocessingFit",\n    "RebuildPreprocessingFitter",\n    "RebuildPreprocessor",\n    "RebuildTrainingError",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "fit_rebuild_model",\n    "fit_rebuild_preprocessor",\n    "validate_run_id",\n]\n'
UPDATED_INIT = '"""Rebuild-only training utilities.\n\nThis package creates new learned state only for explicit rebuild runs.\nIt never writes to the official artifact directory.\n"""\n\nfrom fraud_screening.training.factory import (\n    REBUILD_MODEL_ID,\n    build_rebuild_model,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n    validate_run_id,\n)\nfrom fraud_screening.training.persistence import (\n    RebuildArtifactSet,\n    persist_rebuild_artifacts,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n    RebuildPreprocessingFitter,\n    RebuildPreprocessor,\n    fit_rebuild_preprocessor,\n)\nfrom fraud_screening.training.trainer import (\n    RebuildModelFit,\n    RebuildTrainingError,\n    fit_rebuild_model,\n)\n\n__all__ = [\n    "REBUILD_MODEL_ID",\n    "RebuildArtifactSet",\n    "RebuildModelFit",\n    "RebuildOutputPaths",\n    "RebuildPreprocessingFit",\n    "RebuildPreprocessingFitter",\n    "RebuildPreprocessor",\n    "RebuildTrainingError",\n    "build_rebuild_model",\n    "create_rebuild_output_paths",\n    "fit_rebuild_model",\n    "fit_rebuild_preprocessor",\n    "persist_rebuild_artifacts",\n    "validate_run_id",\n]\n'

NEW_FILES = {
    "final_pipeline/src/fraud_screening/training/persistence.py":
        '"""Guarded persistence for newly learned rebuild artifacts."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nimport hashlib\nimport json\nfrom pathlib import Path\nimport shutil\n\nimport joblib\n\nfrom fraud_screening.artifacts import (\n    EXPECTED_MODEL_PARAMETERS,\n)\nfrom fraud_screening.training.outputs import (\n    RebuildOutputPaths,\n    create_rebuild_output_paths,\n)\nfrom fraud_screening.training.preprocessing_fit import (\n    RebuildPreprocessingFit,\n)\nfrom fraud_screening.training.trainer import (\n    RebuildModelFit,\n)\n\n\n@dataclass(frozen=True, slots=True)\nclass RebuildArtifactSet:\n    """Persisted files and fingerprints for one isolated rebuild run."""\n\n    paths: RebuildOutputPaths\n    model_sha256: str\n    preprocessing_sha256: str\n    manifest: dict\n\n\ndef _sha256_file(\n    path: Path,\n) -> str:\n    digest = hashlib.sha256()\n\n    with path.open("rb") as handle:\n        for chunk in iter(\n            lambda: handle.read(\n                1024 * 1024\n            ),\n            b"",\n        ):\n            digest.update(chunk)\n\n    return digest.hexdigest()\n\n\ndef _guard_fit_status(\n    model_fit: RebuildModelFit,\n    preprocessing_fit: RebuildPreprocessingFit,\n) -> None:\n    if (\n        model_fit.artifact_identity_status\n        != "NEW_REBUILD_MODEL"\n    ):\n        raise ValueError(\n            "Model fit is not marked as a new rebuild artifact."\n        )\n\n    if (\n        model_fit.evaluation_status\n        != "NOT_EVALUATED_AS_OFFICIAL"\n    ):\n        raise ValueError(\n            "Rebuild model may not claim official evaluation status."\n        )\n\n    if (\n        preprocessing_fit.state.get(\n            "rebuild_validation_status"\n        )\n        != "NOT_COMPARED_TO_OFFICIAL_ARTIFACT"\n    ):\n        raise ValueError(\n            "Rebuild preprocessing may not claim official equivalence."\n        )\n\n    if (\n        preprocessing_fit.state.get(\n            "feature_count"\n        )\n        != 47\n    ):\n        raise ValueError(\n            "Rebuild preprocessing feature count must be 47."\n        )\n\n    if (\n        model_fit.feature_count\n        != 47\n    ):\n        raise ValueError(\n            "Rebuild model feature count must be 47."\n        )\n\n    if (\n        model_fit.classes\n        != (0, 1)\n    ):\n        raise ValueError(\n            "Rebuild model classes must be exactly (0, 1)."\n        )\n\n    actual_params = (\n        model_fit.estimator\n        .get_params(\n            deep=False\n        )\n    )\n\n    for key, expected in (\n        EXPECTED_MODEL_PARAMETERS.items()\n    ):\n        if actual_params[key] != expected:\n            raise ValueError(\n                f"Rebuild model parameter mismatch: {key}."\n            )\n\n\ndef _build_manifest(\n    *,\n    run_id: str,\n    paths: RebuildOutputPaths,\n    model_fit: RebuildModelFit,\n    preprocessing_fit: RebuildPreprocessingFit,\n    model_sha256: str,\n    preprocessing_sha256: str,\n) -> dict:\n    return {\n        "manifest_version":\n            "rebuild-1.0",\n        "run_id":\n            run_id,\n        "artifact_identity":\n            "NEW_REBUILD_ARTIFACT_SET",\n        "official_artifact_replacement":\n            False,\n        "official_metrics_inherited":\n            False,\n        "model": {\n            "file":\n                paths.model_path.name,\n            "sha256":\n                model_sha256,\n            "model_id":\n                model_fit.model_id,\n            "artifact_identity_status":\n                model_fit.artifact_identity_status,\n            "evaluation_status":\n                model_fit.evaluation_status,\n            "training_rows":\n                model_fit.training_rows,\n            "fraud_rows":\n                model_fit.fraud_rows,\n            "feature_count":\n                model_fit.feature_count,\n            "classes":\n                list(\n                    model_fit.classes\n                ),\n            "parameters": {\n                key:\n                    model_fit.estimator\n                    .get_params(\n                        deep=False\n                    )[key]\n                for key\n                in EXPECTED_MODEL_PARAMETERS\n            },\n        },\n        "preprocessing": {\n            "file":\n                paths.preprocessing_path.name,\n            "sha256":\n                preprocessing_sha256,\n            "strategy":\n                preprocessing_fit.state[\n                    "strategy"\n                ],\n            "fit_source":\n                preprocessing_fit.state[\n                    "fit_source"\n                ],\n            "fit_row_count":\n                preprocessing_fit.state[\n                    "fit_row_count"\n                ],\n            "fit_min_timestamp":\n                preprocessing_fit.state[\n                    "fit_min_timestamp"\n                ],\n            "fit_max_timestamp":\n                preprocessing_fit.state[\n                    "fit_max_timestamp"\n                ],\n            "feature_count":\n                preprocessing_fit.state[\n                    "feature_count"\n                ],\n            "rebuild_validation_status":\n                preprocessing_fit.state[\n                    "rebuild_validation_status"\n                ],\n        },\n        "safety": {\n            "official_path_written":\n                False,\n            "research_path_written":\n                False,\n            "silent_overwrite":\n                False,\n        },\n    }\n\n\ndef persist_rebuild_artifacts(\n    project_root: str | Path,\n    run_id: str,\n    *,\n    model_fit: RebuildModelFit,\n    preprocessing_fit: RebuildPreprocessingFit,\n) -> RebuildArtifactSet:\n    """Persist one rebuild artifact set under a new unique run directory."""\n\n    _guard_fit_status(\n        model_fit,\n        preprocessing_fit,\n    )\n\n    paths = create_rebuild_output_paths(\n        project_root,\n        run_id,\n        create=True,\n    )\n\n    try:\n        joblib.dump(\n            model_fit.estimator,\n            paths.model_path,\n        )\n\n        paths.preprocessing_path.write_text(\n            json.dumps(\n                preprocessing_fit.state,\n                ensure_ascii=False,\n                indent=2,\n                sort_keys=True,\n            )\n            + "\\n",\n            encoding="utf-8",\n        )\n\n        model_sha256 = _sha256_file(\n            paths.model_path\n        )\n\n        preprocessing_sha256 = (\n            _sha256_file(\n                paths.preprocessing_path\n            )\n        )\n\n        manifest = _build_manifest(\n            run_id=paths.run_id,\n            paths=paths,\n            model_fit=model_fit,\n            preprocessing_fit=\n                preprocessing_fit,\n            model_sha256=\n                model_sha256,\n            preprocessing_sha256=\n                preprocessing_sha256,\n        )\n\n        paths.manifest_path.write_text(\n            json.dumps(\n                manifest,\n                ensure_ascii=False,\n                indent=2,\n                sort_keys=True,\n            )\n            + "\\n",\n            encoding="utf-8",\n        )\n\n        loaded_manifest = json.loads(\n            paths.manifest_path.read_text(\n                encoding="utf-8"\n            )\n        )\n\n        if loaded_manifest != manifest:\n            raise RuntimeError(\n                "Rebuild manifest round-trip mismatch."\n            )\n\n        if (\n            _sha256_file(\n                paths.model_path\n            )\n            != model_sha256\n        ):\n            raise RuntimeError(\n                "Rebuild model fingerprint changed after persistence."\n            )\n\n        if (\n            _sha256_file(\n                paths.preprocessing_path\n            )\n            != preprocessing_sha256\n        ):\n            raise RuntimeError(\n                "Rebuild preprocessing fingerprint changed after persistence."\n            )\n\n        expected_files = {\n            paths.model_path.name,\n            paths.preprocessing_path.name,\n            paths.manifest_path.name,\n        }\n\n        actual_files = {\n            path.name\n            for path\n            in paths.run_dir.iterdir()\n            if path.is_file()\n        }\n\n        if actual_files != expected_files:\n            raise RuntimeError(\n                "Unexpected file set in rebuild run directory."\n            )\n\n    except Exception:\n        shutil.rmtree(\n            paths.run_dir,\n            ignore_errors=True,\n        )\n        raise\n\n    return RebuildArtifactSet(\n        paths=paths,\n        model_sha256=model_sha256,\n        preprocessing_sha256=\n            preprocessing_sha256,\n        manifest=manifest,\n    )\n',
    "final_pipeline/tests/unit/test_rebuild_artifact_persistence.py":
        'from __future__ import annotations\n\nfrom datetime import datetime, timedelta\nimport json\nfrom pathlib import Path\nimport tempfile\nimport unittest\nfrom unittest.mock import patch\n\nimport joblib\nimport numpy as np\n\nfrom fraud_screening.features import (\n    SemanticFeatureRow,\n)\nfrom fraud_screening.training import (\n    fit_rebuild_model,\n    fit_rebuild_preprocessor,\n    persist_rebuild_artifacts,\n)\n\n\ndef semantic_fixture() -> tuple[\n    list[SemanticFeatureRow],\n    list[datetime],\n]:\n    modes = [\n        "Chip Transaction",\n        "Online Transaction",\n        "Swipe Transaction",\n    ]\n\n    locations = [\n        "NON_PHYSICAL_OR_ONLINE",\n        "PHYSICAL_COMPLETE",\n        "PHYSICAL_ZIP_UNAVAILABLE",\n    ]\n\n    rows = []\n    timestamps = []\n    start = datetime(\n        2018,\n        1,\n        1,\n    )\n\n    for index in range(24):\n        rows.append(\n            SemanticFeatureRow(\n                amount_numeric=\n                    float(index + 1),\n                time_since_previous_transaction_min=\n                    None\n                    if index == 0\n                    else float(index + 1),\n                transactions_last_1h=\n                    index % 3,\n                amount_minus_previous_mean=\n                    None\n                    if index == 0\n                    else float(index - 3),\n                is_new_merchant=\n                    index < 4,\n                has_prior_card_history=\n                    index != 0,\n                transaction_mode=\n                    modes[\n                        index % 3\n                    ],\n                location_state=\n                    locations[\n                        index % 3\n                    ],\n                hour_of_day=\n                    str(index),\n                day_of_week=\n                    str(index % 7),\n            )\n        )\n\n        timestamps.append(\n            start\n            + timedelta(\n                hours=index\n            )\n        )\n\n    return rows, timestamps\n\n\ndef fitted_pair():\n    rows, timestamps = (\n        semantic_fixture()\n    )\n\n    preprocessing_fit = (\n        fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n    )\n\n    matrix = (\n        preprocessing_fit\n        .preprocessor\n        .transform(\n            rows\n        )\n    )\n\n    target = np.asarray(\n        [\n            1\n            if index % 6 == 0\n            else 0\n            for index\n            in range(len(rows))\n        ],\n        dtype=np.int8,\n    )\n\n    target[0] = 0\n    target[1] = 1\n\n    model_fit = fit_rebuild_model(\n        matrix,\n        target,\n    )\n\n    return (\n        model_fit,\n        preprocessing_fit,\n    )\n\n\nclass RebuildArtifactPersistenceTests(\n    unittest.TestCase\n):\n    def test_persist_writes_exact_three_files_under_rebuild(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            result = (\n                persist_rebuild_artifacts(\n                    root,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            self.assertEqual(\n                {\n                    path.name\n                    for path\n                    in result.paths.run_dir.iterdir()\n                    if path.is_file()\n                },\n                {\n                    "model.joblib",\n                    "preprocessing_state.json",\n                    "rebuild_manifest.json",\n                },\n            )\n\n            self.assertIn(\n                "final_pipeline/outputs/rebuilds/run-001",\n                result.paths.run_dir.as_posix(),\n            )\n\n    def test_manifest_explicitly_denies_official_identity_and_metrics(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            result = (\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            manifest = result.manifest\n\n            self.assertFalse(\n                manifest[\n                    "official_artifact_replacement"\n                ]\n            )\n            self.assertFalse(\n                manifest[\n                    "official_metrics_inherited"\n                ]\n            )\n            self.assertEqual(\n                manifest[\n                    "model"\n                ][\n                    "evaluation_status"\n                ],\n                "NOT_EVALUATED_AS_OFFICIAL",\n            )\n\n    def test_manifest_fingerprints_match_persisted_files(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            result = (\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            manifest = json.loads(\n                result.paths.manifest_path\n                .read_text(\n                    encoding="utf-8"\n                )\n            )\n\n            self.assertEqual(\n                manifest[\n                    "model"\n                ][\n                    "sha256"\n                ],\n                result.model_sha256,\n            )\n            self.assertEqual(\n                manifest[\n                    "preprocessing"\n                ][\n                    "sha256"\n                ],\n                result.preprocessing_sha256,\n            )\n\n    def test_persisted_model_round_trips_as_fitted_estimator(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            result = (\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            loaded = joblib.load(\n                result.paths.model_path\n            )\n\n            self.assertEqual(\n                loaded.classes_.tolist(),\n                [0, 1],\n            )\n            self.assertEqual(\n                int(\n                    loaded.n_features_in_\n                ),\n                47,\n            )\n            self.assertEqual(\n                len(\n                    loaded.estimators_\n                ),\n                100,\n            )\n\n    def test_existing_run_is_not_overwritten(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            first = (\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            original_manifest = (\n                first.paths.manifest_path\n                .read_text(\n                    encoding="utf-8"\n                )\n            )\n\n            with self.assertRaises(\n                FileExistsError\n            ):\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n\n            self.assertEqual(\n                first.paths.manifest_path\n                .read_text(\n                    encoding="utf-8"\n                ),\n                original_manifest,\n            )\n\n    def test_official_directory_is_never_created(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            persist_rebuild_artifacts(\n                root,\n                "run-001",\n                model_fit=\n                    model_fit,\n                preprocessing_fit=\n                    preprocessing_fit,\n            )\n\n            self.assertFalse(\n                (\n                    root\n                    / "final_pipeline"\n                    / "artifacts"\n                    / "official"\n                ).exists()\n            )\n\n    def test_persistence_failure_rolls_back_run_directory(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n\n            with patch(\n                "fraud_screening.training.persistence.joblib.dump",\n                side_effect=RuntimeError(\n                    "synthetic write failure"\n                ),\n            ):\n                with self.assertRaises(\n                    RuntimeError\n                ):\n                    persist_rebuild_artifacts(\n                        root,\n                        "run-001",\n                        model_fit=\n                            model_fit,\n                        preprocessing_fit=\n                            preprocessing_fit,\n                    )\n\n            self.assertFalse(\n                (\n                    root\n                    / "final_pipeline"\n                    / "outputs"\n                    / "rebuilds"\n                    / "run-001"\n                ).exists()\n            )\n\n    def test_rebuild_preprocessing_status_is_preserved_in_manifest(self) -> None:\n        model_fit, preprocessing_fit = (\n            fitted_pair()\n        )\n\n        with tempfile.TemporaryDirectory() as tmp:\n            result = (\n                persist_rebuild_artifacts(\n                    tmp,\n                    "run-001",\n                    model_fit=\n                        model_fit,\n                    preprocessing_fit=\n                        preprocessing_fit,\n                )\n            )\n\n            self.assertEqual(\n                result.manifest[\n                    "preprocessing"\n                ][\n                    "rebuild_validation_status"\n                ],\n                "NOT_COMPARED_TO_OFFICIAL_ARTIFACT",\n            )\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
}


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
        root
        / "final_pipeline/src/fraud_screening/training/factory.py",
        root
        / "final_pipeline/src/fraud_screening/training/outputs.py",
        root
        / "final_pipeline/src/fraud_screening/training/preprocessing_fit.py",
        root
        / "final_pipeline/src/fraud_screening/training/trainer.py",
        root
        / "final_pipeline/artifacts/official/model/model.joblib",
        root
        / "final_pipeline/artifacts/official/preprocessing/preprocessing_state.json",
        root
        / "final_pipeline/artifacts/official/manifest/artifact_manifest.json",
    ]

    if not all(
        path.exists()
        for path in required
    ):
        raise RuntimeError(
            "Repository training foundation or official artifact set is incomplete."
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Create rebuild persistence source/tests. "
            "The bootstrap itself writes no rebuild artifact."
        ),
    )

    args = parser.parse_args()

    root = detect_root()
    init_path = root / INIT_REL

    missing_dependencies = [
        relative
        for relative in [
            "final_pipeline/src/fraud_screening/training/factory.py",
            "final_pipeline/src/fraud_screening/training/outputs.py",
            "final_pipeline/src/fraud_screening/training/preprocessing_fit.py",
            "final_pipeline/src/fraud_screening/training/trainer.py",
        ]
        if not (
            root / relative
        ).is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (
            root / relative
        ).exists()
    ]

    init_matches = (
        init_path.is_file()
        and init_path.read_text(
            encoding="utf-8"
        )
        == EXPECTED_INIT
    )

    print("=" * 92)
    print(
        "REBUILD ARTIFACT PERSISTENCE — GUARDED NEW-OUTPUT CONTRACT"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED SOURCE UPDATE"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(
        " - missing dependencies:",
        len(
            missing_dependencies
        ),
    )
    for relative in missing_dependencies:
        print("   *", relative)

    print("\n[2] training/__init__.py guard")
    print(
        " - exact expected current content:",
        "YES"
        if init_matches
        else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] Collision gate")
    print(
        " - collisions:",
        len(collisions),
    )
    for relative in collisions:
        print("   *", relative)

    print("\n[5] Rebuild persistence contract")
    print(
        " - destination: final_pipeline/outputs/rebuilds/<run_id>/"
    )
    print(
        " - files: model.joblib + preprocessing_state.json + rebuild_manifest.json"
    )
    print(
        " - fingerprints: SHA-256 for model and preprocessing"
    )
    print(
        " - existing run overwrite: FORBIDDEN"
    )
    print(
        " - failed persistence rollback: REQUIRED"
    )
    print(
        " - official artifact replacement: FALSE"
    )
    print(
        " - official M8 metrics inheritance: FALSE"
    )

    print("\n[6] Side-effect boundary")
    print(
        " - dry-run model.fit(): NO"
    )
    print(
        " - execute model.fit(): NO"
    )
    print(
        " - bootstrap rebuild artifact write: NO"
    )
    print(
        " - unit tests after execute: YES, temp-directory only"
    )
    print(
        " - unit-test model.fit(): YES, small synthetic fixture"
    )
    print(
        " - unit-test rebuild artifact write: YES, temporary directory only"
    )
    print(
        " - project outputs/rebuilds write during tests: NO"
    )
    print(
        " - official artifact modification: NO"
    )
    print(
        " - research modification: NO"
    )
    print(
        " - predict()/predict_proba(): NO"
    )

    if (
        missing_dependencies
        or collisions
        or not init_matches
    ):
        print("\nRESULT: STOP")
        print(
            "No files were written."
        )
        raise SystemExit(1)

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_rebuild_persistence.py --execute"
        )
        print("=" * 92)
        return

    created = []
    original_init = (
        init_path.read_text(
            encoding="utf-8"
        )
    )

    try:
        for relative, content in (
            NEW_FILES.items()
        ):
            path = root / relative

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            created.append(
                path
            )

        init_path.write_text(
            UPDATED_INIT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(
            created
        ):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            init_path.write_text(
                original_init,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[7] Created")
    for path in created:
        print(
            " -",
            path.relative_to(root),
        )

    print("\n[8] Updated")
    print(" -", INIT_REL)

    print("\nBOOTSTRAP RESULT: PASS")
    print(
        "Persistence source was created without training or artifact writes."
    )
    print(
        "Run full unit suite next. Persistence tests write only under "
        "temporary test directories and remove them automatically."
    )
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
