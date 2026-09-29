from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fraud_screening.artifacts.model_loader import (
    MODEL_ID,
    POSITIVE_CLASS,
    VerifiedModelArtifact,
)
from fraud_screening.artifacts.official_loader import (
    load_official_artifacts,
)
from fraud_screening.errors import (
    ArtifactCompatibilityError,
)


MODEL_SHA = "a" * 64
PREPROCESSING_SHA = "b" * 64


class _DummyEstimator:
    pass


class _DummyPreprocessor:
    feature_names = tuple(
        f"f{i}"
        for i in range(47)
    )


def manifest(
    *,
    model_file: str = "model/model.joblib",
    preprocessing_file: str = (
        "preprocessing/preprocessing_state.json"
    ),
) -> dict:
    return {
        "manifest_version": "1.0",
        "model": {
            "classes": [0, 1],
            "file": model_file,
            "model_id": MODEL_ID,
            "positive_class": POSITIVE_CLASS,
            "positive_class_index": 1,
            "sha256": MODEL_SHA,
        },
        "preprocessing": {
            "encoded_feature_count": 47,
            "file": preprocessing_file,
            "matrix_dtype": "float32",
            "matrix_format": "CSR",
            "semantic_feature_count": 10,
            "sha256": PREPROCESSING_SHA,
        },
        "runtime_policy": {
            "official_artifacts_read_only": True,
            "silent_research_fallback": False,
            "training_may_overwrite_official": False,
        },
        "screening": {
            "risk_score_interface":
                "predict_proba positive-class score",
            "threshold": 0.5,
            "threshold_comparator": ">",
        },
    }


def write_bundle(
    root: Path,
    payload: dict,
) -> None:
    manifest_path = (
        root
        / "manifest"
        / "artifact_manifest.json"
    )
    manifest_path.parent.mkdir(
        parents=True
    )
    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    model_path = root / "model/model.joblib"
    model_path.parent.mkdir(
        parents=True
    )
    model_path.write_bytes(
        b"model"
    )

    preprocessing_path = (
        root
        / "preprocessing"
        / "preprocessing_state.json"
    )
    preprocessing_path.parent.mkdir(
        parents=True
    )
    preprocessing_path.write_text(
        "{}",
        encoding="utf-8",
    )


class OfficialArtifactBundleTests(
    unittest.TestCase
):
    def test_valid_manifest_composes_existing_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_bundle(
                root,
                manifest(),
            )

            verified_model = VerifiedModelArtifact(
                estimator=_DummyEstimator(),
                sha256=MODEL_SHA,
                model_id=MODEL_ID,
                classes=(0, 1),
                positive_class_index=1,
            )

            with (
                patch(
                    "fraud_screening.artifacts.official_loader.load_verified_model",
                    return_value=verified_model,
                ) as model_loader,
                patch(
                    "fraud_screening.artifacts.official_loader.sha256_file",
                    return_value=PREPROCESSING_SHA,
                ),
                patch(
                    "fraud_screening.artifacts.official_loader.load_frozen_preprocessor",
                    return_value=_DummyPreprocessor(),
                ) as preprocessing_loader,
            ):
                bundle = load_official_artifacts(
                    root
                )

            self.assertEqual(
                bundle.model,
                verified_model,
            )
            self.assertEqual(
                bundle.manifest_version,
                "1.0",
            )

            model_loader.assert_called_once()
            preprocessing_loader.assert_called_once()

    def test_manifest_threshold_comparator_must_be_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = manifest()
            payload["screening"][
                "threshold_comparator"
            ] = ">="

            write_bundle(
                root,
                payload,
            )

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_official_artifacts(
                    root
                )

    def test_manifest_runtime_policy_must_be_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = manifest()
            payload["runtime_policy"][
                "official_artifacts_read_only"
            ] = False

            write_bundle(
                root,
                payload,
            )

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_official_artifacts(
                    root
                )

    def test_manifest_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "official"
            root.mkdir()

            payload = manifest(
                model_file="../outside.joblib"
            )

            write_bundle(
                root,
                payload,
            )

            outside = (
                root.parent
                / "outside.joblib"
            )
            outside.write_bytes(
                b"outside"
            )

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_official_artifacts(
                    root
                )

    def test_manifest_model_identity_must_match_frozen_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = manifest()
            payload["model"]["model_id"] = "OTHER"

            write_bundle(
                root,
                payload,
            )

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_official_artifacts(
                    root
                )


if __name__ == "__main__":
    unittest.main()
