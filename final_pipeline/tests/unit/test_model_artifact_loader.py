from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
    MODEL_ID,
    POSITIVE_CLASS,
    load_verified_model,
)
from fraud_screening.artifacts.model_loader import (
    sha256_file,
)
from fraud_screening.errors import (
    ArtifactCompatibilityError,
    ArtifactFingerprintError,
    ArtifactNotFoundError,
)


def compatible_estimator() -> RandomForestClassifier:
    estimator = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight="balanced",
        criterion="gini",
        max_depth=None,
        max_features="sqrt",
        max_samples=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
    )

    # Identity tests do not train. Only the fitted-class metadata required
    # by the loader is attached to this temporary compatibility fixture.
    estimator.classes_ = np.asarray(
        [0, 1],
        dtype=np.int64,
    )

    return estimator


class ModelArtifactLoaderTests(unittest.TestCase):
    def dump_fixture(
        self,
        directory: str,
        estimator: object,
    ) -> Path:
        path = Path(directory) / "model.joblib"
        joblib.dump(estimator, path)
        return path

    def test_expected_model_id_is_stable(self) -> None:
        self.assertEqual(
            MODEL_ID,
            "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
        )
        self.assertEqual(
            POSITIVE_CLASS,
            1,
        )

    def test_expected_parameter_contract_is_exact(self) -> None:
        self.assertEqual(
            EXPECTED_MODEL_PARAMETERS,
            {
                "bootstrap": True,
                "ccp_alpha": 0.0,
                "class_weight": "balanced",
                "criterion": "gini",
                "max_depth": None,
                "max_features": "sqrt",
                "max_samples": None,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "n_estimators": 100,
                "n_jobs": -1,
                "random_state": 42,
            },
        )

    def test_verified_loader_accepts_compatible_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                compatible_estimator(),
            )
            digest = sha256_file(path)

            artifact = load_verified_model(
                path,
                expected_sha256=digest,
            )

            self.assertEqual(
                artifact.model_id,
                MODEL_ID,
            )
            self.assertEqual(
                artifact.classes,
                (0, 1),
            )
            self.assertEqual(
                artifact.positive_class_index,
                1,
            )
            self.assertEqual(
                artifact.positive_class,
                1,
            )

    def test_sha_mismatch_is_rejected_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                compatible_estimator(),
            )

            with self.assertRaises(
                ArtifactFingerprintError
            ):
                load_verified_model(
                    path,
                    expected_sha256="0" * 64,
                )

    def test_missing_file_raises_domain_error(self) -> None:
        with self.assertRaises(
            ArtifactNotFoundError
        ):
            load_verified_model(
                "/definitely/missing/model.joblib",
                expected_sha256="0" * 64,
            )

    def test_wrong_estimator_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                {"not": "a model"},
            )
            digest = sha256_file(path)

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_verified_model(
                    path,
                    expected_sha256=digest,
                )

    def test_parameter_mismatch_is_rejected(self) -> None:
        estimator = compatible_estimator()
        estimator.set_params(
            n_estimators=101
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                estimator,
            )
            digest = sha256_file(path)

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_verified_model(
                    path,
                    expected_sha256=digest,
                )

    def test_missing_classes_is_rejected(self) -> None:
        estimator = compatible_estimator()
        del estimator.classes_

        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                estimator,
            )
            digest = sha256_file(path)

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_verified_model(
                    path,
                    expected_sha256=digest,
                )

    def test_wrong_classes_are_rejected(self) -> None:
        estimator = compatible_estimator()
        estimator.classes_ = np.asarray(
            [0, 2],
            dtype=np.int64,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                estimator,
            )
            digest = sha256_file(path)

            with self.assertRaises(
                ArtifactCompatibilityError
            ):
                load_verified_model(
                    path,
                    expected_sha256=digest,
                )

    def test_loader_does_not_expose_prediction_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.dump_fixture(
                tmp,
                compatible_estimator(),
            )
            digest = sha256_file(path)

            artifact = load_verified_model(
                path,
                expected_sha256=digest,
            )

            self.assertFalse(
                hasattr(
                    artifact,
                    "predict",
                )
            )
            self.assertFalse(
                hasattr(
                    artifact,
                    "predict_proba",
                )
            )


if __name__ == "__main__":
    unittest.main()
