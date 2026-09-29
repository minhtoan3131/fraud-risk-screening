from __future__ import annotations

import unittest

from sklearn.ensemble import RandomForestClassifier

from fraud_screening.artifacts import (
    EXPECTED_MODEL_PARAMETERS,
    MODEL_ID,
)
from fraud_screening.training import (
    REBUILD_MODEL_ID,
    build_rebuild_model,
)


class RebuildModelFactoryTests(unittest.TestCase):
    def test_factory_returns_random_forest(self) -> None:
        estimator = build_rebuild_model()

        self.assertIsInstance(
            estimator,
            RandomForestClassifier,
        )

    def test_factory_model_id_matches_official_identity(self) -> None:
        self.assertEqual(
            REBUILD_MODEL_ID,
            MODEL_ID,
        )

    def test_factory_matches_guarded_final_parameters(self) -> None:
        estimator = build_rebuild_model()
        actual = estimator.get_params(
            deep=False
        )

        for key, expected in (
            EXPECTED_MODEL_PARAMETERS.items()
        ):
            self.assertEqual(
                actual[key],
                expected,
                msg=key,
            )

    def test_factory_returns_unfitted_model(self) -> None:
        estimator = build_rebuild_model()

        self.assertFalse(
            hasattr(
                estimator,
                "classes_",
            )
        )
        self.assertFalse(
            hasattr(
                estimator,
                "estimators_",
            )
        )

    def test_factory_creates_fresh_instances(self) -> None:
        first = build_rebuild_model()
        second = build_rebuild_model()

        self.assertIsNot(
            first,
            second,
        )


if __name__ == "__main__":
    unittest.main()
