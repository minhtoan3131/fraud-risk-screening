from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fraud_screening.training import (
    run_rebuild_training,
)

from unit.test_rebuild_training_bundle import (
    TEST_PROFILE,
    write_bundle,
)


class RebuildTrainingOrchestrationTests(
    unittest.TestCase
):
    def test_full_small_rebuild_writes_isolated_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            project_root = (
                base
                / "project"
            )
            project_root.mkdir()

            bundle_dir = write_bundle(
                base
            )

            result = run_rebuild_training(
                project_root,
                bundle_dir,
                "test-run-001",
                batch_size=7,
                profile=
                    TEST_PROFILE,
            )

            run_dir = (
                result
                .artifact_set
                .paths
                .run_dir
            )

            self.assertTrue(
                run_dir.is_dir()
            )
            self.assertEqual(
                result.model_fit.training_rows,
                24,
            )
            self.assertEqual(
                result.model_fit.fraud_rows,
                4,
            )
            self.assertFalse(
                (
                    project_root
                    / "final_pipeline"
                    / "artifacts"
                    / "official"
                ).exists()
            )

    def test_existing_run_stops_before_new_training_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            project_root = (
                base
                / "project"
            )
            project_root.mkdir()

            bundle_dir = write_bundle(
                base
            )

            run_rebuild_training(
                project_root,
                bundle_dir,
                "test-run-001",
                batch_size=8,
                profile=
                    TEST_PROFILE,
            )

            with self.assertRaises(
                FileExistsError
            ):
                run_rebuild_training(
                    project_root,
                    bundle_dir,
                    "test-run-001",
                    batch_size=8,
                    profile=
                        TEST_PROFILE,
                )


if __name__ == "__main__":
    unittest.main()
