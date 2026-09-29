from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fraud_screening.training import (
    create_rebuild_output_paths,
    validate_run_id,
)


class RebuildOutputPolicyTests(unittest.TestCase):
    def test_valid_run_id_is_preserved(self) -> None:
        self.assertEqual(
            validate_run_id(
                "rebuild-20260922-001"
            ),
            "rebuild-20260922-001",
        )

    def test_empty_run_id_is_rejected(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            validate_run_id("   ")

    def test_path_separator_is_rejected(self) -> None:
        for value in [
            "../escape",
            "a/b",
            r"a\b",
        ]:
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    validate_run_id(
                        value
                    )

    def test_reserved_official_term_is_rejected(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            validate_run_id(
                "official"
            )

    def test_reserved_artifacts_term_is_rejected(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            validate_run_id(
                "rebuild-artifacts-001"
            )

    def test_reserved_research_term_is_rejected(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            validate_run_id(
                "research-copy"
            )

    def test_paths_are_under_rebuild_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paths = create_rebuild_output_paths(
                root,
                "run-001",
            )

            expected = (
                root.resolve()
                / "final_pipeline"
                / "outputs"
                / "rebuilds"
                / "run-001"
            )

            self.assertEqual(
                paths.run_dir,
                expected,
            )
            self.assertEqual(
                paths.model_path,
                expected
                / "model.joblib",
            )
            self.assertEqual(
                paths.preprocessing_path,
                expected
                / "preprocessing_state.json",
            )
            self.assertEqual(
                paths.manifest_path,
                expected
                / "rebuild_manifest.json",
            )

    def test_resolution_is_read_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paths = create_rebuild_output_paths(
                root,
                "run-001",
            )

            self.assertFalse(
                paths.run_dir.exists()
            )

    def test_create_makes_only_rebuild_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paths = create_rebuild_output_paths(
                root,
                "run-001",
                create=True,
            )

            self.assertTrue(
                paths.run_dir.is_dir()
            )
            self.assertFalse(
                paths.model_path.exists()
            )

            official = (
                root
                / "final_pipeline"
                / "artifacts"
                / "official"
            )

            self.assertFalse(
                official.exists()
            )

    def test_existing_run_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            first = create_rebuild_output_paths(
                root,
                "run-001",
                create=True,
            )

            marker = (
                first.run_dir
                / "keep.txt"
            )
            marker.write_text(
                "preserve",
                encoding="utf-8",
            )

            with self.assertRaises(
                FileExistsError
            ):
                create_rebuild_output_paths(
                    root,
                    "run-001",
                    create=True,
                )

            self.assertEqual(
                marker.read_text(
                    encoding="utf-8"
                ),
                "preserve",
            )

    def test_returned_file_set_has_no_official_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paths = create_rebuild_output_paths(
                root,
                "run-001",
            )

            for path in (
                paths.all_files()
            ):
                self.assertNotIn(
                    "artifacts/official",
                    path.as_posix(),
                )


if __name__ == "__main__":
    unittest.main()
