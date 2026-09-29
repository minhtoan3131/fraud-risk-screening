from __future__ import annotations

import argparse
from pathlib import Path


PREPROCESSOR_REL = Path(
    "final_pipeline/src/fraud_screening/training/preprocessing_fit.py"
)
PREPROCESSOR_TEST_REL = Path(
    "final_pipeline/tests/unit/test_rebuild_preprocessing_fit.py"
)
BOOTSTRAP_REL = Path(
    "bootstrap_rebuild_training_command.py"
)

PREPROCESSOR_OLD = '        if (\n            self._fit_max_timestamp is not None\n            and batch_min < self._fit_max_timestamp\n        ):\n            raise PreprocessingContractError(\n                "Rebuild preprocessing batches must be chronological."\n            )\n\n'
PREPROCESSOR_NEW = '        # No global chronological-batch requirement here.\n        #\n        # Canonical replay is ordered by contiguous User+Card blocks.\n        # Timestamp monotonicity belongs to each history block before\n        # semantic features are emitted. Preprocessing learned statistics\n        # are order-invariant across already-canonical semantic rows.\n\n'
TEST_OLD = '    def test_non_chronological_batch_is_rejected(self) -> None:\n        rows, timestamps = training_fixture()\n\n        fitter = (\n            RebuildPreprocessingFitter()\n        )\n\n        fitter.partial_fit(\n            rows[12:],\n            timestamps[12:],\n        )\n\n        with self.assertRaises(\n            PreprocessingContractError\n        ):\n            fitter.partial_fit(\n                rows[:12],\n                timestamps[:12],\n            )\n\n'
TEST_NEW = '    def test_batch_order_does_not_require_global_timestamp_sort(self) -> None:\n        rows, timestamps = training_fixture()\n\n        single = fit_rebuild_preprocessor(\n            rows,\n            timestamps,\n        )\n\n        fitter = (\n            RebuildPreprocessingFitter()\n        )\n\n        # Deliberately process the later half first. This represents\n        # order-invariant preprocessing over already-canonical rows;\n        # history causality was enforced before this stage.\n        fitter.partial_fit(\n            rows[12:],\n            timestamps[12:],\n        )\n        fitter.partial_fit(\n            rows[:12],\n            timestamps[:12],\n        )\n\n        reordered = fitter.finalize()\n\n        np.testing.assert_allclose(\n            reordered.state[\n                "numeric_mean"\n            ],\n            single.state[\n                "numeric_mean"\n            ],\n            rtol=0.0,\n            atol=1e-12,\n        )\n\n        np.testing.assert_allclose(\n            reordered.state[\n                "numeric_var"\n            ],\n            single.state[\n                "numeric_var"\n            ],\n            rtol=0.0,\n            atol=1e-12,\n        )\n\n        self.assertEqual(\n            reordered.state[\n                "fit_row_count"\n            ],\n            24,\n        )\n        self.assertEqual(\n            reordered.state[\n                "fit_min_timestamp"\n            ],\n            "2018-01-01 00:00:00",\n        )\n        self.assertEqual(\n            reordered.state[\n                "fit_max_timestamp"\n            ],\n            "2018-01-01 23:00:00",\n        )\n\n'
BOOTSTRAP_ORDER_CHECK_OLD = '    if (\n        timestamps.shape[0] > 1\n        and (\n            timestamps[\n                1:\n            ]\n            < timestamps[\n                :-1\n            ]\n        ).any()\n    ):\n        raise ValueError(\n            "Canonical training timestamps must be globally nondecreasing."\n        )\n\n'
BOOTSTRAP_ORDER_CHECK_NEW = '    # Do not require global timestamp sorting here.\n    #\n    # The canonical semantic bundle may preserve verified User+Card block\n    # replay order. History causality/order must already have been enforced\n    # while producing the semantic bundle. File SHA-256 values lock row order.\n\n'
BOOTSTRAP_TEST_OLD = '    def test_timestamp_order_violation_is_rejected(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n            bundle_dir = write_bundle(\n                root\n            )\n\n            timestamp_path = (\n                bundle_dir\n                / TIMESTAMP_FILE\n            )\n\n            values = np.load(\n                timestamp_path,\n                allow_pickle=False,\n            )\n            values = values.copy()\n            values[\n                5\n            ], values[\n                6\n            ] = (\n                values[\n                    6\n                ],\n                values[\n                    5\n                ],\n            )\n\n            np.save(\n                timestamp_path,\n                values,\n                allow_pickle=False,\n            )\n\n            manifest_path = (\n                bundle_dir\n                / MANIFEST_FILE\n            )\n            manifest = json.loads(\n                manifest_path.read_text(\n                    encoding="utf-8"\n                )\n            )\n            manifest[\n                "files"\n            ][\n                TIMESTAMP_FILE\n            ][\n                "sha256"\n            ] = sha256_file(\n                timestamp_path\n            )\n\n            manifest_path.write_text(\n                json.dumps(\n                    manifest\n                ),\n                encoding="utf-8",\n            )\n\n            with self.assertRaises(\n                ValueError\n            ):\n                load_training_bundle(\n                    bundle_dir,\n                    profile=\n                        TEST_PROFILE,\n                )\n\n'
BOOTSTRAP_TEST_NEW = '    def test_global_timestamp_reordering_is_allowed_when_hashes_match(self) -> None:\n        with tempfile.TemporaryDirectory() as tmp:\n            root = Path(tmp)\n            bundle_dir = write_bundle(\n                root\n            )\n\n            timestamp_path = (\n                bundle_dir\n                / TIMESTAMP_FILE\n            )\n\n            values = np.load(\n                timestamp_path,\n                allow_pickle=False,\n            )\n            values = values.copy()\n            values[\n                5\n            ], values[\n                6\n            ] = (\n                values[\n                    6\n                ],\n                values[\n                    5\n                ],\n            )\n\n            np.save(\n                timestamp_path,\n                values,\n                allow_pickle=False,\n            )\n\n            manifest_path = (\n                bundle_dir\n                / MANIFEST_FILE\n            )\n            manifest = json.loads(\n                manifest_path.read_text(\n                    encoding="utf-8"\n                )\n            )\n            manifest[\n                "files"\n            ][\n                TIMESTAMP_FILE\n            ][\n                "sha256"\n            ] = sha256_file(\n                timestamp_path\n            )\n\n            manifest_path.write_text(\n                json.dumps(\n                    manifest\n                ),\n                encoding="utf-8",\n            )\n\n            bundle = load_training_bundle(\n                bundle_dir,\n                profile=\n                    TEST_PROFILE,\n            )\n\n            self.assertEqual(\n                bundle.row_count,\n                24,\n            )\n\n'
PRINT_OLD = '    print(" - global timestamp order: nondecreasing")\n'
PRINT_NEW = '    print(" - global timestamp sort: NOT REQUIRED")\n    print(" - row order integrity: locked by file SHA-256")\n'
MILESTONE_OLD = '    print(" - raw CSV -> canonical bundle producer: NOT added here")\n'
MILESTONE_NEW = '    print(" - raw CSV -> canonical bundle producer: NOT added here")\n    print(" - per-User+Card causal order belongs to bundle production")\n'


def replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one guarded match, found {count}."
        )

    return text.replace(
        old,
        new,
        1,
    )


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / PREPROCESSOR_REL,
        root / PREPROCESSOR_TEST_REL,
        root / BOOTSTRAP_REL,
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required project files are missing: "
            + ", ".join(
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path)
                for path in missing
            )
        )

    return root


def build_patched_contents(
    root: Path,
) -> dict[Path, str]:
    preprocessor_path = (
        root / PREPROCESSOR_REL
    )
    test_path = (
        root / PREPROCESSOR_TEST_REL
    )
    bootstrap_path = (
        root / BOOTSTRAP_REL
    )

    preprocessor_text = (
        preprocessor_path.read_text(
            encoding="utf-8"
        )
    )
    test_text = (
        test_path.read_text(
            encoding="utf-8"
        )
    )
    bootstrap_text = (
        bootstrap_path.read_text(
            encoding="utf-8"
        )
    )

    preprocessor_text = replace_exact(
        preprocessor_text,
        PREPROCESSOR_OLD,
        PREPROCESSOR_NEW,
        label="preprocessing chronological guard",
    )

    test_text = replace_exact(
        test_text,
        TEST_OLD,
        TEST_NEW,
        label="preprocessing chronological unit test",
    )

    bootstrap_text = replace_exact(
        bootstrap_text,
        BOOTSTRAP_ORDER_CHECK_OLD,
        BOOTSTRAP_ORDER_CHECK_NEW,
        label="bundle global timestamp guard",
    )

    bootstrap_text = replace_exact(
        bootstrap_text,
        BOOTSTRAP_TEST_OLD,
        BOOTSTRAP_TEST_NEW,
        label="bundle timestamp-order unit test",
    )

    bootstrap_text = replace_exact(
        bootstrap_text,
        PRINT_OLD,
        PRINT_NEW,
        label="bootstrap contract display",
    )

    bootstrap_text = replace_exact(
        bootstrap_text,
        MILESTONE_OLD,
        MILESTONE_NEW,
        label="bootstrap milestone display",
    )

    return {
        preprocessor_path:
            preprocessor_text,
        test_path:
            test_text,
        bootstrap_path:
            bootstrap_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Correct the rebuild ordering contract in existing source/tests "
            "and in the not-yet-executed training-command bootstrap."
        ),
    )

    args = parser.parse_args()

    root = detect_root()
    patched = build_patched_contents(
        root
    )

    print("=" * 92)
    print(
        "REBUILD ORDER CONTRACT CORRECTION"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        (
            "EXECUTE — GUARDED SOURCE/TEST/BOOTSTRAP CORRECTION"
            if args.execute
            else "DRY RUN — NO FILES WRITTEN"
        ),
    )

    print("\n[1] Scientific correction")
    print(
        " - previous requirement: global chronological batch/bundle order"
    )
    print(
        " - corrected requirement: NO global chronological requirement"
    )
    print(
        " - canonical causal order: enforced upstream per User+Card history block"
    )
    print(
        " - semantic bundle row order: protected by SHA-256"
    )
    print(
        " - W_SHORT timestamp boundary: STILL REQUIRED"
    )

    print("\n[2] Guarded files")
    for path in patched:
        print(
            " -",
            path.relative_to(root),
        )

    print("\n[3] Side-effect boundary")
    print(" - model.fit(): NO")
    print(" - preprocessing fit: NO")
    print(" - predict()/predict_proba(): NO")
    print(" - rebuild artifact write: NO")
    print(" - official artifact modification: NO")
    print(" - research modification: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print(
            "All six exact correction anchors were found once."
        )
        print("Execute with:")
        print(
            "python fix_rebuild_order_contract.py --execute"
        )
        print("=" * 92)
        return

    originals = {
        path:
            path.read_text(
                encoding="utf-8"
            )
        for path in patched
    }

    written = []

    try:
        for path, content in patched.items():
            path.write_text(
                content,
                encoding="utf-8",
            )
            written.append(
                path
            )
    except Exception:
        for path in reversed(
            written
        ):
            try:
                path.write_text(
                    originals[
                        path
                    ],
                    encoding="utf-8",
                )
            except OSError:
                pass
        raise

    print("\n[4] Updated")
    for path in written:
        print(
            " -",
            path.relative_to(root),
        )

    print("\nCORRECTION RESULT: PASS")
    print(
        "No learned state or artifact was created."
    )
    print(
        "Run the existing 139-test suite before re-running "
        "bootstrap_rebuild_training_command.py."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
