from __future__ import annotations

import argparse
import ast
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


PREPROCESSOR_OLD = """        if (
            self._fit_max_timestamp is not None
            and batch_min < self._fit_max_timestamp
        ):
            raise PreprocessingContractError(
                "Rebuild preprocessing batches must be chronological."
            )

"""

PREPROCESSOR_NEW = """        # No global chronological-batch requirement here.
        #
        # Canonical replay is ordered by contiguous User+Card blocks.
        # Timestamp monotonicity belongs to each history block before
        # semantic features are emitted. Preprocessing learned statistics
        # are order-invariant across already-canonical semantic rows.

"""

PREPROCESSOR_TEST_OLD = """    def test_non_chronological_batch_is_rejected(self) -> None:
        rows, timestamps = training_fixture()

        fitter = (
            RebuildPreprocessingFitter()
        )

        fitter.partial_fit(
            rows[12:],
            timestamps[12:],
        )

        with self.assertRaises(
            PreprocessingContractError
        ):
            fitter.partial_fit(
                rows[:12],
                timestamps[:12],
            )

"""

PREPROCESSOR_TEST_NEW = """    def test_batch_order_does_not_require_global_timestamp_sort(self) -> None:
        rows, timestamps = training_fixture()

        single = fit_rebuild_preprocessor(
            rows,
            timestamps,
        )

        fitter = (
            RebuildPreprocessingFitter()
        )

        # Deliberately process the later half first. This represents
        # order-invariant preprocessing over already-canonical rows;
        # history causality was enforced before this stage.
        fitter.partial_fit(
            rows[12:],
            timestamps[12:],
        )
        fitter.partial_fit(
            rows[:12],
            timestamps[:12],
        )

        reordered = fitter.finalize()

        np.testing.assert_allclose(
            reordered.state[
                "numeric_mean"
            ],
            single.state[
                "numeric_mean"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            reordered.state[
                "numeric_var"
            ],
            single.state[
                "numeric_var"
            ],
            rtol=0.0,
            atol=1e-12,
        )

        self.assertEqual(
            reordered.state[
                "fit_row_count"
            ],
            24,
        )
        self.assertEqual(
            reordered.state[
                "fit_min_timestamp"
            ],
            "2018-01-01 00:00:00",
        )
        self.assertEqual(
            reordered.state[
                "fit_max_timestamp"
            ],
            "2018-01-01 23:00:00",
        )

"""

BUNDLE_OLD = """    if (
        timestamps.shape[0] > 1
        and (
            timestamps[
                1:
            ]
            < timestamps[
                :-1
            ]
        ).any()
    ):
        raise ValueError(
            "Canonical training timestamps must be globally nondecreasing."
        )

"""

BUNDLE_NEW = """    # Global timestamp sorting is intentionally NOT required here.
    #
    # A canonical semantic bundle may preserve verified contiguous
    # User+Card replay order. History causality and within-card ordering
    # belong to bundle production. Per-file SHA-256 values lock row order.

"""

BUNDLE_TEST_OLD = """    def test_timestamp_order_violation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = write_bundle(
                root
            )

            timestamp_path = (
                bundle_dir
                / TIMESTAMP_FILE
            )

            values = np.load(
                timestamp_path,
                allow_pickle=False,
            )
            values = values.copy()
            values[
                5
            ], values[
                6
            ] = (
                values[
                    6
                ],
                values[
                    5
                ],
            )

            np.save(
                timestamp_path,
                values,
                allow_pickle=False,
            )

            manifest_path = (
                bundle_dir
                / MANIFEST_FILE
            )
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            manifest[
                "files"
            ][
                TIMESTAMP_FILE
            ][
                "sha256"
            ] = sha256_file(
                timestamp_path
            )

            manifest_path.write_text(
                json.dumps(
                    manifest
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                load_training_bundle(
                    bundle_dir,
                    profile=
                        TEST_PROFILE,
                )

"""

BUNDLE_TEST_NEW = """    def test_global_timestamp_reordering_is_allowed_when_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = write_bundle(
                root
            )

            timestamp_path = (
                bundle_dir
                / TIMESTAMP_FILE
            )

            values = np.load(
                timestamp_path,
                allow_pickle=False,
            )
            values = values.copy()
            values[
                5
            ], values[
                6
            ] = (
                values[
                    6
                ],
                values[
                    5
                ],
            )

            np.save(
                timestamp_path,
                values,
                allow_pickle=False,
            )

            manifest_path = (
                bundle_dir
                / MANIFEST_FILE
            )
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            manifest[
                "files"
            ][
                TIMESTAMP_FILE
            ][
                "sha256"
            ] = sha256_file(
                timestamp_path
            )

            manifest_path.write_text(
                json.dumps(
                    manifest
                ),
                encoding="utf-8",
            )

            bundle = load_training_bundle(
                bundle_dir,
                profile=
                    TEST_PROFILE,
            )

            self.assertEqual(
                bundle.row_count,
                24,
            )

"""


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}."
        )

    return text.replace(
        old,
        new,
        1,
    )


def load_new_files_assignment(
    text: str,
) -> tuple[dict[str, str], int, int]:
    tree = ast.parse(text)

    matches = []

    for node in tree.body:
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        if len(node.targets) != 1:
            continue

        target = node.targets[0]

        if (
            isinstance(
                target,
                ast.Name,
            )
            and target.id == "NEW_FILES"
        ):
            matches.append(node)

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one NEW_FILES assignment."
        )

    node = matches[0]

    value = ast.literal_eval(
        node.value
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            "NEW_FILES must evaluate to a dict."
        )

    if not all(
        isinstance(key, str)
        and isinstance(item, str)
        for key, item in value.items()
    ):
        raise RuntimeError(
            "NEW_FILES must map string paths to string contents."
        )

    return (
        value,
        node.lineno,
        node.end_lineno,
    )


def render_new_files(
    values: dict[str, str],
) -> str:
    lines = [
        "NEW_FILES = {",
    ]

    for key, value in values.items():
        lines.append(
            f"    {key!r}:"
        )
        lines.append(
            f"        {value!r},"
        )

    lines.append("}")

    return "\n".join(
        lines
    )


def patch_bootstrap(
    text: str,
) -> str:
    values, start_line, end_line = (
        load_new_files_assignment(
            text
        )
    )

    bundle_key = (
        "final_pipeline/src/fraud_screening/training/bundle.py"
    )
    test_key = (
        "final_pipeline/tests/unit/test_rebuild_training_bundle.py"
    )

    if (
        bundle_key not in values
        or test_key not in values
    ):
        raise RuntimeError(
            "Expected bundle/test entries are missing from NEW_FILES."
        )

    values[
        bundle_key
    ] = replace_once(
        values[
            bundle_key
        ],
        BUNDLE_OLD,
        BUNDLE_NEW,
        "bundle global timestamp guard",
    )

    values[
        test_key
    ] = replace_once(
        values[
            test_key
        ],
        BUNDLE_TEST_OLD,
        BUNDLE_TEST_NEW,
        "bundle timestamp-order unit test",
    )

    source_lines = text.splitlines()

    replacement = render_new_files(
        values
    ).splitlines()

    source_lines[
        start_line - 1:
        end_line
    ] = replacement

    patched = "\n".join(
        source_lines
    ) + "\n"

    patched = replace_once(
        patched,
        '    print(" - global timestamp order: nondecreasing")\n',
        (
            '    print(" - global timestamp sort: NOT REQUIRED")\n'
            '    print(" - row order integrity: locked by file SHA-256")\n'
        ),
        "bootstrap contract display",
    )

    patched = replace_once(
        patched,
        '    print(" - raw CSV -> canonical bundle producer: NOT added here")\n',
        (
            '    print(" - raw CSV -> canonical bundle producer: NOT added here")\n'
            '    print(" - per-User+Card causal order belongs to bundle production")\n'
        ),
        "bootstrap milestone display",
    )

    # Syntax-check the corrected bootstrap before it can be written.
    compile(
        patched,
        "bootstrap_rebuild_training_command.py",
        "exec",
    )

    return patched


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
            "Required files are missing: "
            + ", ".join(
                str(path)
                for path in missing
            )
        )

    return root


def build_changes(
    root: Path,
) -> dict[Path, str]:
    preprocessor_path = (
        root / PREPROCESSOR_REL
    )
    preprocessor_test_path = (
        root / PREPROCESSOR_TEST_REL
    )
    bootstrap_path = (
        root / BOOTSTRAP_REL
    )

    preprocessor = (
        preprocessor_path.read_text(
            encoding="utf-8"
        )
    )
    preprocessor_test = (
        preprocessor_test_path.read_text(
            encoding="utf-8"
        )
    )
    bootstrap = (
        bootstrap_path.read_text(
            encoding="utf-8"
        )
    )

    corrected_preprocessor = replace_once(
        preprocessor,
        PREPROCESSOR_OLD,
        PREPROCESSOR_NEW,
        "preprocessing chronological guard",
    )

    corrected_test = replace_once(
        preprocessor_test,
        PREPROCESSOR_TEST_OLD,
        PREPROCESSOR_TEST_NEW,
        "preprocessing chronological unit test",
    )

    corrected_bootstrap = patch_bootstrap(
        bootstrap
    )

    return {
        preprocessor_path:
            corrected_preprocessor,
        preprocessor_test_path:
            corrected_test,
        bootstrap_path:
            corrected_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Apply the corrected per-User+Card ordering contract."
        ),
    )

    args = parser.parse_args()

    root = detect_root()
    changes = build_changes(
        root
    )

    print("=" * 92)
    print(
        "REBUILD ORDER CONTRACT CORRECTION — V2"
    )
    print("=" * 92)
    print("Project root:", root)
    print(
        "Mode:",
        (
            "EXECUTE — GUARDED CORRECTION"
            if args.execute
            else "DRY RUN — NO FILES WRITTEN"
        ),
    )

    print("\n[1] Corrected scientific boundary")
    print(
        " - global dataset timestamp sort: NOT REQUIRED"
    )
    print(
        " - causal history ordering: per User+Card, upstream of preprocessing"
    )
    print(
        " - same-timestamp history rule: unchanged"
    )
    print(
        " - W_SHORT temporal boundary: unchanged"
    )
    print(
        " - canonical bundle row order: protected by SHA-256"
    )

    print("\n[2] Exact guarded changes resolved")
    print(
        " - preprocessing chronological guard: FOUND"
    )
    print(
        " - preprocessing unit-test expectation: FOUND"
    )
    print(
        " - bootstrap NEW_FILES bundle guard: FOUND via AST"
    )
    print(
        " - bootstrap NEW_FILES bundle unit test: FOUND via AST"
    )
    print(
        " - bootstrap displayed contract: FOUND"
    )
    print(
        " - bootstrap milestone note: FOUND"
    )

    print("\n[3] Files affected")
    for path in changes:
        print(
            " -",
            path.relative_to(root),
        )

    print("\n[4] Safety")
    print(" - model.fit(): NO")
    print(" - preprocessing fit: NO")
    print(" - predict()/predict_proba(): NO")
    print(" - rebuild artifact write: NO")
    print(" - official artifact modification: NO")
    print(" - research modification: NO")
    print(" - corrected bootstrap syntax check: PASS")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python fix_rebuild_order_contract_v2.py --execute"
        )
        print("=" * 92)
        return

    originals = {
        path:
            path.read_text(
                encoding="utf-8"
            )
        for path in changes
    }

    written = []

    try:
        for path, content in changes.items():
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

    print("\n[5] Updated")
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
        "Next: run the full 139-test suite."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
