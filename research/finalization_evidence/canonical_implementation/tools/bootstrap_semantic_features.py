from __future__ import annotations

import argparse
from pathlib import Path

NEW_FILES = {'final_pipeline/src/fraud_screening/features/semantic.py': '"""Exact semantic feature interface consumed by frozen preprocessing."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Iterable\n\nfrom fraud_screening.data import CanonicalTransaction\nfrom fraud_screening.features.history import (\n    BehavioralFeatures,\n    compute_behavioral_features,\n)\nfrom fraud_screening.features.transaction import (\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n\nNUMERIC_FEATURES = (\n    "amount_numeric",\n    "time_since_previous_transaction_min",\n    "transactions_last_1h",\n    "amount_minus_previous_mean",\n)\n\nBOOLEAN_FEATURES = (\n    "is_new_merchant",\n    "has_prior_card_history",\n)\n\nCATEGORICAL_FEATURES = (\n    "transaction_mode",\n    "location_state",\n    "hour_of_day",\n    "day_of_week",\n)\n\nSEMANTIC_FEATURE_ORDER = (\n    *NUMERIC_FEATURES,\n    *BOOLEAN_FEATURES,\n    *CATEGORICAL_FEATURES,\n)\n\n\n@dataclass(frozen=True, slots=True)\nclass SemanticFeatureRow:\n    """One exact 10-feature row before frozen preprocessing."""\n\n    amount_numeric: float\n    time_since_previous_transaction_min: float | None\n    transactions_last_1h: int\n    amount_minus_previous_mean: float | None\n    is_new_merchant: bool\n    has_prior_card_history: bool\n    transaction_mode: str\n    location_state: str\n    hour_of_day: str\n    day_of_week: str\n\n    def as_dict(self) -> dict[str, object]:\n        """Return the exact stable semantic-feature order."""\n\n        return {\n            "amount_numeric":\n                self.amount_numeric,\n            "time_since_previous_transaction_min":\n                self.time_since_previous_transaction_min,\n            "transactions_last_1h":\n                self.transactions_last_1h,\n            "amount_minus_previous_mean":\n                self.amount_minus_previous_mean,\n            "is_new_merchant":\n                self.is_new_merchant,\n            "has_prior_card_history":\n                self.has_prior_card_history,\n            "transaction_mode":\n                self.transaction_mode,\n            "location_state":\n                self.location_state,\n            "hour_of_day":\n                self.hour_of_day,\n            "day_of_week":\n                self.day_of_week,\n        }\n\n\ndef combine_feature_parts(\n    transaction_features: TransactionFeatures,\n    behavioral_features: BehavioralFeatures,\n) -> SemanticFeatureRow:\n    """Combine current-transaction and strict-causal feature parts."""\n\n    return SemanticFeatureRow(\n        amount_numeric=\n            transaction_features.amount_numeric,\n        time_since_previous_transaction_min=\n            behavioral_features.time_since_previous_transaction_min,\n        transactions_last_1h=\n            behavioral_features.transactions_last_1h,\n        amount_minus_previous_mean=\n            behavioral_features.amount_minus_previous_mean,\n        is_new_merchant=\n            behavioral_features.is_new_merchant,\n        has_prior_card_history=\n            behavioral_features.has_prior_card_history,\n        transaction_mode=\n            transaction_features.transaction_mode,\n        location_state=\n            transaction_features.location_state,\n        hour_of_day=\n            str(transaction_features.hour_of_day),\n        day_of_week=\n            str(transaction_features.day_of_week),\n    )\n\n\ndef build_semantic_feature_rows(\n    transactions: Iterable[CanonicalTransaction],\n) -> list[SemanticFeatureRow]:\n    """Build exact 10-feature rows for one ordered User+Card block."""\n\n    rows = list(transactions)\n\n    behavioral_rows = compute_behavioral_features(\n        rows\n    )\n\n    if len(behavioral_rows) != len(rows):\n        raise RuntimeError(\n            "Behavioral feature row count does not match input row count."\n        )\n\n    return [\n        combine_feature_parts(\n            build_transaction_features(transaction),\n            behavioral,\n        )\n        for transaction, behavioral\n        in zip(rows, behavioral_rows)\n    ]\n', 'final_pipeline/tests/unit/test_semantic_features.py': 'from __future__ import annotations\n\nimport unittest\nfrom datetime import datetime\n\nfrom fraud_screening.data import CanonicalTransaction\nfrom fraud_screening.features import (\n    BOOLEAN_FEATURES,\n    CATEGORICAL_FEATURES,\n    NUMERIC_FEATURES,\n    SEMANTIC_FEATURE_ORDER,\n    build_semantic_feature_rows,\n)\n\n\ndef tx(\n    timestamp: str,\n    amount: float,\n    merchant: int,\n) -> CanonicalTransaction:\n    return CanonicalTransaction(\n        user_id=1,\n        card_id=2,\n        timestamp=datetime.fromisoformat(timestamp),\n        amount_numeric=amount,\n        transaction_mode="Chip Transaction",\n        merchant_id=merchant,\n        merchant_city="Boston",\n        merchant_state="MA",\n        zip_code=2110,\n        location_state="PHYSICAL_COMPLETE",\n    )\n\n\nclass SemanticFeatureTests(unittest.TestCase):\n    def test_exact_role_groups(self) -> None:\n        self.assertEqual(\n            NUMERIC_FEATURES,\n            (\n                "amount_numeric",\n                "time_since_previous_transaction_min",\n                "transactions_last_1h",\n                "amount_minus_previous_mean",\n            ),\n        )\n        self.assertEqual(\n            BOOLEAN_FEATURES,\n            (\n                "is_new_merchant",\n                "has_prior_card_history",\n            ),\n        )\n        self.assertEqual(\n            CATEGORICAL_FEATURES,\n            (\n                "transaction_mode",\n                "location_state",\n                "hour_of_day",\n                "day_of_week",\n            ),\n        )\n\n    def test_exact_ten_feature_order(self) -> None:\n        self.assertEqual(\n            SEMANTIC_FEATURE_ORDER,\n            (\n                "amount_numeric",\n                "time_since_previous_transaction_min",\n                "transactions_last_1h",\n                "amount_minus_previous_mean",\n                "is_new_merchant",\n                "has_prior_card_history",\n                "transaction_mode",\n                "location_state",\n                "hour_of_day",\n                "day_of_week",\n            ),\n        )\n        self.assertEqual(\n            len(SEMANTIC_FEATURE_ORDER),\n            10,\n        )\n\n    def test_output_mapping_exact_order(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0]\n\n        self.assertEqual(\n            tuple(result.as_dict().keys()),\n            SEMANTIC_FEATURE_ORDER,\n        )\n\n    def test_cold_start_structural_values_preserved(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0]\n\n        self.assertEqual(result.amount_numeric, 10.0)\n        self.assertIsNone(\n            result.time_since_previous_transaction_min\n        )\n        self.assertEqual(result.transactions_last_1h, 0)\n        self.assertIsNone(\n            result.amount_minus_previous_mean\n        )\n        self.assertTrue(result.is_new_merchant)\n        self.assertFalse(\n            result.has_prior_card_history\n        )\n\n    def test_hour_and_day_are_categorical_strings(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 14:35:00", 10.0, 100)]\n        )[0]\n\n        self.assertEqual(result.hour_of_day, "14")\n        self.assertEqual(result.day_of_week, "0")\n        self.assertIsInstance(result.hour_of_day, str)\n        self.assertIsInstance(result.day_of_week, str)\n\n    def test_transaction_categorical_values_preserved(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0]\n\n        self.assertEqual(\n            result.transaction_mode,\n            "Chip Transaction",\n        )\n        self.assertEqual(\n            result.location_state,\n            "PHYSICAL_COMPLETE",\n        )\n\n    def test_row_count_is_preserved(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 200),\n            tx("2018-01-01 10:30:00", 40.0, 100),\n        ]\n\n        result = build_semantic_feature_rows(rows)\n\n        self.assertEqual(len(result), len(rows))\n\n    def test_verified_regression_values_survive_merge(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 200),\n            tx("2018-01-01 10:30:00", 40.0, 100),\n        ]\n\n        result = build_semantic_feature_rows(rows)\n\n        self.assertEqual(\n            [\n                item.time_since_previous_transaction_min\n                for item in result\n            ],\n            [None, 60.0, 60.0, 30.0],\n        )\n        self.assertEqual(\n            [\n                item.transactions_last_1h\n                for item in result\n            ],\n            [0, 1, 1, 2],\n        )\n        self.assertEqual(\n            [\n                item.is_new_merchant\n                for item in result\n            ],\n            [True, True, True, False],\n        )\n\n    def test_no_identifier_or_target_is_exposed(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0].as_dict()\n\n        prohibited = {\n            "User",\n            "Card",\n            "Merchant Name",\n            "user_id",\n            "card_id",\n            "merchant_id",\n            "Errors?",\n            "Is Fraud?",\n            "Timestamp",\n            "raw_row_id",\n        }\n\n        self.assertTrue(\n            prohibited.isdisjoint(result.keys())\n        )\n\n    def test_no_non_core_transaction_features(self) -> None:\n        result = build_semantic_feature_rows(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0].as_dict()\n\n        self.assertNotIn("MCC", result)\n        self.assertNotIn("mcc_code", result)\n        self.assertNotIn("month_of_year", result)\n        self.assertNotIn("is_weekend", result)\n        self.assertEqual(len(result), 10)\n\n    def test_empty_block_returns_empty(self) -> None:\n        self.assertEqual(\n            build_semantic_feature_rows([]),\n            [],\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}

FEATURES_INIT_PATH = Path(
    "final_pipeline/src/fraud_screening/features/__init__.py"
)

EXPECTED_EXISTING_INIT = '"""Deterministic feature construction for screening."""\n\nfrom fraud_screening.features.history import (\n    HISTORY_FEATURE_ORDER,\n    BehavioralFeatures,\n    compute_behavioral_features,\n)\nfrom fraud_screening.features.transaction import (\n    TRANSACTION_FEATURE_ORDER,\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n__all__ = [\n    "HISTORY_FEATURE_ORDER",\n    "TRANSACTION_FEATURE_ORDER",\n    "BehavioralFeatures",\n    "TransactionFeatures",\n    "build_transaction_features",\n    "compute_behavioral_features",\n]\n'
UPDATED_INIT = '"""Deterministic feature construction for screening."""\n\nfrom fraud_screening.features.history import (\n    HISTORY_FEATURE_ORDER,\n    BehavioralFeatures,\n    compute_behavioral_features,\n)\nfrom fraud_screening.features.semantic import (\n    BOOLEAN_FEATURES,\n    CATEGORICAL_FEATURES,\n    NUMERIC_FEATURES,\n    SEMANTIC_FEATURE_ORDER,\n    SemanticFeatureRow,\n    build_semantic_feature_rows,\n    combine_feature_parts,\n)\nfrom fraud_screening.features.transaction import (\n    TRANSACTION_FEATURE_ORDER,\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n__all__ = [\n    "BOOLEAN_FEATURES",\n    "CATEGORICAL_FEATURES",\n    "HISTORY_FEATURE_ORDER",\n    "NUMERIC_FEATURES",\n    "SEMANTIC_FEATURE_ORDER",\n    "TRANSACTION_FEATURE_ORDER",\n    "BehavioralFeatures",\n    "SemanticFeatureRow",\n    "TransactionFeatures",\n    "build_semantic_feature_rows",\n    "build_transaction_features",\n    "combine_feature_parts",\n    "compute_behavioral_features",\n]\n'


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Run this script from the repository root."
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the exact semantic feature interface.",
    )
    args = parser.parse_args()

    root = detect_root()
    features_init = root / FEATURES_INIT_PATH

    dependencies = [
        root / "final_pipeline/src/fraud_screening/features/history.py",
        root / "final_pipeline/src/fraud_screening/features/transaction.py",
        root / "final_pipeline/tests/unit/test_history_features.py",
        root / "final_pipeline/tests/unit/test_transaction_features.py",
        features_init,
    ]

    missing_dependencies = [
        str(path.relative_to(root))
        for path in dependencies
        if not path.is_file()
    ]

    collisions = [
        relative
        for relative in NEW_FILES
        if (root / relative).exists()
    ]

    init_matches_expected = (
        features_init.is_file()
        and features_init.read_text(encoding="utf-8")
        == EXPECTED_EXISTING_INIT
    )

    print("=" * 86)
    print("FINAL PIPELINE — SEMANTIC FEATURE INTERFACE BOOTSTRAP")
    print("=" * 86)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — GUARDED UPDATE + NEW FILES"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(" - missing dependencies:", len(missing_dependencies))
    for relative in missing_dependencies:
        print("   *", relative)

    print("\n[2] Existing source guard")
    print(
        " - features/__init__.py exact expected content:",
        "YES" if init_matches_expected else "NO",
    )

    print("\n[3] Planned new files")
    for relative in NEW_FILES:
        print(" -", relative)

    print("\n[4] New-file collision gate")
    print(" - collisions:", len(collisions))
    for relative in collisions:
        print("   *", relative)

    if (
        missing_dependencies
        or collisions
        or not init_matches_expected
    ):
        print("\nRESULT: STOP")
        print("No files were written.")
        raise SystemExit(1)

    print("\n[5] Exact semantic interface")
    print(" - numeric: 4")
    print(" - boolean: 2")
    print(" - categorical: 4")
    print(" - total: 10")
    print(" - hour_of_day categorical representation: string")
    print(" - day_of_week categorical representation: string")
    print(" - structural NA allowed only in:")
    print("   * time_since_previous_transaction_min")
    print("   * amount_minus_previous_mean")
    print(" - identifiers/target/direct metadata: EXCLUDED")

    print("\n[6] Scientific/runtime boundary")
    print(" - model fit: NO")
    print(" - preprocessing fit: NO")
    print(" - preprocessing transform: NO")
    print(" - prediction: NO")
    print(" - feature selection change: NO")
    print(" - research files modified: NO")
    print(" - official artifacts modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_semantic_features.py --execute"
        )
        print("=" * 86)
        return

    created = []
    original_init = features_init.read_text(
        encoding="utf-8"
    )

    try:
        for relative, content in NEW_FILES.items():
            path = root / relative
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            path.write_text(
                content,
                encoding="utf-8",
            )
            created.append(path)

        features_init.write_text(
            UPDATED_INIT,
            encoding="utf-8",
        )

    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass

        try:
            features_init.write_text(
                original_init,
                encoding="utf-8",
            )
        except OSError:
            pass

        raise

    print("\n[7] Created")
    for path in created:
        print(" -", path.relative_to(root))

    print("\n[8] Guarded source update")
    print(" -", FEATURES_INIT_PATH)

    print("\nBOOTSTRAP RESULT: PASS")
    print("Verification command:")
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
