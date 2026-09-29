from __future__ import annotations

import argparse
from pathlib import Path

FILES = {'final_pipeline/src/fraud_screening/features/__init__.py': '"""Deterministic feature construction for screening."""\n\nfrom fraud_screening.features.transaction import (\n    TRANSACTION_FEATURE_ORDER,\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n__all__ = [\n    "TRANSACTION_FEATURE_ORDER",\n    "TransactionFeatures",\n    "build_transaction_features",\n]\n', 'final_pipeline/src/fraud_screening/features/transaction.py': '"""Current-transaction feature representation."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\n\nfrom fraud_screening.data import CanonicalTransaction\n\n\nTRANSACTION_FEATURE_ORDER = (\n    "amount_numeric",\n    "transaction_mode",\n    "location_state",\n    "hour_of_day",\n    "day_of_week",\n)\n\n\n@dataclass(frozen=True, slots=True)\nclass TransactionFeatures:\n    """Features derived only from the current transaction."""\n\n    amount_numeric: float\n    transaction_mode: str\n    location_state: str\n    hour_of_day: int\n    day_of_week: int\n\n    def as_dict(self) -> dict[str, object]:\n        """Return fields in the stable transaction-feature order."""\n\n        return {\n            "amount_numeric": self.amount_numeric,\n            "transaction_mode": self.transaction_mode,\n            "location_state": self.location_state,\n            "hour_of_day": self.hour_of_day,\n            "day_of_week": self.day_of_week,\n        }\n\n\ndef build_transaction_features(\n    transaction: CanonicalTransaction,\n) -> TransactionFeatures:\n    """Build deterministic current-transaction features."""\n\n    return TransactionFeatures(\n        amount_numeric=transaction.amount_numeric,\n        transaction_mode=transaction.transaction_mode,\n        location_state=transaction.location_state,\n        hour_of_day=transaction.timestamp.hour,\n        day_of_week=transaction.timestamp.weekday(),\n    )\n', 'final_pipeline/tests/unit/test_transaction_features.py': 'from __future__ import annotations\n\nimport unittest\nfrom datetime import datetime\n\nfrom fraud_screening.data import CanonicalTransaction\nfrom fraud_screening.features import (\n    TRANSACTION_FEATURE_ORDER,\n    build_transaction_features,\n)\n\n\ndef transaction_at(timestamp: datetime) -> CanonicalTransaction:\n    return CanonicalTransaction(\n        user_id=17,\n        card_id=3,\n        timestamp=timestamp,\n        amount_numeric=-12.5,\n        transaction_mode="Chip Transaction",\n        merchant_id=998877,\n        merchant_city="Boston",\n        merchant_state="MA",\n        zip_code=2110,\n        location_state="PHYSICAL_COMPLETE",\n    )\n\n\nclass TransactionFeatureTests(unittest.TestCase):\n    def test_exact_transaction_feature_order(self) -> None:\n        self.assertEqual(\n            TRANSACTION_FEATURE_ORDER,\n            (\n                "amount_numeric",\n                "transaction_mode",\n                "location_state",\n                "hour_of_day",\n                "day_of_week",\n            ),\n        )\n\n    def test_build_current_transaction_features(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 14, 35)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.amount_numeric, -12.5)\n        self.assertEqual(\n            features.transaction_mode,\n            "Chip Transaction",\n        )\n        self.assertEqual(\n            features.location_state,\n            "PHYSICAL_COMPLETE",\n        )\n        self.assertEqual(features.hour_of_day, 14)\n        self.assertEqual(features.day_of_week, 0)\n\n    def test_output_mapping_has_exact_fields_and_order(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 14, 35)\n        )\n\n        mapping = build_transaction_features(tx).as_dict()\n\n        self.assertEqual(\n            tuple(mapping.keys()),\n            TRANSACTION_FEATURE_ORDER,\n        )\n\n    def test_identifiers_are_not_exposed(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 14, 35)\n        )\n\n        mapping = build_transaction_features(tx).as_dict()\n\n        self.assertNotIn("User", mapping)\n        self.assertNotIn("Card", mapping)\n        self.assertNotIn("Merchant Name", mapping)\n        self.assertNotIn("user_id", mapping)\n        self.assertNotIn("card_id", mapping)\n        self.assertNotIn("merchant_id", mapping)\n\n    def test_amount_sign_is_preserved(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 14, 35)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.amount_numeric, -12.5)\n\n    def test_midnight_hour_is_zero(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 0, 0)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.hour_of_day, 0)\n\n    def test_last_hour_is_twenty_three(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 23, 59)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.hour_of_day, 23)\n\n    def test_monday_is_day_zero(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 1, 12, 0)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.day_of_week, 0)\n\n    def test_sunday_is_day_six(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 7, 12, 0)\n        )\n\n        features = build_transaction_features(tx)\n\n        self.assertEqual(features.day_of_week, 6)\n\n    def test_no_month_or_weekend_feature_is_emitted(self) -> None:\n        tx = transaction_at(\n            datetime(2024, 1, 7, 12, 0)\n        )\n\n        mapping = build_transaction_features(tx).as_dict()\n\n        self.assertNotIn("month_of_year", mapping)\n        self.assertNotIn("is_weekend", mapping)\n        self.assertEqual(len(mapping), 5)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}


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
        help="Create transaction-feature implementation files.",
    )
    args = parser.parse_args()

    root = detect_root()

    dependencies = [
        root / "final_pipeline/src/fraud_screening/data/models.py",
        root / "final_pipeline/src/fraud_screening/data/parsing.py",
        root / "final_pipeline/tests/unit/test_data_boundary.py",
    ]

    missing_dependencies = [
        str(path.relative_to(root))
        for path in dependencies
        if not path.is_file()
    ]

    collisions = [
        relative
        for relative in FILES
        if (root / relative).exists()
    ]

    print("=" * 84)
    print("FINAL PIPELINE — TRANSACTION FEATURE BOOTSTRAP")
    print("=" * 84)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — NEW FILES ONLY"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Dependency gate")
    print(" - missing dependencies:", len(missing_dependencies))
    for relative in missing_dependencies:
        print("   *", relative)

    print("\n[2] Planned files")
    for relative in FILES:
        print(" -", relative)

    print("\n[3] Collision gate")
    print(" - collisions:", len(collisions))
    for relative in collisions:
        print("   *", relative)

    if missing_dependencies or collisions:
        print("\nRESULT: STOP")
        print("No files were written.")
        raise SystemExit(1)

    print("\n[4] Locked feature boundary")
    print(" - amount_numeric")
    print(" - transaction_mode")
    print(" - location_state")
    print(" - hour_of_day")
    print(" - day_of_week")
    print(" - identifiers as direct features: NO")
    print(" - month_of_year: NO")
    print(" - is_weekend: NO")

    print("\n[5] Scientific/runtime boundary")
    print(" - model fit: NO")
    print(" - preprocessing fit: NO")
    print(" - preprocessing transform: NO")
    print(" - prediction: NO")
    print(" - history feature computation: NO")
    print(" - research files modified: NO")
    print(" - official artifacts modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_transaction_features.py --execute"
        )
        print("=" * 84)
        return

    created = []

    try:
        for relative, content in FILES.items():
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
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise

    print("\n[6] Created")
    for path in created:
        print(" -", path.relative_to(root))

    print("\nBOOTSTRAP RESULT: PASS")
    print("Verification command:")
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
