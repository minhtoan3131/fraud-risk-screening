from __future__ import annotations

import argparse
from pathlib import Path

NEW_FILES = {'final_pipeline/src/fraud_screening/features/history.py': '"""Strict-causal behavioral features built from card history."""\n\nfrom __future__ import annotations\n\nfrom collections import deque\nfrom dataclasses import dataclass\nfrom datetime import timedelta\nfrom typing import Iterable\n\nfrom fraud_screening.data import CanonicalTransaction\nfrom fraud_screening.errors import HistoryValidationError\n\n\nHISTORY_FEATURE_ORDER = (\n    "time_since_previous_transaction_min",\n    "transactions_last_1h",\n    "amount_minus_previous_mean",\n    "is_new_merchant",\n    "has_prior_card_history",\n)\n\n\n@dataclass(frozen=True, slots=True)\nclass BehavioralFeatures:\n    """Strict-prior behavioral features for one transaction."""\n\n    time_since_previous_transaction_min: float | None\n    transactions_last_1h: int\n    amount_minus_previous_mean: float | None\n    is_new_merchant: bool\n    has_prior_card_history: bool\n\n    def as_dict(self) -> dict[str, object]:\n        return {\n            "time_since_previous_transaction_min":\n                self.time_since_previous_transaction_min,\n            "transactions_last_1h":\n                self.transactions_last_1h,\n            "amount_minus_previous_mean":\n                self.amount_minus_previous_mean,\n            "is_new_merchant":\n                self.is_new_merchant,\n            "has_prior_card_history":\n                self.has_prior_card_history,\n        }\n\n\ndef _validate_same_card(\n    transactions: list[CanonicalTransaction],\n) -> None:\n    if not transactions:\n        return\n\n    key = (\n        transactions[0].user_id,\n        transactions[0].card_id,\n    )\n\n    for transaction in transactions[1:]:\n        if (\n            transaction.user_id,\n            transaction.card_id,\n        ) != key:\n            raise HistoryValidationError(\n                "All transactions in a card-history block must "\n                "belong to the same User + Card."\n            )\n\n\ndef _validate_non_decreasing_time(\n    transactions: list[CanonicalTransaction],\n) -> None:\n    for previous, current in zip(\n        transactions,\n        transactions[1:],\n    ):\n        if current.timestamp < previous.timestamp:\n            raise HistoryValidationError(\n                "Transactions must be ordered by non-decreasing timestamp."\n            )\n\n\ndef compute_behavioral_features(\n    transactions: Iterable[CanonicalTransaction],\n) -> list[BehavioralFeatures]:\n    """Compute strict-causal features for an ordered User+Card block.\n\n    Transactions sharing one timestamp are treated as one prediction point.\n    Every row in that timestamp group sees only transactions with strictly\n    earlier timestamps. State is updated only after the whole timestamp group\n    has been scored.\n    """\n\n    rows = list(transactions)\n\n    if not rows:\n        return []\n\n    _validate_same_card(rows)\n    _validate_non_decreasing_time(rows)\n\n    results: list[BehavioralFeatures] = []\n\n    prior_count = 0\n    prior_amount_sum = 0.0\n    seen_merchants: set[int] = set()\n    previous_distinct_timestamp = None\n    one_hour_window = deque()\n\n    start = 0\n\n    while start < len(rows):\n        timestamp = rows[start].timestamp\n        end = start + 1\n\n        while (\n            end < len(rows)\n            and rows[end].timestamp == timestamp\n        ):\n            end += 1\n\n        group = rows[start:end]\n        has_prior = prior_count > 0\n\n        if has_prior:\n            assert previous_distinct_timestamp is not None\n            delta = timestamp - previous_distinct_timestamp\n            time_since_previous_min = (\n                delta.total_seconds() / 60.0\n            )\n            previous_amount_mean = (\n                prior_amount_sum / prior_count\n            )\n        else:\n            time_since_previous_min = None\n            previous_amount_mean = None\n\n        window_start = timestamp - timedelta(hours=1)\n\n        while (\n            one_hour_window\n            and one_hour_window[0] < window_start\n        ):\n            one_hour_window.popleft()\n\n        transactions_last_1h = len(one_hour_window)\n\n        for transaction in group:\n            if previous_amount_mean is None:\n                amount_minus_previous_mean = None\n            else:\n                amount_minus_previous_mean = (\n                    transaction.amount_numeric\n                    - previous_amount_mean\n                )\n\n            results.append(\n                BehavioralFeatures(\n                    time_since_previous_transaction_min=\n                        time_since_previous_min,\n                    transactions_last_1h=\n                        transactions_last_1h,\n                    amount_minus_previous_mean=\n                        amount_minus_previous_mean,\n                    is_new_merchant=(\n                        transaction.merchant_id\n                        not in seen_merchants\n                    ),\n                    has_prior_card_history=has_prior,\n                )\n            )\n\n        # Strict-causal state update happens only after the whole timestamp group.\n        group_count = len(group)\n        prior_count += group_count\n        prior_amount_sum += sum(\n            transaction.amount_numeric\n            for transaction in group\n        )\n        seen_merchants.update(\n            transaction.merchant_id\n            for transaction in group\n        )\n        one_hour_window.extend(\n            transaction.timestamp\n            for transaction in group\n        )\n        previous_distinct_timestamp = timestamp\n        start = end\n\n    return results\n', 'final_pipeline/tests/unit/test_history_features.py': 'from __future__ import annotations\n\nimport unittest\nfrom datetime import datetime\n\nfrom fraud_screening.data import CanonicalTransaction\nfrom fraud_screening.errors import HistoryValidationError\nfrom fraud_screening.features import (\n    HISTORY_FEATURE_ORDER,\n    compute_behavioral_features,\n)\n\n\ndef tx(\n    timestamp: str,\n    amount: float,\n    merchant: int,\n    *,\n    user: int = 1,\n    card: int = 1,\n) -> CanonicalTransaction:\n    return CanonicalTransaction(\n        user_id=user,\n        card_id=card,\n        timestamp=datetime.fromisoformat(timestamp),\n        amount_numeric=amount,\n        transaction_mode="Chip Transaction",\n        merchant_id=merchant,\n        merchant_city="Boston",\n        merchant_state="MA",\n        zip_code=2110,\n        location_state="PHYSICAL_COMPLETE",\n    )\n\n\nclass HistoryFeatureTests(unittest.TestCase):\n    def test_exact_history_feature_order(self) -> None:\n        self.assertEqual(\n            HISTORY_FEATURE_ORDER,\n            (\n                "time_since_previous_transaction_min",\n                "transactions_last_1h",\n                "amount_minus_previous_mean",\n                "is_new_merchant",\n                "has_prior_card_history",\n            ),\n        )\n\n    def test_cold_start_semantics(self) -> None:\n        result = compute_behavioral_features(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0]\n\n        self.assertIsNone(\n            result.time_since_previous_transaction_min\n        )\n        self.assertEqual(result.transactions_last_1h, 0)\n        self.assertIsNone(\n            result.amount_minus_previous_mean\n        )\n        self.assertTrue(result.is_new_merchant)\n        self.assertFalse(\n            result.has_prior_card_history\n        )\n\n    def test_regression_case_from_verified_builder(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 200),\n            tx("2018-01-01 10:30:00", 40.0, 100),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            [\n                item.time_since_previous_transaction_min\n                for item in result\n            ],\n            [None, 60.0, 60.0, 30.0],\n        )\n        self.assertEqual(\n            [\n                item.transactions_last_1h\n                for item in result\n            ],\n            [0, 1, 1, 2],\n        )\n        self.assertEqual(\n            [\n                item.is_new_merchant\n                for item in result\n            ],\n            [True, True, True, False],\n        )\n\n    def test_same_timestamp_peer_not_in_amount_mean(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 300),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            result[1].amount_minus_previous_mean,\n            10.0,\n        )\n        self.assertEqual(\n            result[2].amount_minus_previous_mean,\n            20.0,\n        )\n\n    def test_previous_mean_uses_full_strict_prior_history(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 11:00:00", 60.0, 300),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        # Previous mean for 11:00 is mean(10, 20) = 15.\n        self.assertEqual(\n            result[2].amount_minus_previous_mean,\n            45.0,\n        )\n\n    def test_one_hour_window_includes_exact_left_boundary(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            result[1].transactions_last_1h,\n            1,\n        )\n\n    def test_one_hour_window_excludes_older_history(self) -> None:\n        rows = [\n            tx("2018-01-01 08:59:59", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            result[1].transactions_last_1h,\n            0,\n        )\n\n    def test_same_timestamp_peers_not_in_velocity(self) -> None:\n        rows = [\n            tx("2018-01-01 09:30:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 300),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            result[1].transactions_last_1h,\n            1,\n        )\n        self.assertEqual(\n            result[2].transactions_last_1h,\n            1,\n        )\n\n    def test_same_timestamp_first_merchant_all_new(self) -> None:\n        rows = [\n            tx("2018-01-01 10:00:00", 20.0, 200),\n            tx("2018-01-01 10:00:00", 30.0, 200),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            [item.is_new_merchant for item in result],\n            [True, True],\n        )\n\n    def test_existing_merchant_is_not_new(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 100),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertFalse(result[1].is_new_merchant)\n\n    def test_same_timestamp_does_not_create_prior_history(self) -> None:\n        rows = [\n            tx("2018-01-01 10:00:00", 10.0, 100),\n            tx("2018-01-01 10:00:00", 20.0, 200),\n        ]\n\n        result = compute_behavioral_features(rows)\n\n        self.assertEqual(\n            [\n                item.has_prior_card_history\n                for item in result\n            ],\n            [False, False],\n        )\n        self.assertEqual(\n            [\n                item.time_since_previous_transaction_min\n                for item in result\n            ],\n            [None, None],\n        )\n\n    def test_mixed_cards_are_rejected(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100, card=1),\n            tx("2018-01-01 10:00:00", 20.0, 200, card=2),\n        ]\n\n        with self.assertRaises(HistoryValidationError):\n            compute_behavioral_features(rows)\n\n    def test_mixed_users_are_rejected(self) -> None:\n        rows = [\n            tx("2018-01-01 09:00:00", 10.0, 100, user=1),\n            tx("2018-01-01 10:00:00", 20.0, 200, user=2),\n        ]\n\n        with self.assertRaises(HistoryValidationError):\n            compute_behavioral_features(rows)\n\n    def test_decreasing_timestamp_is_rejected(self) -> None:\n        rows = [\n            tx("2018-01-01 10:00:00", 10.0, 100),\n            tx("2018-01-01 09:00:00", 20.0, 200),\n        ]\n\n        with self.assertRaises(HistoryValidationError):\n            compute_behavioral_features(rows)\n\n    def test_empty_history_block_returns_empty(self) -> None:\n        self.assertEqual(\n            compute_behavioral_features([]),\n            [],\n        )\n\n    def test_mapping_order_is_stable(self) -> None:\n        result = compute_behavioral_features(\n            [tx("2018-01-01 09:00:00", 10.0, 100)]\n        )[0]\n\n        self.assertEqual(\n            tuple(result.as_dict().keys()),\n            HISTORY_FEATURE_ORDER,\n        )\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}

FEATURES_INIT_PATH = Path(
    "final_pipeline/src/fraud_screening/features/__init__.py"
)

EXPECTED_EXISTING_INIT = '"""Deterministic feature construction for screening."""\n\nfrom fraud_screening.features.transaction import (\n    TRANSACTION_FEATURE_ORDER,\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n__all__ = [\n    "TRANSACTION_FEATURE_ORDER",\n    "TransactionFeatures",\n    "build_transaction_features",\n]\n'
UPDATED_INIT = '"""Deterministic feature construction for screening."""\n\nfrom fraud_screening.features.history import (\n    HISTORY_FEATURE_ORDER,\n    BehavioralFeatures,\n    compute_behavioral_features,\n)\nfrom fraud_screening.features.transaction import (\n    TRANSACTION_FEATURE_ORDER,\n    TransactionFeatures,\n    build_transaction_features,\n)\n\n__all__ = [\n    "HISTORY_FEATURE_ORDER",\n    "TRANSACTION_FEATURE_ORDER",\n    "BehavioralFeatures",\n    "TransactionFeatures",\n    "build_transaction_features",\n    "compute_behavioral_features",\n]\n'


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
        help="Create strict-causal history feature files.",
    )
    args = parser.parse_args()

    root = detect_root()
    features_init = root / FEATURES_INIT_PATH

    dependencies = [
        root / "final_pipeline/src/fraud_screening/data/models.py",
        root / "final_pipeline/src/fraud_screening/errors.py",
        root / "final_pipeline/src/fraud_screening/features/transaction.py",
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

    print("=" * 88)
    print("FINAL PIPELINE — STRICT-CAUSAL HISTORY FEATURE BOOTSTRAP")
    print("=" * 88)
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

    print("\n[5] Locked history semantics")
    print(" - entity: User + Card")
    print(" - strict prior rule: history timestamp < current timestamp")
    print(" - same-timestamp peers as history: NO")
    print(" - state update: AFTER WHOLE TIMESTAMP GROUP")
    print(" - recency: nearest distinct strict-prior timestamp")
    print(" - velocity window: [T - 1 hour, T)")
    print(" - amount baseline: full strict-prior mean")
    print(" - merchant novelty: full strict-prior merchant history")
    print(" - cold start: explicit structural state")
    print(" - velocity implementation: O(n) sliding deque")

    print("\n[6] Scientific/runtime boundary")
    print(" - model fit: NO")
    print(" - preprocessing fit: NO")
    print(" - preprocessing transform: NO")
    print(" - prediction: NO")
    print(" - target-label history: NO")
    print(" - research files modified: NO")
    print(" - official artifacts modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_history_features.py --execute"
        )
        print("=" * 88)
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
    print("=" * 88)


if __name__ == "__main__":
    main()
