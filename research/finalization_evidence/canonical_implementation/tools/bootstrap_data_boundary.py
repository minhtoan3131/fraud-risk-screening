from __future__ import annotations

import argparse
from pathlib import Path

FILES = {'final_pipeline/pyproject.toml': '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "fraud-screening"\nversion = "0.1.0"\ndescription = "Transaction fraud risk screening pipeline"\nrequires-python = ">=3.11"\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n', 'final_pipeline/src/fraud_screening/__init__.py': '"""Core package for transaction fraud risk screening."""\n\nfrom fraud_screening.data import (\n    CanonicalTransaction,\n    extract_scoring_payload,\n    parse_scoring_payload,\n)\n\n__all__ = [\n    "CanonicalTransaction",\n    "extract_scoring_payload",\n    "parse_scoring_payload",\n]\n', 'final_pipeline/src/fraud_screening/errors.py': '"""Domain-specific errors used by the screening pipeline."""\n\n\nclass FraudScreeningError(Exception):\n    """Base class for pipeline errors."""\n\n\nclass InputValidationError(FraudScreeningError):\n    """Input transaction is missing, malformed, or semantically invalid."""\n\n\nclass HistoryValidationError(FraudScreeningError):\n    """Historical context violates the causal-history rules."""\n\n\nclass PreprocessingContractError(FraudScreeningError):\n    """Preprocessing input/output does not match the frozen representation."""\n\n\nclass ArtifactNotFoundError(FraudScreeningError):\n    """A required official artifact cannot be found."""\n\n\nclass ArtifactFingerprintError(FraudScreeningError):\n    """An artifact fingerprint does not match the expected identity."""\n\n\nclass ArtifactCompatibilityError(FraudScreeningError):\n    """An artifact loads but is incompatible with the expected model state."""\n\n\nclass InferenceContractError(FraudScreeningError):\n    """Inference output violates the stable screening interface."""\n', 'final_pipeline/src/fraud_screening/data/__init__.py': '"""Input schema, parsing, and canonical transaction representation."""\n\nfrom fraud_screening.data.models import CanonicalTransaction\nfrom fraud_screening.data.parsing import (\n    extract_scoring_payload,\n    parse_scoring_payload,\n)\nfrom fraud_screening.data.schema import (\n    RAW_SCORING_FIELDS,\n    SEMANTIC_LOCATION_STATES,\n)\n\n__all__ = [\n    "CanonicalTransaction",\n    "RAW_SCORING_FIELDS",\n    "SEMANTIC_LOCATION_STATES",\n    "extract_scoring_payload",\n    "parse_scoring_payload",\n]\n', 'final_pipeline/src/fraud_screening/data/schema.py': '"""Stable raw-input schema for transaction screening."""\n\nRAW_SCORING_FIELDS = (\n    "User",\n    "Card",\n    "Year",\n    "Month",\n    "Day",\n    "Time",\n    "Amount",\n    "Use Chip",\n    "Merchant Name",\n    "Merchant City",\n    "Merchant State",\n    "Zip",\n)\n\nEXCLUDED_DATASET_FIELDS = (\n    "MCC",\n    "Errors?",\n    "Is Fraud?",\n)\n\nSEMANTIC_LOCATION_STATES = (\n    "NON_PHYSICAL_OR_ONLINE",\n    "PHYSICAL_COMPLETE",\n    "PHYSICAL_ZIP_UNAVAILABLE",\n)\n', 'final_pipeline/src/fraud_screening/data/models.py': '"""Canonical in-memory transaction representation."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom datetime import datetime\n\n\n@dataclass(frozen=True, slots=True)\nclass CanonicalTransaction:\n    """Validated transaction state before historical feature construction."""\n\n    user_id: int\n    card_id: int\n    timestamp: datetime\n    amount_numeric: float\n    transaction_mode: str\n    merchant_id: int\n    merchant_city: str\n    merchant_state: str | None\n    zip_code: int | None\n    location_state: str\n', 'final_pipeline/src/fraud_screening/data/parsing.py': '"""Deterministic parsing and validation for scoring transactions."""\n\nfrom __future__ import annotations\n\nimport math\nimport re\nfrom collections.abc import Mapping\nfrom datetime import datetime\nfrom typing import Any\n\nfrom fraud_screening.data.models import CanonicalTransaction\nfrom fraud_screening.data.schema import RAW_SCORING_FIELDS\nfrom fraud_screening.errors import InputValidationError\n\n\n_TIME_PATTERN = re.compile(r"^\\d{2}:\\d{2}$")\n\n\ndef _is_missing(value: Any) -> bool:\n    if value is None:\n        return True\n\n    if isinstance(value, float):\n        return math.isnan(value)\n\n    try:\n        result = value != value\n    except Exception:\n        return False\n\n    try:\n        return bool(result)\n    except Exception:\n        return False\n\n\ndef _required_value(payload: Mapping[str, Any], field: str) -> Any:\n    if field not in payload:\n        raise InputValidationError(\n            f"Missing required field: {field}"\n        )\n\n    value = payload[field]\n\n    if _is_missing(value):\n        raise InputValidationError(\n            f"Required field is missing: {field}"\n        )\n\n    return value\n\n\ndef _parse_integer_identifier(value: Any, field: str) -> int:\n    if isinstance(value, bool):\n        raise InputValidationError(\n            f"{field} must be an integer identifier."\n        )\n\n    if isinstance(value, int):\n        return value\n\n    if isinstance(value, float):\n        if not math.isfinite(value) or not value.is_integer():\n            raise InputValidationError(\n                f"{field} must be an integer identifier."\n            )\n        return int(value)\n\n    text = str(value).strip()\n\n    if not text:\n        raise InputValidationError(\n            f"{field} must not be blank."\n        )\n\n    try:\n        return int(text)\n    except ValueError as exc:\n        raise InputValidationError(\n            f"{field} must be an integer identifier."\n        ) from exc\n\n\ndef _parse_timestamp(payload: Mapping[str, Any]) -> datetime:\n    year = _parse_integer_identifier(\n        _required_value(payload, "Year"),\n        "Year",\n    )\n    month = _parse_integer_identifier(\n        _required_value(payload, "Month"),\n        "Month",\n    )\n    day = _parse_integer_identifier(\n        _required_value(payload, "Day"),\n        "Day",\n    )\n\n    time_text = str(\n        _required_value(payload, "Time")\n    ).strip()\n\n    if not _TIME_PATTERN.fullmatch(time_text):\n        raise InputValidationError(\n            "Time must use HH:MM format."\n        )\n\n    try:\n        hour = int(time_text[:2])\n        minute = int(time_text[3:5])\n\n        return datetime(\n            year,\n            month,\n            day,\n            hour,\n            minute,\n        )\n    except ValueError as exc:\n        raise InputValidationError(\n            "Invalid Year/Month/Day/Time combination."\n        ) from exc\n\n\ndef _parse_amount(value: Any) -> float:\n    text = (\n        str(value)\n        .replace("$", "")\n        .replace(",", "")\n        .strip()\n    )\n\n    if not text:\n        raise InputValidationError(\n            "Amount must not be blank."\n        )\n\n    try:\n        amount = float(text)\n    except ValueError as exc:\n        raise InputValidationError(\n            "Amount cannot be parsed as a number."\n        ) from exc\n\n    if not math.isfinite(amount):\n        raise InputValidationError(\n            "Amount must be finite."\n        )\n\n    return amount\n\n\ndef _parse_required_string(value: Any, field: str) -> str:\n    text = str(value).strip()\n\n    if not text:\n        raise InputValidationError(\n            f"{field} must not be blank."\n        )\n\n    return text\n\n\ndef _parse_optional_state(value: Any) -> str | None:\n    if _is_missing(value):\n        return None\n\n    text = str(value).strip()\n\n    if not text:\n        raise InputValidationError(\n            "Merchant State must be missing or a non-blank string."\n        )\n\n    return text\n\n\ndef _parse_optional_zip(value: Any) -> int | None:\n    if _is_missing(value):\n        return None\n\n    if isinstance(value, bool):\n        raise InputValidationError(\n            "Zip must be an integer categorical code when present."\n        )\n\n    if isinstance(value, int):\n        return value\n\n    if isinstance(value, float):\n        if not math.isfinite(value) or not value.is_integer():\n            raise InputValidationError(\n                "Zip must be an integer categorical code when present."\n            )\n        return int(value)\n\n    text = str(value).strip()\n\n    if not text:\n        raise InputValidationError(\n            "Zip must be missing or an integer categorical code."\n        )\n\n    try:\n        numeric = float(text)\n    except ValueError as exc:\n        raise InputValidationError(\n            "Zip must be an integer categorical code when present."\n        ) from exc\n\n    if not math.isfinite(numeric) or not numeric.is_integer():\n        raise InputValidationError(\n            "Zip must be an integer categorical code when present."\n        )\n\n    return int(numeric)\n\n\ndef _derive_location_state(\n    merchant_city: str,\n    merchant_state: str | None,\n    zip_code: int | None,\n) -> str:\n    city_online = merchant_city.upper() == "ONLINE"\n    state_missing = merchant_state is None\n    zip_missing = zip_code is None\n\n    if city_online and state_missing and zip_missing:\n        return "NON_PHYSICAL_OR_ONLINE"\n\n    if (\n        not city_online\n        and not state_missing\n        and not zip_missing\n    ):\n        return "PHYSICAL_COMPLETE"\n\n    if (\n        not city_online\n        and not state_missing\n        and zip_missing\n    ):\n        return "PHYSICAL_ZIP_UNAVAILABLE"\n\n    raise InputValidationError(\n        "Location fields form an inconsistent semantic state."\n    )\n\n\ndef extract_scoring_payload(\n    raw_record: Mapping[str, Any],\n) -> dict[str, Any]:\n    """Select only the authorized scoring fields from a dataset record."""\n\n    missing = [\n        field\n        for field in RAW_SCORING_FIELDS\n        if field not in raw_record\n    ]\n\n    if missing:\n        raise InputValidationError(\n            "Missing required scoring fields: "\n            + ", ".join(missing)\n        )\n\n    return {\n        field: raw_record[field]\n        for field in RAW_SCORING_FIELDS\n    }\n\n\ndef parse_scoring_payload(\n    payload: Mapping[str, Any],\n) -> CanonicalTransaction:\n    """Validate an exact scoring payload and build canonical transaction state."""\n\n    expected = set(RAW_SCORING_FIELDS)\n    observed = set(payload.keys())\n\n    missing = sorted(expected - observed)\n    extra = sorted(observed - expected)\n\n    if missing:\n        raise InputValidationError(\n            "Missing required scoring fields: "\n            + ", ".join(missing)\n        )\n\n    if extra:\n        raise InputValidationError(\n            "Unexpected scoring fields: "\n            + ", ".join(extra)\n        )\n\n    user_id = _parse_integer_identifier(\n        _required_value(payload, "User"),\n        "User",\n    )\n\n    card_id = _parse_integer_identifier(\n        _required_value(payload, "Card"),\n        "Card",\n    )\n\n    merchant_id = _parse_integer_identifier(\n        _required_value(payload, "Merchant Name"),\n        "Merchant Name",\n    )\n\n    timestamp = _parse_timestamp(payload)\n\n    amount_numeric = _parse_amount(\n        _required_value(payload, "Amount")\n    )\n\n    transaction_mode = _parse_required_string(\n        _required_value(payload, "Use Chip"),\n        "Use Chip",\n    )\n\n    merchant_city = _parse_required_string(\n        _required_value(payload, "Merchant City"),\n        "Merchant City",\n    )\n\n    merchant_state = _parse_optional_state(\n        payload["Merchant State"]\n    )\n\n    zip_code = _parse_optional_zip(\n        payload["Zip"]\n    )\n\n    location_state = _derive_location_state(\n        merchant_city=merchant_city,\n        merchant_state=merchant_state,\n        zip_code=zip_code,\n    )\n\n    return CanonicalTransaction(\n        user_id=user_id,\n        card_id=card_id,\n        timestamp=timestamp,\n        amount_numeric=amount_numeric,\n        transaction_mode=transaction_mode,\n        merchant_id=merchant_id,\n        merchant_city=merchant_city,\n        merchant_state=merchant_state,\n        zip_code=zip_code,\n        location_state=location_state,\n    )\n', 'final_pipeline/tests/__init__.py': '', 'final_pipeline/tests/unit/__init__.py': '', 'final_pipeline/tests/unit/test_data_boundary.py': 'from __future__ import annotations\n\nimport math\nimport unittest\nfrom datetime import datetime\n\nfrom fraud_screening.data import (\n    extract_scoring_payload,\n    parse_scoring_payload,\n)\nfrom fraud_screening.errors import InputValidationError\n\n\ndef valid_payload() -> dict:\n    return {\n        "User": 0,\n        "Card": 0,\n        "Year": 2002,\n        "Month": 9,\n        "Day": 1,\n        "Time": "06:21",\n        "Amount": "$134.09",\n        "Use Chip": " Swipe Transaction ",\n        "Merchant Name": 3527213246127876953,\n        "Merchant City": " La Verne ",\n        "Merchant State": "CA",\n        "Zip": 91750.0,\n    }\n\n\nclass DataBoundaryTests(unittest.TestCase):\n    def test_valid_physical_transaction(self) -> None:\n        tx = parse_scoring_payload(valid_payload())\n\n        self.assertEqual(tx.user_id, 0)\n        self.assertEqual(tx.card_id, 0)\n        self.assertEqual(\n            tx.timestamp,\n            datetime(2002, 9, 1, 6, 21),\n        )\n        self.assertEqual(tx.amount_numeric, 134.09)\n        self.assertEqual(\n            tx.transaction_mode,\n            "Swipe Transaction",\n        )\n        self.assertEqual(\n            tx.merchant_id,\n            3527213246127876953,\n        )\n        self.assertEqual(tx.merchant_city, "La Verne")\n        self.assertEqual(tx.merchant_state, "CA")\n        self.assertEqual(tx.zip_code, 91750)\n        self.assertEqual(\n            tx.location_state,\n            "PHYSICAL_COMPLETE",\n        )\n\n    def test_online_location(self) -> None:\n        payload = valid_payload()\n        payload["Merchant City"] = "ONLINE"\n        payload["Merchant State"] = None\n        payload["Zip"] = math.nan\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(\n            tx.location_state,\n            "NON_PHYSICAL_OR_ONLINE",\n        )\n\n    def test_physical_zip_unavailable(self) -> None:\n        payload = valid_payload()\n        payload["Zip"] = None\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(\n            tx.location_state,\n            "PHYSICAL_ZIP_UNAVAILABLE",\n        )\n\n    def test_negative_amount_is_preserved(self) -> None:\n        payload = valid_payload()\n        payload["Amount"] = "-$12.50"\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(tx.amount_numeric, -12.50)\n\n    def test_zero_amount_is_preserved(self) -> None:\n        payload = valid_payload()\n        payload["Amount"] = "$0.00"\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(tx.amount_numeric, 0.0)\n\n    def test_amount_with_comma_is_parsed(self) -> None:\n        payload = valid_payload()\n        payload["Amount"] = "$1,234.50"\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(tx.amount_numeric, 1234.50)\n\n    def test_unknown_transaction_mode_is_not_rejected(self) -> None:\n        payload = valid_payload()\n        payload["Use Chip"] = "Future Mode"\n\n        tx = parse_scoring_payload(payload)\n\n        self.assertEqual(\n            tx.transaction_mode,\n            "Future Mode",\n        )\n\n    def test_invalid_timestamp_fails(self) -> None:\n        payload = valid_payload()\n        payload["Time"] = "25:99"\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_invalid_amount_fails(self) -> None:\n        payload = valid_payload()\n        payload["Amount"] = "not-money"\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_inconsistent_location_fails(self) -> None:\n        payload = valid_payload()\n        payload["Merchant City"] = "ONLINE"\n        payload["Merchant State"] = "CA"\n        payload["Zip"] = None\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_fractional_zip_fails(self) -> None:\n        payload = valid_payload()\n        payload["Zip"] = 91750.5\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_missing_required_field_fails(self) -> None:\n        payload = valid_payload()\n        del payload["Amount"]\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_exact_scoring_payload_rejects_extra_fields(self) -> None:\n        payload = valid_payload()\n        payload["MCC"] = 5300\n\n        with self.assertRaises(InputValidationError):\n            parse_scoring_payload(payload)\n\n    def test_dataset_adapter_drops_excluded_fields(self) -> None:\n        raw = valid_payload()\n        raw["MCC"] = 5300\n        raw["Errors?"] = None\n        raw["Is Fraud?"] = "No"\n\n        scoring = extract_scoring_payload(raw)\n\n        self.assertEqual(\n            set(scoring),\n            set(valid_payload()),\n        )\n        self.assertNotIn("MCC", scoring)\n        self.assertNotIn("Errors?", scoring)\n        self.assertNotIn("Is Fraud?", scoring)\n\n    def test_dataset_adapter_requires_all_scoring_fields(self) -> None:\n        raw = valid_payload()\n        del raw["Merchant Name"]\n\n        with self.assertRaises(InputValidationError):\n            extract_scoring_payload(raw)\n\n\nif __name__ == "__main__":\n    unittest.main()\n'}


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
        help="Create the package/data-boundary files.",
    )
    args = parser.parse_args()

    root = detect_root()

    collisions = [
        relative
        for relative in FILES
        if (root / relative).exists()
    ]

    print("=" * 84)
    print("FINAL PIPELINE — DATA BOUNDARY BOOTSTRAP")
    print("=" * 84)
    print("Project root:", root)
    print(
        "Mode:",
        "EXECUTE — NEW FILES ONLY"
        if args.execute
        else "DRY RUN — NO FILES WRITTEN",
    )

    print("\n[1] Planned files")
    for relative in FILES:
        print(" -", relative)

    print("\n[2] Collision gate")
    print(" - collisions:", len(collisions))
    for relative in collisions:
        print("   *", relative)

    if collisions:
        print("\nRESULT: STOP")
        print(
            "Existing target files detected; "
            "nothing will be overwritten."
        )
        raise SystemExit(1)

    print("\n[3] Scientific/runtime boundary")
    print(" - model fit: NO")
    print(" - preprocessing fit: NO")
    print(" - preprocessing transform: NO")
    print(" - prediction: NO")
    print(" - research files modified: NO")
    print(" - official artifacts modified: NO")

    if not args.execute:
        print("\nDRY-RUN RESULT: PASS")
        print("Execute with:")
        print(
            "python bootstrap_data_boundary.py --execute"
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

    print("\n[4] Created")
    for path in created:
        print(" -", path.relative_to(root))

    print("\nBOOTSTRAP RESULT: PASS")
    print("Next verification command:")
    print(
        "PYTHONPATH=final_pipeline/src "
        "python -m unittest discover "
        "-s final_pipeline/tests -v"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
