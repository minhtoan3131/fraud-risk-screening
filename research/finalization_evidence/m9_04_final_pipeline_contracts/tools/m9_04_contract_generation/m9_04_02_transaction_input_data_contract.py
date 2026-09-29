from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_VERSION = "M9.4.2-transaction-input-data-contract-v1"

RAW_DATA_REL = (
    "research/data/raw/ibm_tabformer/card_transaction.v1.csv"
)

SOURCE_REGISTRY_REL = (
    "final_pipeline/docs/contracts/contract_source_registry.json"
)

OUTPUT_REL = (
    "final_pipeline/docs/contracts/transaction_input_data_contract.json"
)

EXPECTED_RAW_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]

SCORING_REQUIRED_FIELDS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
]

CONTEXT_ONLY_FIELDS = [
    "User",
    "Card",
    "Merchant Name",
]

CURRENT_TRANSACTION_FEATURE_SOURCES = [
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant City",
    "Merchant State",
    "Zip",
]

EXCLUDED_FROM_FROZEN_BASELINE = [
    "MCC",
    "Errors?",
    "Is Fraud?",
]

PROHIBITED_DIRECT_CLASSIFIER_FIELDS = [
    "User",
    "Card",
    "Merchant Name",
    "Errors?",
    "Is Fraud?",
    "Timestamp",
]

LOCKED_LOCATION_STATES = [
    "NON_PHYSICAL_OR_ONLINE",
    "PHYSICAL_COMPLETE",
    "PHYSICAL_ZIP_UNAVAILABLE",
]


def detect_root() -> Path:
    root = Path.cwd().resolve()

    required = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required):
        raise RuntimeError(
            "Không đứng ở project root sau M9.3."
        )

    return root


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_header(path: Path) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
        newline="",
    ) as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration as exc:
            raise RuntimeError(
                "Raw CSV rỗng; không đọc được header."
            ) from exc


def build_contract() -> dict:
    field_contract = {
        "User": {
            "required_for_scoring": True,
            "role": "identifier / User+Card history key",
            "observed_raw_type": "integer identifier",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
            "semantic_rule": (
                "Identifier equality is meaningful for history grouping; "
                "numeric magnitude is not a model feature."
            ),
        },
        "Card": {
            "required_for_scoring": True,
            "role": "identifier / User+Card history key",
            "observed_raw_type": "integer identifier",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
            "semantic_rule": (
                "Card is interpreted within User context; "
                "numeric magnitude is not a model feature."
            ),
        },
        "Year": {
            "required_for_scoring": True,
            "role": "Timestamp component",
            "observed_raw_type": "integer",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
        },
        "Month": {
            "required_for_scoring": True,
            "role": "Timestamp component",
            "observed_raw_type": "integer",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
        },
        "Day": {
            "required_for_scoring": True,
            "role": "Timestamp component",
            "observed_raw_type": "integer",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
        },
        "Time": {
            "required_for_scoring": True,
            "role": "Timestamp component",
            "observed_raw_type": "HH:MM string",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
        },
        "Amount": {
            "required_for_scoring": True,
            "role": "current transaction numerical source",
            "observed_raw_type": "currency-formatted string",
            "direct_classifier_feature": False,
            "derived_feature": "amount_numeric",
            "missing_policy": "NOT_ALLOWED",
            "normalization": [
                "cast to string",
                "remove literal $",
                "remove comma separators",
                "strip surrounding whitespace",
                "convert to numeric",
            ],
            "semantic_rule": (
                "Preserve sign, preserve zero, no abs(), "
                "no automatic clipping/winsorization."
            ),
        },
        "Use Chip": {
            "required_for_scoring": True,
            "role": "current transaction categorical source",
            "observed_raw_type": "string",
            "direct_classifier_feature": False,
            "derived_feature": "transaction_mode",
            "missing_policy": "NOT_ALLOWED",
            "normalization": "deterministic str.strip() only",
            "observed_core_values": [
                "Swipe Transaction",
                "Chip Transaction",
                "Online Transaction",
            ],
        },
        "Merchant Name": {
            "required_for_scoring": True,
            "role": "merchant history key",
            "observed_raw_type": "identifier",
            "direct_classifier_feature": False,
            "missing_policy": "NOT_ALLOWED",
            "derived_feature": "is_new_merchant",
            "semantic_rule": (
                "Used only for strict-prior merchant novelty state; "
                "not placed directly in classifier X."
            ),
        },
        "Merchant City": {
            "required_for_scoring": True,
            "role": "location semantic source",
            "observed_raw_type": "string",
            "direct_classifier_feature": False,
            "derived_feature": "location_state",
            "missing_policy": "NOT_EXPECTED_IN_OFFICIAL_ARTIFACT",
            "normalization": "deterministic str.strip()",
        },
        "Merchant State": {
            "required_for_scoring": True,
            "role": "location semantic source",
            "observed_raw_type": "string / missing",
            "direct_classifier_feature": False,
            "derived_feature": "location_state",
            "missing_policy": (
                "ALLOWED_AS_STRUCTURAL_LOCATION_MISSINGNESS"
            ),
            "normalization": "deterministic str.strip() when present",
        },
        "Zip": {
            "required_for_scoring": True,
            "role": "location semantic source",
            "observed_raw_type": "categorical code / missing",
            "direct_classifier_feature": False,
            "derived_feature": "location_state",
            "missing_policy": (
                "ALLOWED_AS_STRUCTURAL_LOCATION_MISSINGNESS"
            ),
            "semantic_rule": (
                "Zip is a categorical/location code, "
                "not a numerical magnitude."
            ),
        },
        "MCC": {
            "required_for_scoring": False,
            "role": "conditional experiment candidate only",
            "direct_classifier_feature": False,
            "frozen_baseline_use": "EXCLUDED",
        },
        "Errors?": {
            "required_for_scoring": False,
            "role": "excluded processing signal",
            "direct_classifier_feature": False,
            "frozen_baseline_use": "PROHIBITED",
            "reason": (
                "Prediction-time availability was not proven "
                "for Model V1."
            ),
        },
        "Is Fraud?": {
            "required_for_scoring": False,
            "role": "supervised-learning target",
            "direct_classifier_feature": False,
            "frozen_baseline_use": "PROHIBITED",
            "reason": (
                "Ground-truth label is never an inference input."
            ),
        },
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "scope": (
            "Canonical raw transaction data boundary for the "
            "frozen final screening pipeline."
        ),
        "layers": {
            "dataset_adapter": {
                "source": (
                    "IBM TabFormer card_transaction.v1.csv"
                ),
                "expected_raw_columns": EXPECTED_RAW_COLUMNS,
                "column_count": 15,
                "responsibility": (
                    "Read/validate dataset rows and extract only "
                    "fields authorized for the screening path."
                ),
            },
            "scoring_payload": {
                "required_fields": SCORING_REQUIRED_FIELDS,
                "field_count": len(SCORING_REQUIRED_FIELDS),
                "extra_field_policy": (
                    "Frozen inference logic must not consume extra "
                    "dataset fields implicitly."
                ),
                "target_allowed": False,
                "errors_field_allowed": False,
                "mcc_used_by_frozen_model": False,
            },
            "canonical_internal_transaction": {
                "required_derived_values": [
                    "Timestamp",
                    "amount_numeric",
                    "transaction_mode",
                    "location_state",
                ],
                "context_keys": CONTEXT_ONLY_FIELDS,
                "note": (
                    "This layer is an internal semantic representation; "
                    "it is not the 47-column classifier matrix."
                ),
            },
        },
        "fields": field_contract,
        "timestamp_contract": {
            "sources": [
                "Year",
                "Month",
                "Day",
                "Time",
            ],
            "time_format": "HH:MM",
            "resolution": "minute",
            "construction": (
                "date = Year/Month/Day; "
                "time delta = Time + ':00'; "
                "Timestamp = date + time delta"
            ),
            "parse_failure_policy": "INVALID_INPUT",
            "raw_row_order_may_replace_timestamp": False,
            "classifier_direct_feature": False,
            "uses": [
                "history ordering",
                "strict causal boundary",
                "rolling windows",
                "calendar feature derivation",
            ],
        },
        "amount_contract": {
            "source": "Amount",
            "output": "amount_numeric",
            "parser": [
                "astype string",
                "remove $",
                "remove comma",
                "strip whitespace",
                "numeric conversion",
            ],
            "parse_failure_policy": "INVALID_INPUT",
            "preserve_negative": True,
            "preserve_zero": True,
            "automatic_abs": False,
            "automatic_clipping": False,
        },
        "transaction_mode_contract": {
            "source": "Use Chip",
            "output": "transaction_mode",
            "normalization": "str.strip()",
            "automatic_casefold": False,
            "automatic_fuzzy_merge": False,
        },
        "location_contract": {
            "sources": [
                "Merchant City",
                "Merchant State",
                "Zip",
            ],
            "output": "location_state",
            "locked_states": LOCKED_LOCATION_STATES,
            "rules": {
                "NON_PHYSICAL_OR_ONLINE": (
                    "Merchant City normalized to ONLINE "
                    "and Merchant State missing and Zip missing"
                ),
                "PHYSICAL_COMPLETE": (
                    "Merchant City not ONLINE "
                    "and Merchant State present and Zip present"
                ),
                "PHYSICAL_ZIP_UNAVAILABLE": (
                    "Merchant City not ONLINE "
                    "and Merchant State present and Zip missing"
                ),
            },
            "other_combination": (
                "OTHER_INCONSISTENT in research implementation; "
                "must not be silently coerced into a locked state. "
                "Final error behavior is locked in M9.4.5."
            ),
        },
        "classifier_boundary": {
            "context_only_fields": CONTEXT_ONLY_FIELDS,
            "current_transaction_feature_sources":
                CURRENT_TRANSACTION_FEATURE_SOURCES,
            "prohibited_direct_classifier_fields":
                PROHIBITED_DIRECT_CLASSIFIER_FIELDS,
            "excluded_from_frozen_baseline":
                EXCLUDED_FROM_FROZEN_BASELINE,
            "statement": (
                "Raw transaction fields are validated/derived first; "
                "the classifier receives only the locked semantic "
                "feature/preprocessing representation."
            ),
        },
        "source_trace": {
            "raw_schema": (
                "M1.5 technical audit + M1.8 data dictionary"
            ),
            "timestamp_and_amount": "M4.2",
            "data_quality": "M4.3",
            "transaction_representation": "M4.4",
            "frozen_core_feature_boundary": "M4.6 / M4.8",
            "contract_source_registry":
                SOURCE_REGISTRY_REL,
        },
        "guardrails": {
            "model_fit_performed": False,
            "prediction_performed": False,
            "preprocessing_fit_performed": False,
            "preprocessing_transform_performed": False,
            "feature_semantics_changed": False,
            "threshold_changed": False,
        },
        "next_step": (
            "M9.4.3 — History + Feature Contract"
        ),
    }


def validate_contract(
    contract: dict,
    raw_header: list[str],
    source_registry: dict,
) -> dict:
    fields = contract["fields"]
    scoring = contract["layers"]["scoring_payload"]

    gates = {}

    gates["G01_SOURCE_REGISTRY_PASS"] = (
        source_registry.get("gate_status") == "PASS"
    )

    gates["G02_RAW_SCHEMA_15_COLUMNS"] = (
        raw_header == EXPECTED_RAW_COLUMNS
    )

    gates["G03_SCORING_FIELDS_EXACT"] = (
        scoring["required_fields"]
        == SCORING_REQUIRED_FIELDS
        and scoring["field_count"] == 12
    )

    gates["G04_REQUIRED_FIELDS_IN_RAW_SCHEMA"] = all(
        field in EXPECTED_RAW_COLUMNS
        for field in SCORING_REQUIRED_FIELDS
    )

    gates["G05_CONTEXT_KEYS_NOT_DIRECT_FEATURES"] = all(
        fields[field]["direct_classifier_feature"] is False
        for field in CONTEXT_ONLY_FIELDS
    )

    gates["G06_TARGET_AND_ERRORS_PROHIBITED"] = (
        fields["Is Fraud?"]["frozen_baseline_use"]
        == "PROHIBITED"
        and fields["Errors?"]["frozen_baseline_use"]
        == "PROHIBITED"
    )

    gates["G07_MCC_NOT_IN_FROZEN_BASELINE"] = (
        fields["MCC"]["frozen_baseline_use"]
        == "EXCLUDED"
    )

    gates["G08_TIMESTAMP_DERIVATION_LOCKED"] = (
        contract["timestamp_contract"]["sources"]
        == ["Year", "Month", "Day", "Time"]
        and contract["timestamp_contract"][
            "raw_row_order_may_replace_timestamp"
        ] is False
    )

    gates["G09_AMOUNT_SEMANTICS_LOCKED"] = (
        contract["amount_contract"]["preserve_negative"]
        and contract["amount_contract"]["preserve_zero"]
        and not contract["amount_contract"]["automatic_abs"]
        and not contract["amount_contract"]["automatic_clipping"]
    )

    gates["G10_LOCATION_STATES_LOCKED"] = (
        contract["location_contract"]["locked_states"]
        == LOCKED_LOCATION_STATES
    )

    gates["G11_NO_SCIENTIFIC_CHANGE"] = all(
        value is False
        for value in contract["guardrails"].values()
    )

    return gates


def main() -> None:
    root = detect_root()

    source_registry_path = root / SOURCE_REGISTRY_REL
    if not source_registry_path.is_file():
        raise RuntimeError(
            "Thiếu contract_source_registry.json. "
            "M9.4.1 phải PASS trước."
        )

    source_registry = read_json(source_registry_path)

    raw_path = root / RAW_DATA_REL
    if not raw_path.is_file():
        raise RuntimeError(
            "Không tìm thấy raw IBM TabFormer CSV:\n"
            f"{raw_path}"
        )

    print("=" * 84)
    print("M9.4.2 — TRANSACTION INPUT + DATA CONTRACT")
    print("=" * 84)
    print("Project root:", root)
    print("Raw dataset:", raw_path.relative_to(root))
    print(
        "Mode: header + governance validation only — "
        "NO FULL DATA LOAD / NO FIT / NO PREDICTION"
    )

    raw_header = read_csv_header(raw_path)

    print("\n[1] Raw header")
    print(" - column_count:", len(raw_header))
    for column in raw_header:
        print(" -", column)

    contract = build_contract()
    gates = validate_contract(
        contract=contract,
        raw_header=raw_header,
        source_registry=source_registry,
    )

    print("\n[2] Canonical scoring payload")
    for field in SCORING_REQUIRED_FIELDS:
        print(" -", field)

    print("\n[3] Explicit non-input / excluded fields")
    print(" - MCC: not used by frozen baseline")
    print(" - Errors?: prohibited from frozen inference logic")
    print(" - Is Fraud?: target; prohibited from inference input")

    print("\n[4] Derived canonical values")
    print(" - Timestamp ← Year + Month + Day + Time")
    print(" - amount_numeric ← Amount")
    print(" - transaction_mode ← Use Chip")
    print(
        " - location_state ← Merchant City + Merchant State + Zip"
    )

    print("\n[5] M9.4.2 gate")
    for gate, passed in gates.items():
        print(f" - {gate}: {'PASS' if passed else 'FAIL'}")

    if not all(gates.values()):
        raise RuntimeError(
            "M9.4.2 gate FAIL. Không ghi contract."
        )

    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise RuntimeError(
            "STOP: transaction/data contract đã tồn tại; "
            "không ghi đè tự động:\n"
            f"{output_path}"
        )

    output_path.write_text(
        json.dumps(
            {
                **contract,
                "observed_raw_header": raw_header,
                "raw_header_verified": True,
                "gates": gates,
                "gate_status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n[6] Contract written")
    print(" -", output_path.relative_to(root))

    print("\n" + "=" * 84)
    print("M9.4.2 TRANSACTION INPUT + DATA CONTRACT: PASS")
    print("Raw schema verified from CSV header.")
    print("Scoring payload boundary locked.")
    print("No full dataset load.")
    print("No model fit.")
    print("No prediction.")
    print("NEXT: M9.4.3 — History + Feature Contract")
    print("=" * 84)


if __name__ == "__main__":
    main()
