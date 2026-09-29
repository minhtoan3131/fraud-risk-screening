from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_VERSION = "M9.4.3-history-feature-contract-v1"

SOURCE_REGISTRY_REL = (
    "final_pipeline/docs/contracts/contract_source_registry.json"
)
INPUT_CONTRACT_REL = (
    "final_pipeline/docs/contracts/transaction_input_data_contract.json"
)
OUTPUT_REL = (
    "final_pipeline/docs/contracts/history_feature_contract.json"
)

SEMANTIC_FEATURE_ORDER = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]

CURRENT_TRANSACTION_FEATURES = [
    "amount_numeric",
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]

BEHAVIORAL_FEATURES = [
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
    "is_new_merchant",
    "has_prior_card_history",
]

HISTORY_KEYS = [
    "User",
    "Card",
]

MERCHANT_HISTORY_KEY = "Merchant Name"


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


def build_contract() -> dict:
    feature_definitions = {
        "amount_numeric": {
            "group": "current_transaction",
            "type": "numeric",
            "source": "Amount",
            "definition": (
                "Parsed signed numeric Amount from current transaction."
            ),
            "prediction_time_available": True,
            "history_required": False,
        },
        "time_since_previous_transaction_min": {
            "group": "behavioral",
            "type": "numeric_with_structural_na",
            "entity": "User + Card",
            "definition": (
                "Minutes from current Timestamp to the nearest "
                "distinct strict-prior Timestamp for the same card."
            ),
            "history_rule": "Timestamp(history) < Timestamp(current)",
            "same_timestamp_peers_allowed": False,
            "cold_start_value": None,
        },
        "transactions_last_1h": {
            "group": "behavioral",
            "type": "numeric",
            "entity": "User + Card",
            "definition": (
                "Count of strict-prior card transactions in "
                "[T - 1 hour, T)."
            ),
            "history_window": "[T - 1 hour, T)",
            "current_transaction_included": False,
            "same_timestamp_peers_included": False,
            "cold_start_value": 0,
            "zero_is_valid_observed_value": True,
        },
        "amount_minus_previous_mean": {
            "group": "behavioral",
            "type": "numeric_with_structural_na",
            "entity": "User + Card",
            "definition": (
                "Current amount_numeric minus mean Amount over "
                "all strict-prior transactions for the same card."
            ),
            "history_scope": "full strict-prior card history",
            "cold_start_value": None,
        },
        "is_new_merchant": {
            "group": "behavioral",
            "type": "boolean",
            "entity": "User + Card + Merchant Name",
            "definition": (
                "True iff current Merchant Name has never appeared "
                "for the same card at a strict-prior Timestamp."
            ),
            "same_timestamp_first_occurrences": (
                "All remain True because same-timestamp peers "
                "cannot provide history to each other."
            ),
            "cold_start_value": True,
        },
        "has_prior_card_history": {
            "group": "behavioral",
            "type": "boolean",
            "entity": "User + Card",
            "definition": (
                "True iff at least one transaction exists for the "
                "same card with Timestamp(history) < Timestamp(current)."
            ),
            "cold_start_value": False,
            "role": (
                "Companion state that preserves cold-start semantics "
                "after numerical structural NA handling."
            ),
        },
        "transaction_mode": {
            "group": "current_transaction",
            "type": "categorical",
            "source": "Use Chip",
            "definition": (
                "Deterministically stripped transaction-mode string."
            ),
            "history_required": False,
        },
        "location_state": {
            "group": "current_transaction",
            "type": "categorical",
            "sources": [
                "Merchant City",
                "Merchant State",
                "Zip",
            ],
            "definition": (
                "Locked low-cardinality semantic location state."
            ),
            "history_required": False,
        },
        "hour_of_day": {
            "group": "current_transaction",
            "type": "categorical_integer",
            "source": "Timestamp",
            "domain": "0..23",
            "history_required": False,
        },
        "day_of_week": {
            "group": "current_transaction",
            "type": "categorical_integer",
            "source": "Timestamp",
            "domain": "0..6",
            "history_required": False,
        },
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "scope": (
            "Strict-causal history semantics and exact ten-feature "
            "semantic interface before preprocessing."
        ),
        "history_contract": {
            "primary_entity_key": HISTORY_KEYS,
            "merchant_history_key": MERCHANT_HISTORY_KEY,
            "strict_causal_rule": (
                "Timestamp(history) < Timestamp(current)"
            ),
            "forbidden_history_sources": [
                "current transaction",
                "same-timestamp peer",
                "future transaction",
                "future-inclusive aggregate",
                "target label / Is Fraud?",
            ],
            "same_timestamp_policy": {
                "interpretation": "same prediction point",
                "peer_history_allowed": False,
                "state_update_timing": (
                    "Update behavioral state only after all transactions "
                    "in the current Timestamp group have been scored/"
                    "feature-computed."
                ),
            },
            "historical_warmup": {
                "allowed": True,
                "preserved": True,
                "reset_at_w_long_start": False,
                "reset_at_w_short_start": False,
                "reset_at_validation_start": False,
                "principle": (
                    "Classifier-training membership and history "
                    "availability are separate concepts."
                ),
            },
            "lookback_semantics": {
                "recency": "nearest strict-prior distinct Timestamp",
                "velocity": "[T - 1 hour, T)",
                "historical_amount": "full strict-prior card history",
                "merchant_novelty": "full strict-prior card history",
            },
            "cold_start": {
                "condition": "no strict-prior User+Card transaction",
                "values": {
                    "has_prior_card_history": False,
                    "time_since_previous_transaction_min": None,
                    "transactions_last_1h": 0,
                    "amount_minus_previous_mean": None,
                    "is_new_merchant": True,
                },
                "na_semantics": (
                    "None/NA here means STRUCTURAL COLD-START STATE, "
                    "not data-quality missingness."
                ),
            },
            "raw_identifiers": {
                "User": "history key only",
                "Card": "history key only",
                "Merchant Name": "merchant-history key only",
                "direct_classifier_use": False,
            },
            "target_history_allowed": False,
        },
        "feature_contract": {
            "feature_count_pre_encoding": 10,
            "semantic_feature_order": SEMANTIC_FEATURE_ORDER,
            "current_transaction_features":
                CURRENT_TRANSACTION_FEATURES,
            "behavioral_features": BEHAVIORAL_FEATURES,
            "definitions": feature_definitions,
            "support_states_not_direct_features": [
                "prior_card_transaction_count",
                "previous_amount_mean",
            ],
            "excluded_candidate_features": [
                "mcc_code",
                "merchant_state_cat",
                "merchant_city_cat",
                "zip_cat",
                "month_of_year",
                "is_weekend",
                "amount_signed_log1p",
                "is_negative_amount",
                "is_zero_amount",
                "prior_card_transaction_count",
                "previous_amount_mean",
            ],
        },
        "implementation_invariants": {
            "raw_row_order_is_history_semantics": False,
            "simple_shift_without_timestamp_grouping_allowed": False,
            "same_timestamp_grouping_required": True,
            "future_history_violation_allowed_count": 0,
            "same_timestamp_history_violation_allowed_count": 0,
            "target_history_violation_allowed_count": 0,
        },
        "source_trace": {
            "behavioral_runtime_evidence":
                "CANON-M4.5 — causal behavioral features",
            "consolidated_contract":
                "CANON-M4.8 — Feature/Preprocessing/Behavioral Contract",
            "m9_1_identity":
                "09_01_artifact_inventory_identity_audit.ipynb",
            "upstream_data_contract":
                INPUT_CONTRACT_REL,
            "source_registry":
                SOURCE_REGISTRY_REL,
        },
        "guardrails": {
            "model_fit_performed": False,
            "prediction_performed": False,
            "preprocessing_fit_performed": False,
            "preprocessing_transform_performed": False,
            "feature_values_computed_on_dataset": False,
            "feature_semantics_changed": False,
            "threshold_changed": False,
        },
        "next_step": (
            "M9.4.4 — Preprocessing + Model + Threshold Contract"
        ),
    }


def validate(
    source_registry: dict,
    input_contract: dict,
    contract: dict,
) -> dict:
    history = contract["history_contract"]
    feature = contract["feature_contract"]
    definitions = feature["definitions"]

    gates = {}

    gates["G01_SOURCE_REGISTRY_PASS"] = (
        source_registry.get("gate_status") == "PASS"
    )

    gates["G02_INPUT_DATA_CONTRACT_PASS"] = (
        input_contract.get("gate_status") == "PASS"
    )

    gates["G03_STRICT_CAUSAL_RULE_LOCKED"] = (
        history["strict_causal_rule"]
        == "Timestamp(history) < Timestamp(current)"
    )

    gates["G04_SAME_TIMESTAMP_PEERS_EXCLUDED"] = (
        history["same_timestamp_policy"]["peer_history_allowed"]
        is False
    )

    gates["G05_HISTORY_UPDATE_AFTER_TIMESTAMP_GROUP"] = (
        "after all transactions"
        in history["same_timestamp_policy"][
            "state_update_timing"
        ].lower()
    )

    gates["G06_HISTORICAL_WARMUP_PRESERVED"] = (
        history["historical_warmup"]["allowed"] is True
        and history["historical_warmup"]["preserved"] is True
        and history["historical_warmup"][
            "reset_at_w_short_start"
        ] is False
        and history["historical_warmup"][
            "reset_at_validation_start"
        ] is False
    )

    expected_cold_start = {
        "has_prior_card_history": False,
        "time_since_previous_transaction_min": None,
        "transactions_last_1h": 0,
        "amount_minus_previous_mean": None,
        "is_new_merchant": True,
    }
    gates["G07_COLD_START_EXACT"] = (
        history["cold_start"]["values"]
        == expected_cold_start
    )

    gates["G08_TEN_FEATURES_EXACT_ORDER"] = (
        feature["feature_count_pre_encoding"] == 10
        and feature["semantic_feature_order"]
        == SEMANTIC_FEATURE_ORDER
    )

    gates["G09_CURRENT_AND_BEHAVIORAL_SPLIT"] = (
        feature["current_transaction_features"]
        == CURRENT_TRANSACTION_FEATURES
        and feature["behavioral_features"]
        == BEHAVIORAL_FEATURES
    )

    gates["G10_HISTORY_FEATURE_DEFINITIONS_LOCKED"] = (
        definitions[
            "transactions_last_1h"
        ]["history_window"] == "[T - 1 hour, T)"
        and definitions[
            "amount_minus_previous_mean"
        ]["history_scope"]
        == "full strict-prior card history"
        and definitions[
            "is_new_merchant"
        ]["cold_start_value"] is True
    )

    gates["G11_IDENTIFIERS_NOT_DIRECT_FEATURES"] = (
        history["raw_identifiers"]["direct_classifier_use"]
        is False
    )

    gates["G12_TARGET_HISTORY_PROHIBITED"] = (
        history["target_history_allowed"] is False
    )

    invariants = contract["implementation_invariants"]
    gates["G13_CAUSAL_VIOLATION_TOLERANCE_ZERO"] = (
        invariants[
            "future_history_violation_allowed_count"
        ] == 0
        and invariants[
            "same_timestamp_history_violation_allowed_count"
        ] == 0
        and invariants[
            "target_history_violation_allowed_count"
        ] == 0
    )

    gates["G14_NO_NAIVE_ROW_SHIFT_CONTRACT"] = (
        invariants[
            "simple_shift_without_timestamp_grouping_allowed"
        ] is False
        and invariants["same_timestamp_grouping_required"]
        is True
    )

    gates["G15_NO_SCIENTIFIC_CHANGE"] = all(
        value is False
        for value in contract["guardrails"].values()
    )

    return gates


def main() -> None:
    root = detect_root()

    source_registry_path = root / SOURCE_REGISTRY_REL
    input_contract_path = root / INPUT_CONTRACT_REL

    if not source_registry_path.is_file():
        raise RuntimeError(
            "Thiếu M9.4.1 source registry."
        )
    if not input_contract_path.is_file():
        raise RuntimeError(
            "Thiếu M9.4.2 transaction/data contract."
        )

    source_registry = read_json(source_registry_path)
    input_contract = read_json(input_contract_path)

    print("=" * 86)
    print("M9.4.3 — HISTORY + FEATURE CONTRACT")
    print("=" * 86)
    print("Project root:", root)
    print(
        "Mode: governance/specification only — "
        "NO DATA FEATURE COMPUTATION / NO FIT / NO PREDICTION"
    )

    contract = build_contract()
    gates = validate(
        source_registry=source_registry,
        input_contract=input_contract,
        contract=contract,
    )

    print("\n[1] Strict-causal history")
    print(
        " -",
        contract["history_contract"]["strict_causal_rule"],
    )
    print(" - same-timestamp peer history: FORBIDDEN")
    print(" - future history: FORBIDDEN")
    print(" - target-label history: FORBIDDEN")
    print(
        " - timestamp-group state update: AFTER WHOLE GROUP"
    )

    print("\n[2] Historical warm-up")
    print(" - allowed: True")
    print(" - preserved: True")
    print(" - reset at W_SHORT start: False")
    print(" - reset at VALIDATION start: False")

    print("\n[3] Cold-start")
    for key, value in (
        contract["history_contract"]["cold_start"]["values"]
        .items()
    ):
        print(f" - {key}: {value}")

    print("\n[4] Exact semantic feature order")
    for index, feature_name in enumerate(
        SEMANTIC_FEATURE_ORDER,
        start=1,
    ):
        print(f" - {index:02d}. {feature_name}")

    print("\n[5] M9.4.3 gate")
    for gate, passed in gates.items():
        print(
            f" - {gate}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(gates.values()):
        raise RuntimeError(
            "M9.4.3 gate FAIL. Không ghi contract."
        )

    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise RuntimeError(
            "STOP: history/feature contract đã tồn tại; "
            "không ghi đè tự động:\n"
            f"{output_path}"
        )

    output_path.write_text(
        json.dumps(
            {
                **contract,
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

    print("\n" + "=" * 86)
    print("M9.4.3 HISTORY + FEATURE CONTRACT: PASS")
    print("Strict-causal history semantics locked.")
    print("Exact 10-feature semantic interface locked.")
    print("No dataset feature computation.")
    print("No model fit.")
    print("No prediction.")
    print(
        "NEXT: M9.4.4 — Preprocessing + Model + Threshold Contract"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
