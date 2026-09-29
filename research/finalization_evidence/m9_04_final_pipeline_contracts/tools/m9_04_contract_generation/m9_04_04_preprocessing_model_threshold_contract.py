from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_VERSION = "M9.4.4-preprocessing-model-threshold-contract-v1"

SOURCE_REGISTRY_REL = (
    "final_pipeline/docs/contracts/contract_source_registry.json"
)
HISTORY_FEATURE_CONTRACT_REL = (
    "final_pipeline/docs/contracts/history_feature_contract.json"
)
OUTPUT_REL = (
    "final_pipeline/docs/contracts/"
    "preprocessing_model_threshold_contract.json"
)

SOURCE_MODEL_REL = (
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_selected_rf_estimator.joblib"
)
SOURCE_PREPROCESSING_REL = (
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
)

OFFICIAL_MODEL_TARGET_REL = (
    "final_pipeline/artifacts/official/model/"
    "m8_02_selected_rf_estimator.joblib"
)
OFFICIAL_PREPROCESSING_TARGET_REL = (
    "final_pipeline/artifacts/official/preprocessing/"
    "m8_02_w_short_preprocessing_state.json"
)

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)
EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
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

NUMERIC_COLUMNS = [
    "amount_numeric",
    "time_since_previous_transaction_min",
    "transactions_last_1h",
    "amount_minus_previous_mean",
]

BOOLEAN_COLUMNS = [
    "is_new_merchant",
    "has_prior_card_history",
]

CATEGORICAL_COLUMNS = [
    "transaction_mode",
    "location_state",
    "hour_of_day",
    "day_of_week",
]

EXPECTED_MODEL_PARAMS = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "class_weight": "balanced",
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "max_samples": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": 42,
}

THRESHOLD = 0.50
POSITIVE_CLASS = 1
EXPECTED_CLASSES = [0, 1]
POSITIVE_CLASS_INDEX = 1


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_identity(path: Path) -> dict:
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thiếu joblib/scikit-learn trong .venv."
        ) from exc

    estimator = joblib.load(path)

    if not isinstance(estimator, RandomForestClassifier):
        raise RuntimeError(
            "Frozen estimator không phải RandomForestClassifier."
        )

    actual_params = estimator.get_params(deep=False)
    classes = estimator.classes_.tolist()

    return {
        "estimator_type": type(estimator).__name__,
        "sklearn_version_runtime": sklearn.__version__,
        "classes": classes,
        "positive_class_index": (
            classes.index(POSITIVE_CLASS)
            if POSITIVE_CLASS in classes
            else None
        ),
        "selected_params": {
            key: actual_params.get(key)
            for key in EXPECTED_MODEL_PARAMS
        },
    }


def build_contract(
    model_identity: dict,
    preprocessing_state: dict,
    model_sha: str,
    preprocessing_sha: str,
) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "scope": (
            "Frozen preprocessing representation, frozen estimator "
            "identity, risk-score interface and numerical threshold."
        ),
        "preprocessing_contract": {
            "input_feature_count": 10,
            "input_semantic_feature_order": SEMANTIC_FEATURE_ORDER,
            "branches": {
                "numeric": {
                    "columns": NUMERIC_COLUMNS,
                    "transform": "StandardScaler",
                    "learned_state_source": "W_SHORT TRAIN ONLY",
                    "fit_during_inference": False,
                    "structural_na_columns": [
                        "time_since_previous_transaction_min",
                        "amount_minus_previous_mean",
                    ],
                    "structural_na_policy": (
                        "Exclude NA from scaler fit statistics; "
                        "preserve NA through scaling; then map "
                        "structural NA to standardized 0.0."
                    ),
                },
                "boolean": {
                    "columns": BOOLEAN_COLUMNS,
                    "transform": "PASSTHROUGH",
                    "output_dtype": "float32",
                },
                "categorical": {
                    "columns": CATEGORICAL_COLUMNS,
                    "transform": "OneHotEncoder",
                    "vocabulary_source": "W_SHORT TRAIN ONLY",
                    "unknown_policy": "__UNKNOWN__",
                    "vocabulary_extension_during_inference": False,
                },
            },
            "output": {
                "feature_count": 47,
                "matrix_type": "CSR sparse matrix",
                "dtype": "float32",
                "feature_names_and_order": "FROZEN / EXACT",
            },
            "frozen_state_identity": {
                "current_source_path": SOURCE_PREPROCESSING_REL,
                "sha256": preprocessing_sha,
                "expected_sha256": EXPECTED_PREPROCESSING_SHA256,
                "promotion_target_path":
                    OFFICIAL_PREPROCESSING_TARGET_REL,
                "promotion_timing": "M9.6",
                "promotion_policy": (
                    "Byte-preserving copy/promotion; no refit."
                ),
            },
            "observed_state": {
                "feature_count":
                    preprocessing_state.get("feature_count"),
                "numeric_columns":
                    preprocessing_state.get("numeric_columns"),
                "feature_names_count": len(
                    preprocessing_state.get("feature_names", [])
                ),
                "numeric_mean_count": len(
                    preprocessing_state.get("numeric_mean", [])
                ),
                "numeric_scale_count": len(
                    preprocessing_state.get("numeric_scale", [])
                ),
            },
        },
        "model_contract": {
            "training_window": "W_SHORT",
            "family": "RandomForestClassifier",
            "config_id": "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
            "imbalance_strategy": "CLASS_WEIGHT_BALANCED",
            "expected_params": EXPECTED_MODEL_PARAMS,
            "classes": EXPECTED_CLASSES,
            "positive_class": POSITIVE_CLASS,
            "positive_class_index": POSITIVE_CLASS_INDEX,
            "probability_interface": "predict_proba",
            "fit_during_application_runtime": False,
            "deployment_refit_authorized": False,
            "frozen_artifact_identity": {
                "current_source_path": SOURCE_MODEL_REL,
                "sha256": model_sha,
                "expected_sha256": EXPECTED_MODEL_SHA256,
                "promotion_target_path":
                    OFFICIAL_MODEL_TARGET_REL,
                "promotion_timing": "M9.6",
                "promotion_policy": (
                    "Byte-preserving copy/promotion; "
                    "do not silently replace evaluated subject."
                ),
            },
            "observed_identity": model_identity,
        },
        "risk_score_contract": {
            "definition": (
                "predict_proba(X)[:, positive_class_index]"
            ),
            "positive_class": POSITIVE_CLASS,
            "positive_class_index": POSITIVE_CLASS_INDEX,
            "authorized_language": [
                "risk score",
                "positive-class score",
            ],
            "unauthorized_language": [
                "calibrated confidence",
                "real-world fraud probability",
            ],
            "calibration_performed": False,
        },
        "threshold_contract": {
            "threshold": THRESHOLD,
            "comparator": ">",
            "decision_rule": "predicted_positive = risk_score > 0.50",
            "boundary_examples": {
                "0.49": False,
                "0.50": False,
                "0.51": True,
            },
            "greater_or_equal_allowed": False,
            "runtime_threshold_tuning_allowed": False,
        },
        "pipeline_boundary": {
            "ordered_flow": [
                "10 semantic features",
                "frozen W_SHORT preprocessing transform",
                "47-column CSR float32",
                "frozen RandomForestClassifier",
                "predict_proba positive-class column",
                "risk_score",
                "strict risk_score > 0.50 decision",
            ],
            "inference_may_fit_preprocessing": False,
            "inference_may_fit_model": False,
            "inference_may_change_feature_order": False,
            "inference_may_change_threshold": False,
        },
        "source_trace": {
            "feature_history_contract":
                HISTORY_FEATURE_CONTRACT_REL,
            "source_registry":
                SOURCE_REGISTRY_REL,
            "preprocessing_semantics":
                "CANON-M4.8 + frozen M8.2 preprocessing state",
            "model_threshold":
                "CANON-M7.9",
            "evaluated_subject":
                "CANON-M8.6",
            "runtime_identity":
                "M9.1 artifact identity audit",
        },
        "guardrails": {
            "model_fit_performed": False,
            "prediction_performed": False,
            "predict_proba_performed": False,
            "preprocessing_fit_performed": False,
            "preprocessing_transform_performed": False,
            "artifact_copy_performed": False,
            "artifact_overwrite_performed": False,
            "threshold_changed": False,
        },
        "next_step": (
            "M9.4.5 — Inference Output + Artifact + Error Contract"
        ),
    }


def validate(
    source_registry: dict,
    history_feature_contract: dict,
    preprocessing_state: dict,
    model_identity: dict,
    model_sha: str,
    preprocessing_sha: str,
    contract: dict,
) -> dict:
    gates = {}

    gates["G01_SOURCE_REGISTRY_PASS"] = (
        source_registry.get("gate_status") == "PASS"
    )

    gates["G02_HISTORY_FEATURE_CONTRACT_PASS"] = (
        history_feature_contract.get("gate_status") == "PASS"
    )

    gates["G03_MODEL_SHA_MATCH"] = (
        model_sha == EXPECTED_MODEL_SHA256
    )

    gates["G04_PREPROCESSING_SHA_MATCH"] = (
        preprocessing_sha == EXPECTED_PREPROCESSING_SHA256
    )

    gates["G05_PREPROCESSING_10_TO_47"] = (
        preprocessing_state.get("feature_count") == 47
        and len(
            preprocessing_state.get("feature_names", [])
        ) == 47
        and contract["preprocessing_contract"][
            "input_feature_count"
        ] == 10
    )

    gates["G06_NUMERIC_STATE_IDENTITY"] = (
        preprocessing_state.get("numeric_columns")
        == NUMERIC_COLUMNS
        and len(
            preprocessing_state.get("numeric_mean", [])
        ) == 4
        and len(
            preprocessing_state.get("numeric_scale", [])
        ) == 4
    )

    gates["G07_PREPROCESSING_BRANCHES_LOCKED"] = (
        contract["preprocessing_contract"]["branches"][
            "numeric"
        ]["columns"] == NUMERIC_COLUMNS
        and contract["preprocessing_contract"]["branches"][
            "boolean"
        ]["columns"] == BOOLEAN_COLUMNS
        and contract["preprocessing_contract"]["branches"][
            "categorical"
        ]["columns"] == CATEGORICAL_COLUMNS
    )

    gates["G08_MODEL_TYPE_AND_CLASSES"] = (
        model_identity["estimator_type"]
        == "RandomForestClassifier"
        and model_identity["classes"] == EXPECTED_CLASSES
        and model_identity["positive_class_index"]
        == POSITIVE_CLASS_INDEX
    )

    gates["G09_MODEL_PARAMS_MATCH"] = (
        model_identity["selected_params"]
        == EXPECTED_MODEL_PARAMS
    )

    gates["G10_RISK_SCORE_INTERFACE_LOCKED"] = (
        contract["risk_score_contract"]["definition"]
        == "predict_proba(X)[:, positive_class_index]"
        and contract["risk_score_contract"][
            "positive_class"
        ] == 1
        and contract["risk_score_contract"][
            "positive_class_index"
        ] == 1
    )

    threshold = contract["threshold_contract"]
    gates["G11_THRESHOLD_EXACT"] = (
        threshold["threshold"] == 0.50
        and threshold["comparator"] == ">"
        and threshold["greater_or_equal_allowed"] is False
    )

    gates["G12_THRESHOLD_BOUNDARY_SEMANTICS"] = (
        threshold["boundary_examples"]["0.49"] is False
        and threshold["boundary_examples"]["0.50"] is False
        and threshold["boundary_examples"]["0.51"] is True
    )

    gates["G13_NO_RUNTIME_FIT"] = (
        contract["pipeline_boundary"][
            "inference_may_fit_preprocessing"
        ] is False
        and contract["pipeline_boundary"][
            "inference_may_fit_model"
        ] is False
    )

    gates["G14_PROMOTION_DEFERRED_TO_M9_6"] = (
        contract["model_contract"][
            "frozen_artifact_identity"
        ]["promotion_timing"] == "M9.6"
        and contract["preprocessing_contract"][
            "frozen_state_identity"
        ]["promotion_timing"] == "M9.6"
    )

    gates["G15_NO_SCIENTIFIC_CHANGE"] = all(
        value is False
        for value in contract["guardrails"].values()
    )

    return gates


def main() -> None:
    root = detect_root()

    source_registry_path = root / SOURCE_REGISTRY_REL
    history_feature_path = root / HISTORY_FEATURE_CONTRACT_REL
    model_path = root / SOURCE_MODEL_REL
    preprocessing_path = root / SOURCE_PREPROCESSING_REL

    required_paths = [
        source_registry_path,
        history_feature_path,
        model_path,
        preprocessing_path,
    ]

    missing = [
        str(path.relative_to(root))
        for path in required_paths
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Thiếu dependency/artifact:\n"
            + "\n".join(missing)
        )

    print("=" * 88)
    print("M9.4.4 — PREPROCESSING + MODEL + THRESHOLD CONTRACT")
    print("=" * 88)
    print("Project root:", root)
    print(
        "Mode: READ-ONLY IDENTITY + CONTRACT — "
        "NO FIT / NO TRANSFORM / NO PREDICTION"
    )

    source_registry = read_json(source_registry_path)
    history_feature_contract = read_json(
        history_feature_path
    )
    preprocessing_state = read_json(preprocessing_path)

    model_sha = sha256_file(model_path)
    preprocessing_sha = sha256_file(
        preprocessing_path
    )

    print("\n[1] Frozen artifact fingerprints")
    print(" - model:", model_sha)
    print(" - preprocessing:", preprocessing_sha)

    model_identity = load_model_identity(model_path)

    print("\n[2] Estimator identity — load only")
    print(
        " - type:",
        model_identity["estimator_type"],
    )
    print(
        " - classes:",
        model_identity["classes"],
    )
    print(
        " - positive class index:",
        model_identity["positive_class_index"],
    )
    print(
        " - sklearn runtime:",
        model_identity["sklearn_version_runtime"],
    )

    print("\n[3] Frozen preprocessing identity")
    print(
        " - feature_count:",
        preprocessing_state.get("feature_count"),
    )
    print(
        " - numeric_columns:",
        preprocessing_state.get("numeric_columns"),
    )
    print(
        " - feature_names count:",
        len(preprocessing_state.get("feature_names", [])),
    )
    print(
        " - numeric_mean count:",
        len(preprocessing_state.get("numeric_mean", [])),
    )
    print(
        " - numeric_scale count:",
        len(preprocessing_state.get("numeric_scale", [])),
    )

    contract = build_contract(
        model_identity=model_identity,
        preprocessing_state=preprocessing_state,
        model_sha=model_sha,
        preprocessing_sha=preprocessing_sha,
    )

    print("\n[4] Frozen inference decision semantics")
    print(" - risk_score = predict_proba class-1 score")
    print(" - threshold = 0.50")
    print(" - comparator = >")
    print(" - 0.49 → negative")
    print(" - 0.50 → negative")
    print(" - 0.51 → positive")

    gates = validate(
        source_registry=source_registry,
        history_feature_contract=history_feature_contract,
        preprocessing_state=preprocessing_state,
        model_identity=model_identity,
        model_sha=model_sha,
        preprocessing_sha=preprocessing_sha,
        contract=contract,
    )

    print("\n[5] M9.4.4 gate")
    for gate, passed in gates.items():
        print(
            f" - {gate}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(gates.values()):
        raise RuntimeError(
            "M9.4.4 gate FAIL. Không ghi contract."
        )

    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise RuntimeError(
            "STOP: preprocessing/model/threshold contract đã tồn tại; "
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

    print("\n" + "=" * 88)
    print(
        "M9.4.4 PREPROCESSING + MODEL + THRESHOLD CONTRACT: PASS"
    )
    print("Frozen preprocessing identity verified.")
    print("Frozen estimator identity verified by load only.")
    print("Strict threshold semantics locked.")
    print("No preprocessing transform.")
    print("No model fit.")
    print("No prediction / predict_proba call.")
    print("No artifact copy or overwrite.")
    print(
        "NEXT: M9.4.5 — Inference Output + Artifact + Error Contract"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
