from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_VERSION = "M9.4.5-inference-output-artifact-error-contract-v1"

SOURCE_REGISTRY_REL = (
    "final_pipeline/docs/contracts/contract_source_registry.json"
)
INPUT_CONTRACT_REL = (
    "final_pipeline/docs/contracts/transaction_input_data_contract.json"
)
HISTORY_FEATURE_REL = (
    "final_pipeline/docs/contracts/history_feature_contract.json"
)
PREPROCESSING_MODEL_REL = (
    "final_pipeline/docs/contracts/"
    "preprocessing_model_threshold_contract.json"
)
OUTPUT_REL = (
    "final_pipeline/docs/contracts/"
    "inference_output_artifact_error_contract.json"
)

SOURCE_MODEL_REL = (
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_selected_rf_estimator.joblib"
)
SOURCE_PREPROCESSING_REL = (
    "research/data/processed/m8_02_final_test_artifact_audit/"
    "m8_02_w_short_preprocessing_state.json"
)

OFFICIAL_MODEL_REL = (
    "final_pipeline/artifacts/official/model/"
    "m8_02_selected_rf_estimator.joblib"
)
OFFICIAL_PREPROCESSING_REL = (
    "final_pipeline/artifacts/official/preprocessing/"
    "m8_02_w_short_preprocessing_state.json"
)
OFFICIAL_MANIFEST_REL = (
    "final_pipeline/artifacts/official/manifest/"
    "official_artifact_manifest.json"
)

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)
EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)

MODEL_ID = "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
THRESHOLD = 0.50


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(
    observed_model_sha: str,
    observed_preprocessing_sha: str,
) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "scope": (
            "Stable inference result schema, official-artifact loading "
            "policy, validation/error taxonomy and application boundary."
        ),
        "inference_output_contract": {
            "required_fields": {
                "risk_score": {
                    "type": "float",
                    "domain": "[0.0, 1.0]",
                    "meaning": "positive-class / class-1 model score",
                    "finite_required": True,
                },
                "threshold": {
                    "type": "float",
                    "value": THRESHOLD,
                },
                "screening_prediction": {
                    "type": "integer",
                    "allowed_values": [0, 1],
                    "rule": "1 iff risk_score > 0.50; else 0",
                    "semantics": {
                        "0": "negative screening result / not flagged",
                        "1": "positive screening result / flagged",
                    },
                    "not_equal_to_ground_truth_fraud_label": True,
                },
                "model_id": {
                    "type": "string",
                    "value": MODEL_ID,
                },
            },
            "optional_fields": {
                "cold_start": {
                    "type": "boolean",
                    "meaning": (
                        "True iff no strict-prior User+Card history "
                        "was available for this prediction point."
                    ),
                },
                "warnings": {
                    "type": "list[string]",
                    "default": [],
                    "meaning": (
                        "Non-fatal inference notices only; warnings "
                        "must never hide validation/artifact failures."
                    ),
                },
            },
            "risk_score_language": {
                "allowed": [
                    "risk score",
                    "positive-class score",
                    "model score",
                ],
                "prohibited": [
                    "calibrated confidence",
                    "real-world fraud probability",
                    "x% chắc chắn là fraud",
                ],
                "calibrated": False,
            },
            "determinism": {
                "same_input_and_same_state_same_result_required": True,
            },
        },
        "artifact_contract": {
            "current_pre_promotion_sources": {
                "model": SOURCE_MODEL_REL,
                "preprocessing": SOURCE_PREPROCESSING_REL,
                "purpose": (
                    "Identity/source evidence before M9.6 promotion only."
                ),
            },
            "official_runtime_paths_after_m9_6": {
                "model": OFFICIAL_MODEL_REL,
                "preprocessing": OFFICIAL_PREPROCESSING_REL,
                "manifest": OFFICIAL_MANIFEST_REL,
            },
            "official_runtime_policy": {
                "application_may_load_from_research": False,
                "final_pipeline_runtime_may_load_from_research_after_m9_6":
                    False,
                "silent_fallback_to_research": False,
                "official_artifacts_immutable_in_normal_runtime": True,
                "promotion_requires_exact_identity": True,
                "promotion_method": "byte-preserving copy",
                "promotion_step": "M9.6",
            },
            "loader_validation_order": [
                "resolve official manifest/path",
                "require artifact files to exist",
                "verify model SHA-256",
                "verify preprocessing SHA-256",
                "load preprocessing state",
                "verify preprocessing feature_count == 47",
                "verify semantic/numeric schema identity",
                "load estimator",
                "verify estimator type/config/classes",
                "verify positive class index == 1",
                "only then authorize inference",
            ],
            "expected_identity": {
                "model_id": MODEL_ID,
                "model_sha256": EXPECTED_MODEL_SHA256,
                "preprocessing_sha256":
                    EXPECTED_PREPROCESSING_SHA256,
                "feature_count": 47,
                "classes": [0, 1],
                "positive_class": 1,
                "positive_class_index": 1,
            },
            "observed_source_identity_at_contract_time": {
                "model_sha256": observed_model_sha,
                "preprocessing_sha256":
                    observed_preprocessing_sha,
            },
            "overwrite_policy": {
                "silent_overwrite_allowed": False,
                "training_may_write_official": False,
                "rebuild_target": "final_pipeline/outputs/rebuilds/<run_id>/",
                "official_change_requires": [
                    "explicit promotion action",
                    "identity validation",
                    "overwrite guard",
                    "documented reason",
                ],
            },
        },
        "error_validation_contract": {
            "principle": (
                "Fail fast on invalid input, causal-state violation, "
                "artifact mismatch or inference-contract violation."
            ),
            "error_types": {
                "InputValidationError": {
                    "fatal": True,
                    "examples": [
                        "required scoring field missing",
                        "Timestamp parse failure",
                        "Amount parse failure",
                        "invalid Time/date component",
                        "location combination resolves to OTHER_INCONSISTENT",
                        "non-integer Zip when Zip is present",
                    ],
                },
                "HistoryValidationError": {
                    "fatal": True,
                    "examples": [
                        "future history supplied",
                        "same-timestamp peer used as history",
                        "target label appears in history state",
                        "history ordering violates strict causality",
                    ],
                },
                "PreprocessingContractError": {
                    "fatal": True,
                    "examples": [
                        "semantic feature count/order mismatch",
                        "preprocessing output width != 47",
                        "unexpected output dtype/representation",
                        "frozen preprocessing state incompatible",
                    ],
                },
                "ArtifactNotFoundError": {
                    "fatal": True,
                    "examples": [
                        "official model missing",
                        "official preprocessing state missing",
                        "official manifest missing",
                    ],
                },
                "ArtifactFingerprintError": {
                    "fatal": True,
                    "examples": [
                        "model SHA-256 mismatch",
                        "preprocessing SHA-256 mismatch",
                    ],
                },
                "ArtifactCompatibilityError": {
                    "fatal": True,
                    "examples": [
                        "estimator is not expected RandomForestClassifier",
                        "classes are not [0, 1]",
                        "positive class index is not 1",
                        "model config identity mismatch",
                    ],
                },
                "InferenceContractError": {
                    "fatal": True,
                    "examples": [
                        "risk_score is non-finite",
                        "risk_score outside [0, 1]",
                        "screening_prediction disagrees with strict > 0.50 rule",
                    ],
                },
            },
            "non_errors": {
                "cold_start": (
                    "Valid structural state; not an exception."
                ),
                "known_structural_location_missingness": (
                    "Valid when it resolves deterministically to a locked "
                    "location_state."
                ),
                "unknown_frozen_categorical_value": (
                    "Handled by frozen __UNKNOWN__ preprocessing policy; "
                    "not automatically fatal."
                ),
            },
            "no_silent_repair": [
                "do not fabricate required fields",
                "do not coerce OTHER_INCONSISTENT to a valid location state",
                "do not change threshold on error",
                "do not fallback to another model",
                "do not refit preprocessing",
                "do not retrain model",
            ],
        },
        "application_boundary_contract": {
            "application_calls": (
                "one final_pipeline inference/service interface"
            ),
            "application_must_not_know": [
                "47-column encoded feature order",
                "numeric scaler internals",
                "categorical vocabularies",
                "positive probability column index implementation",
                "joblib loading details",
                "artifact SHA verification implementation",
                "history-state implementation details",
            ],
            "application_must_not": [
                "fit preprocessing",
                "fit model",
                "import research notebooks",
                "read research artifacts directly",
                "apply its own threshold logic",
            ],
        },
        "side_effect_classification": {
            "artifact_load_and_validation": "A_READ_ONLY",
            "parse_feature_transform_inference":
                "B_DETERMINISTIC_TRANSFORM_OR_INFERENCE",
            "fit_or_training": "C_LEARNED_STATE_CREATION",
            "official_promotion_or_overwrite":
                "D_PROTECTED_PROMOTION_OR_OVERWRITE",
        },
        "source_trace": {
            "m9_architecture":
                "CANON — Kiến trúc ba vùng và kế hoạch thực thi M9–M14",
            "m8_score_language":
                "CANON-M8.6 — Final Evaluation Registry",
            "m8_output_guard":
                "CANON-M8.3 — Frozen Final Inference Run",
            "upstream_contracts": [
                INPUT_CONTRACT_REL,
                HISTORY_FEATURE_REL,
                PREPROCESSING_MODEL_REL,
                SOURCE_REGISTRY_REL,
            ],
        },
        "guardrails": {
            "model_fit_performed": False,
            "prediction_performed": False,
            "predict_proba_performed": False,
            "preprocessing_transform_performed": False,
            "artifact_copy_performed": False,
            "artifact_overwrite_performed": False,
            "official_manifest_created": False,
            "threshold_changed": False,
        },
        "next_step": (
            "M9.4.6 — Contract Registry + M9.4 Final Gate"
        ),
    }


def validate(
    source_registry: dict,
    input_contract: dict,
    history_feature_contract: dict,
    preprocessing_model_contract: dict,
    contract: dict,
) -> dict:
    output = contract["inference_output_contract"]
    artifact = contract["artifact_contract"]
    errors = contract["error_validation_contract"]
    app = contract["application_boundary_contract"]

    gates = {}

    gates["G01_SOURCE_REGISTRY_PASS"] = (
        source_registry.get("gate_status") == "PASS"
    )

    gates["G02_INPUT_CONTRACT_PASS"] = (
        input_contract.get("gate_status") == "PASS"
    )

    gates["G03_HISTORY_FEATURE_CONTRACT_PASS"] = (
        history_feature_contract.get("gate_status") == "PASS"
    )

    gates["G04_PREPROCESSING_MODEL_CONTRACT_PASS"] = (
        preprocessing_model_contract.get("gate_status") == "PASS"
    )

    required_fields = output["required_fields"]
    gates["G05_MINIMUM_OUTPUT_FIELDS_LOCKED"] = (
        list(required_fields.keys())
        == [
            "risk_score",
            "threshold",
            "screening_prediction",
            "model_id",
        ]
    )

    gates["G06_SCREENING_LABEL_NOT_GROUND_TRUTH"] = (
        required_fields["screening_prediction"][
            "not_equal_to_ground_truth_fraud_label"
        ] is True
    )

    gates["G07_STRICT_THRESHOLD_OUTPUT_RULE"] = (
        required_fields["threshold"]["value"] == 0.50
        and required_fields["screening_prediction"]["rule"]
        == "1 iff risk_score > 0.50; else 0"
    )

    gates["G08_RISK_SCORE_LANGUAGE_SAFE"] = (
        output["risk_score_language"]["calibrated"] is False
        and "real-world fraud probability"
        in output["risk_score_language"]["prohibited"]
    )

    gates["G09_OFFICIAL_RUNTIME_NO_RESEARCH_FALLBACK"] = (
        artifact["official_runtime_policy"][
            "application_may_load_from_research"
        ] is False
        and artifact["official_runtime_policy"][
            "final_pipeline_runtime_may_load_from_research_after_m9_6"
        ] is False
        and artifact["official_runtime_policy"][
            "silent_fallback_to_research"
        ] is False
    )

    gates["G10_PROMOTION_POLICY_PROTECTED"] = (
        artifact["official_runtime_policy"][
            "promotion_requires_exact_identity"
        ] is True
        and artifact["official_runtime_policy"][
            "promotion_method"
        ] == "byte-preserving copy"
        and artifact["official_runtime_policy"][
            "promotion_step"
        ] == "M9.6"
    )

    gates["G11_FINGERPRINTS_LOCKED"] = (
        artifact["expected_identity"]["model_sha256"]
        == EXPECTED_MODEL_SHA256
        and artifact["expected_identity"][
            "preprocessing_sha256"
        ] == EXPECTED_PREPROCESSING_SHA256
    )

    gates["G12_ERROR_TAXONOMY_COMPLETE"] = (
        set(errors["error_types"].keys())
        == {
            "InputValidationError",
            "HistoryValidationError",
            "PreprocessingContractError",
            "ArtifactNotFoundError",
            "ArtifactFingerprintError",
            "ArtifactCompatibilityError",
            "InferenceContractError",
        }
    )

    gates["G13_COLD_START_NOT_ERROR"] = (
        "cold_start" in errors["non_errors"]
    )

    gates["G14_NO_SILENT_REPAIR"] = (
        "do not fallback to another model"
        in errors["no_silent_repair"]
        and "do not refit preprocessing"
        in errors["no_silent_repair"]
        and "do not retrain model"
        in errors["no_silent_repair"]
    )

    gates["G15_APPLICATION_BOUNDARY_LOCKED"] = (
        "fit model" in app["application_must_not"]
        and "import research notebooks"
        in app["application_must_not"]
        and "apply its own threshold logic"
        in app["application_must_not"]
    )

    gates["G16_NO_SCIENTIFIC_CHANGE"] = all(
        value is False
        for value in contract["guardrails"].values()
    )

    return gates


def main() -> None:
    root = detect_root()

    dependency_paths = {
        "source_registry": root / SOURCE_REGISTRY_REL,
        "input_contract": root / INPUT_CONTRACT_REL,
        "history_feature_contract": root / HISTORY_FEATURE_REL,
        "preprocessing_model_contract":
            root / PREPROCESSING_MODEL_REL,
        "source_model": root / SOURCE_MODEL_REL,
        "source_preprocessing":
            root / SOURCE_PREPROCESSING_REL,
    }

    missing = [
        str(path.relative_to(root))
        for path in dependency_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Thiếu dependency/artifact:\n"
            + "\n".join(missing)
        )

    print("=" * 90)
    print("M9.4.5 — INFERENCE OUTPUT + ARTIFACT + ERROR CONTRACT")
    print("=" * 90)
    print("Project root:", root)
    print(
        "Mode: GOVERNANCE + READ-ONLY IDENTITY — "
        "NO TRANSFORM / NO FIT / NO PREDICTION / NO PROMOTION"
    )

    source_registry = read_json(
        dependency_paths["source_registry"]
    )
    input_contract = read_json(
        dependency_paths["input_contract"]
    )
    history_feature_contract = read_json(
        dependency_paths["history_feature_contract"]
    )
    preprocessing_model_contract = read_json(
        dependency_paths["preprocessing_model_contract"]
    )

    observed_model_sha = sha256_file(
        dependency_paths["source_model"]
    )
    observed_preprocessing_sha = sha256_file(
        dependency_paths["source_preprocessing"]
    )

    print("\n[1] Source identity re-check")
    print(" - model:", observed_model_sha)
    print(
        " - preprocessing:",
        observed_preprocessing_sha,
    )

    if observed_model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "STOP: model fingerprint mismatch."
        )
    if (
        observed_preprocessing_sha
        != EXPECTED_PREPROCESSING_SHA256
    ):
        raise RuntimeError(
            "STOP: preprocessing fingerprint mismatch."
        )

    contract = build_contract(
        observed_model_sha=observed_model_sha,
        observed_preprocessing_sha=
            observed_preprocessing_sha,
    )

    print("\n[2] Required inference output")
    for field, spec in (
        contract["inference_output_contract"][
            "required_fields"
        ].items()
    ):
        print(f" - {field}: {spec['type']}")

    print("\n[3] Artifact runtime policy")
    print(" - official runtime path: final_pipeline/artifacts/official/")
    print(" - application → research artifact: FORBIDDEN")
    print(" - silent research fallback: FORBIDDEN")
    print(" - promotion: M9.6 ONLY")
    print(" - normal runtime official overwrite: FORBIDDEN")

    print("\n[4] Fatal error classes")
    for name in (
        contract["error_validation_contract"][
            "error_types"
        ]
    ):
        print(" -", name)

    print("\n[5] Non-error states")
    for name in (
        contract["error_validation_contract"][
            "non_errors"
        ]
    ):
        print(" -", name)

    gates = validate(
        source_registry=source_registry,
        input_contract=input_contract,
        history_feature_contract=
            history_feature_contract,
        preprocessing_model_contract=
            preprocessing_model_contract,
        contract=contract,
    )

    print("\n[6] M9.4.5 gate")
    for gate, passed in gates.items():
        print(
            f" - {gate}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(gates.values()):
        raise RuntimeError(
            "M9.4.5 gate FAIL. Không ghi contract."
        )

    output_path = root / OUTPUT_REL
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        raise RuntimeError(
            "STOP: inference/artifact/error contract đã tồn tại; "
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

    print("\n[7] Contract written")
    print(" -", output_path.relative_to(root))

    print("\n" + "=" * 90)
    print(
        "M9.4.5 INFERENCE OUTPUT + ARTIFACT + ERROR CONTRACT: PASS"
    )
    print("Stable inference output schema locked.")
    print("Official artifact loader policy locked.")
    print("Fail-fast error taxonomy locked.")
    print("No preprocessing transform.")
    print("No model fit.")
    print("No prediction / predict_proba call.")
    print("No artifact promotion/copy/overwrite.")
    print(
        "NEXT: M9.4.6 — Contract Registry + M9.4 Final Gate"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
