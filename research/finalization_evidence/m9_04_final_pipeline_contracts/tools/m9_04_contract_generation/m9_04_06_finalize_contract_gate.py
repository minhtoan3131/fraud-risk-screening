from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_VERSION = "M9.4-final-contract-registry-v1"

CONTRACT_DIR_REL = "final_pipeline/docs/contracts"

CONTRACT_FILES = {
    "source_registry": "contract_source_registry.json",
    "transaction_input_data": "transaction_input_data_contract.json",
    "history_feature": "history_feature_contract.json",
    "preprocessing_model_threshold":
        "preprocessing_model_threshold_contract.json",
    "inference_output_artifact_error":
        "inference_output_artifact_error_contract.json",
}

FINAL_REGISTRY_NAME = "m9_04_final_contract_registry.json"
CANON_NAME = "CANON-M9.4 — Final Pipeline Contract.md"

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)
EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)

EXPECTED_SEMANTIC_FEATURE_ORDER = [
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

EXPECTED_OUTPUT_FIELDS = {
    "risk_score",
    "threshold",
    "screening_prediction",
    "model_id",
}

ROOT_TOOL_SCRIPTS = [
    "m9_04_01_contract_source_audit.py",
    "m9_04_02_transaction_input_data_contract.py",
    "m9_04_03_history_feature_contract.py",
    "m9_04_04_preprocessing_model_threshold_contract.py",
    "m9_04_05_inference_output_artifact_error_contract.py",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def collect_contracts(root: Path) -> dict:
    contract_dir = root / CONTRACT_DIR_REL
    contracts = {}

    missing = []
    for key, filename in CONTRACT_FILES.items():
        path = contract_dir / filename
        if not path.is_file():
            missing.append(str(path.relative_to(root)))
            continue
        contracts[key] = read_json(path)

    if missing:
        raise RuntimeError(
            "Thiếu contract M9.4:\n"
            + "\n".join(missing)
        )

    return contracts


def contract_hashes(root: Path) -> dict:
    contract_dir = root / CONTRACT_DIR_REL
    return {
        key: sha256_file(contract_dir / filename)
        for key, filename in CONTRACT_FILES.items()
    }


def cross_validate(contracts: dict) -> dict:
    source = contracts["source_registry"]
    input_data = contracts["transaction_input_data"]
    history = contracts["history_feature"]
    pmt = contracts["preprocessing_model_threshold"]
    output_artifact = contracts["inference_output_artifact_error"]

    gates = {}

    gates["G01_ALL_SUBCONTRACTS_PASS"] = all(
        contract.get("gate_status") == "PASS"
        for contract in contracts.values()
    )

    gates["G02_EXACT_10_FEATURE_ORDER_CONSISTENT"] = (
        history["feature_contract"]["semantic_feature_order"]
        == EXPECTED_SEMANTIC_FEATURE_ORDER
        and pmt["preprocessing_contract"][
            "input_semantic_feature_order"
        ] == EXPECTED_SEMANTIC_FEATURE_ORDER
    )

    gates["G03_FEATURE_COUNTS_10_TO_47_CONSISTENT"] = (
        history["feature_contract"][
            "feature_count_pre_encoding"
        ] == 10
        and pmt["preprocessing_contract"][
            "input_feature_count"
        ] == 10
        and pmt["preprocessing_contract"]["output"][
            "feature_count"
        ] == 47
    )

    gates["G04_STRICT_CAUSAL_RULE_CONSISTENT"] = (
        history["history_contract"]["strict_causal_rule"]
        == "Timestamp(history) < Timestamp(current)"
    )

    gates["G05_COLD_START_CONSISTENT"] = (
        history["history_contract"]["cold_start"]["values"]
        == {
            "has_prior_card_history": False,
            "time_since_previous_transaction_min": None,
            "transactions_last_1h": 0,
            "amount_minus_previous_mean": None,
            "is_new_merchant": True,
        }
    )

    gates["G06_MODEL_ID_CONSISTENT"] = (
        pmt["model_contract"]["config_id"]
        == "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
        and output_artifact["inference_output_contract"][
            "required_fields"
        ]["model_id"]["value"]
        == "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
    )

    gates["G07_MODEL_FINGERPRINT_CONSISTENT"] = (
        pmt["model_contract"]["frozen_artifact_identity"][
            "sha256"
        ] == EXPECTED_MODEL_SHA256
        and output_artifact["artifact_contract"][
            "expected_identity"
        ]["model_sha256"] == EXPECTED_MODEL_SHA256
    )

    gates["G08_PREPROCESSING_FINGERPRINT_CONSISTENT"] = (
        pmt["preprocessing_contract"]["frozen_state_identity"][
            "sha256"
        ] == EXPECTED_PREPROCESSING_SHA256
        and output_artifact["artifact_contract"][
            "expected_identity"
        ]["preprocessing_sha256"]
        == EXPECTED_PREPROCESSING_SHA256
    )

    gates["G09_POSITIVE_CLASS_CONSISTENT"] = (
        pmt["model_contract"]["classes"] == [0, 1]
        and pmt["model_contract"]["positive_class"] == 1
        and pmt["model_contract"]["positive_class_index"] == 1
        and output_artifact["artifact_contract"][
            "expected_identity"
        ]["positive_class_index"] == 1
    )

    gates["G10_THRESHOLD_STRICT_GT_CONSISTENT"] = (
        pmt["threshold_contract"]["threshold"] == 0.50
        and pmt["threshold_contract"]["comparator"] == ">"
        and output_artifact["inference_output_contract"][
            "required_fields"
        ]["screening_prediction"]["rule"]
        == "1 iff risk_score > 0.50; else 0"
    )

    # FIX v2:
    # JSON object order is not part of the contract semantics.
    # M9.4.5 serializes with sort_keys=True, so checking list(keys)
    # would create a false failure even when all required fields exist.
    required_fields = output_artifact[
        "inference_output_contract"
    ]["required_fields"]

    gates["G11_OUTPUT_FIELDS_EXACT_MINIMUM"] = (
        set(required_fields.keys())
        == EXPECTED_OUTPUT_FIELDS
        and required_fields["risk_score"]["type"] == "float"
        and required_fields["threshold"]["value"] == 0.50
        and required_fields["screening_prediction"][
            "allowed_values"
        ] == [0, 1]
        and required_fields["model_id"]["value"]
        == "RF-REF-100-GINI-SQRT-UNPRUNED-CW"
    )

    gates["G12_TARGET_EXCLUDED_FROM_INFERENCE"] = (
        input_data["layers"]["scoring_payload"][
            "target_allowed"
        ] is False
        and history["history_contract"][
            "target_history_allowed"
        ] is False
    )

    gates["G13_ERRORS_FIELD_EXCLUDED"] = (
        input_data["layers"]["scoring_payload"][
            "errors_field_allowed"
        ] is False
    )

    gates["G14_RAW_IDENTIFIERS_NOT_DIRECT_FEATURES"] = (
        history["history_contract"]["raw_identifiers"][
            "direct_classifier_use"
        ] is False
    )

    gates["G15_RUNTIME_RESEARCH_DEPENDENCY_FORBIDDEN"] = (
        output_artifact["artifact_contract"][
            "official_runtime_policy"
        ]["application_may_load_from_research"] is False
        and output_artifact["artifact_contract"][
            "official_runtime_policy"
        ][
            "final_pipeline_runtime_may_load_from_research_after_m9_6"
        ] is False
    )

    gates["G16_PROMOTION_DEFERRED_AND_PROTECTED"] = (
        pmt["model_contract"]["frozen_artifact_identity"][
            "promotion_timing"
        ] == "M9.6"
        and output_artifact["artifact_contract"][
            "official_runtime_policy"
        ]["promotion_step"] == "M9.6"
        and output_artifact["artifact_contract"][
            "overwrite_policy"
        ]["silent_overwrite_allowed"] is False
    )

    gates["G17_RISK_SCORE_LANGUAGE_CONSISTENT"] = (
        pmt["risk_score_contract"]["calibration_performed"]
        is False
        and output_artifact["inference_output_contract"][
            "risk_score_language"
        ]["calibrated"] is False
        and "real-world fraud probability"
        in output_artifact["inference_output_contract"][
            "risk_score_language"
        ]["prohibited"]
    )

    gates["G18_APPLICATION_BOUNDARY_LOCKED"] = (
        "import research notebooks"
        in output_artifact["application_boundary_contract"][
            "application_must_not"
        ]
        and "apply its own threshold logic"
        in output_artifact["application_boundary_contract"][
            "application_must_not"
        ]
    )

    gates["G19_SOURCE_REGISTRY_HAS_10_TARGET_CONTRACTS"] = (
        len(source["planned_m9_4_contracts"]) == 10
    )

    all_guardrails = []
    for contract in contracts.values():
        guardrails = contract.get("guardrails", {})
        all_guardrails.extend(guardrails.values())

    gates["G20_NO_SCIENTIFIC_CHANGE_ACROSS_M9_4"] = (
        all(value is False for value in all_guardrails)
    )

    return gates


def build_registry(
    root: Path,
    contracts: dict,
    hashes: dict,
    gates: dict,
) -> dict:
    return {
        "registry_version": REGISTRY_VERSION,
        "completed_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "decision": "PASS",
        "m9_5_authorization": "AUTHORIZED",
        "contract_directory": CONTRACT_DIR_REL,
        "contracts": {
            key: {
                "file": CONTRACT_FILES[key],
                "sha256": hashes[key],
                "gate_status": contracts[key].get("gate_status"),
                "contract_version":
                    contracts[key].get(
                        "contract_version",
                        contracts[key].get("registry_version"),
                    ),
            }
            for key in CONTRACT_FILES
        },
        "cross_contract_gates": gates,
        "frozen_invariants": {
            "semantic_feature_count": 10,
            "semantic_feature_order":
                EXPECTED_SEMANTIC_FEATURE_ORDER,
            "encoded_feature_count": 47,
            "matrix_type": "CSR sparse matrix",
            "matrix_dtype": "float32",
            "strict_causal_rule":
                "Timestamp(history) < Timestamp(current)",
            "same_timestamp_peer_history_allowed": False,
            "model_id":
                "RF-REF-100-GINI-SQRT-UNPRUNED-CW",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "preprocessing_sha256":
                EXPECTED_PREPROCESSING_SHA256,
            "classes": [0, 1],
            "positive_class": 1,
            "positive_class_index": 1,
            "risk_score_interface":
                "predict_proba positive-class score",
            "threshold": 0.50,
            "threshold_comparator": ">",
            "minimum_output_fields":
                sorted(EXPECTED_OUTPUT_FIELDS),
        },
        "architecture_invariants": {
            "research_role":
                "historical evidence / provenance",
            "final_pipeline_role":
                "canonical implementation",
            "application_role":
                "user-facing consumer of final_pipeline",
            "application_runtime_dependency_on_research": "NONE",
            "final_pipeline_runtime_dependency_on_research_after_m9_6":
                "NONE",
            "core_logic_single_source_of_truth_required": True,
        },
        "m9_5_scope": {
            "name": "Canonical Implementation Extraction",
            "implementation_order": [
                "data + validation",
                "transaction feature representation",
                "behavioral/history features",
                "frozen preprocessing",
                "artifact loader/config",
                "inference primitives",
            ],
            "rules": [
                "extract/refactor from reviewed evidence; do not redesign semantics",
                "add unit/regression tests while extracting",
                "compare against locked contracts",
                "do not train or promote official artifacts in M9.5",
                "do not make application depend on research",
            ],
        },
        "attestations": {
            "model_fit_performed_in_m9_4": False,
            "prediction_performed_in_m9_4": False,
            "preprocessing_transform_performed_in_m9_4": False,
            "artifact_promotion_performed_in_m9_4": False,
            "feature_semantics_changed_in_m9_4": False,
            "threshold_changed_in_m9_4": False,
        },
    }


def build_canon_markdown(registry: dict) -> str:
    feature_lines = "\n".join(
        f"{i}. `{name}`"
        for i, name in enumerate(
            registry["frozen_invariants"]["semantic_feature_order"],
            start=1,
        )
    )

    contract_lines = "\n".join(
        f"- `{record['file']}` — SHA-256 `{record['sha256']}`"
        for record in registry["contracts"].values()
    )

    gate_lines = "\n".join(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in registry["cross_contract_gates"].items()
    )

    return f"""# CANON-M9.4 — Final Pipeline Contract

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M9.4 — Final Pipeline Contracts  
**Decision:** `PASS`  
**M9.5 authorization:** `AUTHORIZED`

---

## 1. Mục tiêu

M9.4 khóa interface và semantics của final pipeline trước khi trích xuất implementation Python ở M9.5.

M9.4 không train model, không refit preprocessing, không chạy inference và không promote official artifact.

---

## 2. Contract artifacts

{contract_lines}

Registry tổng:

- `{FINAL_REGISTRY_NAME}`

---

## 3. Input / data boundary

Frozen scoring payload sử dụng 12 raw fields:

- `User`
- `Card`
- `Year`
- `Month`
- `Day`
- `Time`
- `Amount`
- `Use Chip`
- `Merchant Name`
- `Merchant City`
- `Merchant State`
- `Zip`

`MCC` không thuộc frozen baseline.

`Errors?` và `Is Fraud?` không được sử dụng làm inference input.

---

## 4. Strict-causal history

Rule bắt buộc:

`Timestamp(history) < Timestamp(current)`

Không cho phép:

- current transaction làm history;
- same-timestamp peer làm history;
- future transaction;
- future-inclusive aggregate;
- target-label history.

State chỉ update sau khi toàn bộ transaction ở cùng prediction timestamp đã được tính.

Cold-start:

- `has_prior_card_history = False`
- `time_since_previous_transaction_min = NA`
- `transactions_last_1h = 0`
- `amount_minus_previous_mean = NA`
- `is_new_merchant = True`

---

## 5. Exact semantic feature interface

{feature_lines}

Pre-encoding feature count: `10`.

---

## 6. Frozen preprocessing

Numeric:

- `amount_numeric`
- `time_since_previous_transaction_min`
- `transactions_last_1h`
- `amount_minus_previous_mean`

Boolean:

- `is_new_merchant`
- `has_prior_card_history`

Categorical:

- `transaction_mode`
- `location_state`
- `hour_of_day`
- `day_of_week`

Frozen output:

- width: `47`
- matrix: `CSR sparse matrix`
- dtype: `float32`

Preprocessing state SHA-256:

`{EXPECTED_PREPROCESSING_SHA256}`

---

## 7. Frozen model

Model ID:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Model:

`RandomForestClassifier`

Classes:

`[0, 1]`

Positive class:

`1`

Positive-class index:

`1`

Frozen estimator SHA-256:

`{EXPECTED_MODEL_SHA256}`

---

## 8. Risk score và threshold

Risk score:

`predict_proba` positive-class / class-1 score.

Không được gọi là calibrated confidence hoặc real-world fraud probability.

Frozen threshold:

`0.50`

Comparator:

`risk_score > 0.50`

Do đó:

- `0.49` → negative screening
- `0.50` → negative screening
- `0.51` → positive screening

---

## 9. Minimum inference output

- `risk_score`
- `threshold`
- `screening_prediction`
- `model_id`

Optional:

- `cold_start`
- `warnings`

`screening_prediction = 1` có nghĩa giao dịch bị flag bởi screening rule, không phải ground-truth fraud label.

---

## 10. Official artifact policy

Trước M9.6, frozen artifact trong `research/` chỉ là source identity/provenance.

Ở M9.6:

M8 evaluated artifact  
→ byte-preserving copy  
→ SHA-256 verification  
→ `final_pipeline/artifacts/official/`

Sau M9.6:

- application không load artifact từ `research/`;
- final pipeline runtime không fallback sang `research/`;
- training/rebuild không ghi đè official artifact.

---

## 11. Error policy

Fatal categories:

- `InputValidationError`
- `HistoryValidationError`
- `PreprocessingContractError`
- `ArtifactNotFoundError`
- `ArtifactFingerprintError`
- `ArtifactCompatibilityError`
- `InferenceContractError`

Không silent repair model/preprocessing/threshold.

Cold-start là trạng thái hợp lệ, không phải exception.

---

## 12. Architecture boundary

`research/` = historical evidence / provenance.

`final_pipeline/` = canonical ML implementation.

`application/` = user-facing consumer.

Sau M9 Gate:

- application runtime dependency on research = `NONE`;
- final_pipeline runtime dependency on research = `NONE`;
- core ML logic có một source of truth duy nhất.

---

## 13. Cross-contract gate

{gate_lines}

Overall:

`PASS`

---

## 14. Handoff sang M9.5

M9.5 được phép bắt đầu:

`Canonical Implementation Extraction`

Thứ tự ưu tiên:

1. data + validation;
2. transaction features;
3. behavioral/history features;
4. frozen preprocessing;
5. artifact/config loader;
6. inference primitives;
7. unit/regression tests song song.

M9.5 không được:

- thay semantic đã khóa;
- train/retrain model;
- promote official artifact;
- đổi threshold;
- làm application phụ thuộc `research/`.
"""


def print_review(
    root: Path,
    hashes: dict,
    gates: dict,
) -> None:
    print("=" * 92)
    print("M9.4.6 — CONTRACT REGISTRY + FINAL GATE")
    print("=" * 92)
    print("Project root:", root)

    print("\n[1] Contract artifacts")
    for key, filename in CONTRACT_FILES.items():
        print(f" - {filename}")
        print(f"   sha256={hashes[key]}")

    print("\n[2] Cross-contract consistency gates")
    for name, passed in gates.items():
        print(f" - {name}: {'PASS' if passed else 'FAIL'}")

    print("\n[3] Decision")
    print(
        " - M9.4 FINAL GATE:",
        "PASS" if all(gates.values()) else "FAIL",
    )
    print(
        " - M9.5 AUTHORIZATION:",
        "AUTHORIZED" if all(gates.values()) else "NOT AUTHORIZED",
    )


def execute(
    root: Path,
    contracts: dict,
    hashes: dict,
    gates: dict,
) -> None:
    if not all(gates.values()):
        raise RuntimeError(
            "M9.4 final gate chưa PASS; không finalize."
        )

    contract_dir = root / CONTRACT_DIR_REL
    registry_path = contract_dir / FINAL_REGISTRY_NAME
    canon_path = contract_dir / CANON_NAME

    existing = [
        path
        for path in (registry_path, canon_path)
        if path.exists()
    ]
    if existing:
        raise RuntimeError(
            "STOP: M9.4 final output đã tồn tại; "
            "không ghi đè tự động:\n"
            + "\n".join(str(p) for p in existing)
        )

    registry = build_registry(
        root=root,
        contracts=contracts,
        hashes=hashes,
        gates=gates,
    )

    registry_path.write_text(
        json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    canon_path.write_text(
        build_canon_markdown(registry),
        encoding="utf-8",
    )

    archive_dir = (
        root
        / "final_pipeline"
        / "tools"
        / "m9_04_contract_generation"
    )
    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archived = []
    missing_tools = []

    for name in ROOT_TOOL_SCRIPTS:
        src = root / name
        dst = archive_dir / name

        if not src.exists():
            missing_tools.append(name)
            continue

        if dst.exists():
            raise RuntimeError(
                f"Archive collision: {dst}"
            )

        shutil.move(str(src), str(dst))
        archived.append(str(dst.relative_to(root)))

    self_path = Path(__file__).resolve()
    self_archive = archive_dir / self_path.name

    if self_archive.exists():
        raise RuntimeError(
            f"Finalizer archive collision: {self_archive}"
        )
    shutil.copy2(self_path, self_archive)

    tools_readme = archive_dir / "README.md"
    if not tools_readme.exists():
        tools_readme.write_text(
            "# M9.4 Contract Generation Tools\n\n"
            "Các script trong thư mục này dùng để xây và audit "
            "governance contracts M9.4.\n\n"
            "Chúng không phải runtime dependency của final pipeline "
            "hoặc application.\n",
            encoding="utf-8",
        )

    print("\n" + "=" * 92)
    print("M9.4 FINALIZATION EXECUTION: PASS")
    print("M9.4 FINAL GATE: PASS")
    print("M9.5: AUTHORIZED")
    print("Final registry:")
    print(" -", registry_path.relative_to(root))
    print("Canonical readable contract:")
    print(" -", canon_path.relative_to(root))
    print("M9.4 tooling archived:")
    print(" -", archive_dir.relative_to(root))
    if missing_tools:
        print("Optional root tools not found:")
        for name in missing_tools:
            print(" -", name)

    print("\nOne root cleanup remains:")
    print(f" rm {self_path.name}")
    print("Run only after reading this PASS output.")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Write final M9.4 registry/CANON and archive "
            "contract-generation tooling."
        ),
    )
    args = parser.parse_args()

    root = detect_root()
    contracts = collect_contracts(root)
    hashes = contract_hashes(root)
    gates = cross_validate(contracts)

    print_review(
        root=root,
        hashes=hashes,
        gates=gates,
    )

    if not all(gates.values()):
        print("\nM9.4 FINAL GATE REVIEW: FAIL")
        print("STOP. Do not begin M9.5.")
        raise SystemExit(1)

    if not args.execute:
        print("\n" + "=" * 92)
        print("M9.4 FINAL GATE REVIEW: PASS")
        print("DRY-RUN ONLY — no final registry/archive written.")
        print("Execute finalization with:")
        print("python m9_04_06_finalize_contract_gate.py --execute")
        print("=" * 92)
        return

    execute(
        root=root,
        contracts=contracts,
        hashes=hashes,
        gates=gates,
    )


if __name__ == "__main__":
    main()
