from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_VERSION = "M9.4.1-contract-source-registry-v1"

EXPECTED_MODEL_SHA256 = (
    "61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f"
)

EXPECTED_PREPROCESSING_SHA256 = (
    "c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98"
)


def detect_project_root() -> Path:
    root = Path.cwd().resolve()

    required_dirs = [
        root / "research",
        root / "final_pipeline",
        root / "application",
    ]

    if not all(path.is_dir() for path in required_dirs):
        raise RuntimeError(
            "Không đứng ở project root sau M9.3.\n"
            "Cần có đồng thời research/, final_pipeline/, application/."
        )

    return root


def load_m9_3_gate(root: Path) -> dict:
    registry_path = (
        root
        / "research"
        / "data"
        / "processed"
        / "m9_03_repository_migration"
        / "m9_03_final_registry.json"
    )

    if not registry_path.is_file():
        raise RuntimeError(
            "Không tìm thấy M9.3 final registry:\n"
            f"{registry_path}"
        )

    registry = json.loads(
        registry_path.read_text(encoding="utf-8")
    )

    if registry.get("decision") != "PASS":
        raise RuntimeError(
            "M9.3 final registry không ở trạng thái PASS."
        )

    if registry.get("m9_4_authorization") != "AUTHORIZED":
        raise RuntimeError(
            "M9.4 chưa được authorize bởi M9.3."
        )

    return registry


def build_registry(m9_3: dict) -> dict:
    sources = {
        "M4_FEATURE_PREPROCESSING": {
            "source_document": (
                "CANON-M4.8 — Tổng hợp Feature Specification v1.0, "
                "Preprocessing Specification v1.0, "
                "Behavioral Feature Contract, Decision Log và M4 Gate"
            ),
            "role": (
                "Canonical source for feature semantics, "
                "history semantics and preprocessing semantics"
            ),
            "locked_decisions": [
                "exactly 10 pre-encoding semantic features",
                "strict causal history: Timestamp(history) < Timestamp(current)",
                "same-timestamp peers cannot be history for each other",
                "historical warm-up is allowed and preserved",
                "cold-start semantics are explicit structural state",
                "numeric branch uses TRAIN-only StandardScaler state",
                "boolean branch is passthrough float32",
                "categorical branch uses TRAIN-only OneHotEncoder vocabulary",
                "unknown categories map to explicit __UNKNOWN__",
                "final representation width is 47",
                "final representation is CSR sparse float32",
                "raw User/Card/Merchant Name are history keys only, not direct classifier features",
                "Errors? is not part of baseline model input",
            ],
            "contract_targets": [
                "transaction_input_contract",
                "history_contract",
                "feature_contract",
                "preprocessing_contract",
            ],
        },
        "M7_MODEL_THRESHOLD": {
            "source_document": (
                "CANON-M7.9 — Final Selection Registry, Decision Log và M7 Gate"
            ),
            "role": (
                "Canonical source for selected model configuration "
                "and numerical threshold"
            ),
            "locked_decisions": [
                "training window = W_SHORT",
                "model family = RandomForestClassifier",
                "model config = RF-REF-100-GINI-SQRT-UNPRUNED-CW",
                "imbalance strategy = class_weight balanced",
                "random_state = 42",
                "probability interface = predict_proba",
                "positive class = 1",
                "risk score = positive-class score",
                "threshold = 0.50",
                "prediction comparator is strictly risk_score > 0.50",
            ],
            "contract_targets": [
                "model_contract",
                "threshold_contract",
                "inference_output_contract",
            ],
        },
        "M8_FROZEN_EVALUATED_SUBJECT": {
            "source_document": (
                "CANON-M8.6 — Final Evaluation Registry, Decision Log và M8 Gate"
            ),
            "role": (
                "Canonical source for frozen evaluated subject, "
                "risk-score language and no-retroactive-optimization guardrail"
            ),
            "locked_decisions": [
                "M9 default packaging subject is the same M8 evaluated frozen subject",
                "no silent TRAIN+VALIDATION refit replacement",
                "risk score may be called risk score / positive-class score",
                "do not call score calibrated confidence",
                "do not call score real-world fraud probability",
                "retroactive optimization using FINAL TEST is prohibited",
            ],
            "contract_targets": [
                "model_contract",
                "artifact_contract",
                "inference_output_contract",
            ],
        },
        "M9_ARCHITECTURE_TRANSITION": {
            "source_document": (
                "CANON-M9.2 — Final Architecture & Transition Charter"
            ),
            "role": (
                "Canonical source for repository boundaries, "
                "runtime dependency direction and official artifact policy"
            ),
            "locked_decisions": [
                "research is provenance/evidence, not runtime dependency",
                "canonical implementation lives in final_pipeline",
                "application depends on final_pipeline only",
                "notebooks consume source; source does not depend on notebooks",
                "one canonical implementation per core logic",
                "official evaluated artifact is promoted byte-preserving",
                "rebuild outputs must not overwrite official artifacts",
                "silent official overwrite is prohibited",
            ],
            "contract_targets": [
                "artifact_contract",
                "error_validation_contract",
                "implementation_boundary_contract",
            ],
        },
        "M9_3_REPOSITORY_MIGRATION": {
            "source_document": (
                "research/data/processed/m9_03_repository_migration/"
                "m9_03_final_registry.json"
            ),
            "role": (
                "Runtime evidence that three-zone migration completed "
                "without changing frozen scientific identity"
            ),
            "locked_decisions": [
                "research/final_pipeline/application zones exist",
                "legacy data/notebooks/reports moved under research",
                "M9.4 is authorized",
                "frozen model identity preserved",
                "frozen preprocessing identity preserved",
            ],
            "contract_targets": [
                "implementation_boundary_contract",
                "artifact_contract",
            ],
        },
    }

    frozen_identity = {
        "model_sha256": m9_3.get(
            "model_sha256",
            EXPECTED_MODEL_SHA256,
        ),
        "preprocessing_sha256": m9_3.get(
            "preprocessing_sha256",
            EXPECTED_PREPROCESSING_SHA256,
        ),
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "expected_preprocessing_sha256":
            EXPECTED_PREPROCESSING_SHA256,
    }

    return {
        "registry_version": REGISTRY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "purpose": (
            "Trace each M9.4 final-pipeline contract back to "
            "the reviewed scientific and architectural source."
        ),
        "source_precedence": [
            "verified runtime evidence",
            "reviewed notebook evidence",
            "latest locked CANON",
            "plan / generic guidance",
        ],
        "frozen_identity": frozen_identity,
        "sources": sources,
        "planned_m9_4_contracts": [
            "transaction_input_contract",
            "history_contract",
            "feature_contract",
            "preprocessing_contract",
            "model_contract",
            "threshold_contract",
            "inference_output_contract",
            "artifact_contract",
            "error_validation_contract",
            "implementation_boundary_contract",
        ],
        "guardrails": {
            "model_fit_performed": False,
            "prediction_performed": False,
            "preprocessing_refit_performed": False,
            "feature_contract_changed": False,
            "threshold_changed": False,
            "official_artifact_overwrite_performed": False,
        },
        "next_step": (
            "M9.4.2 — Transaction Input + Data Contract"
        ),
    }


def validate_registry(registry: dict) -> dict:
    checks = {}

    checks["G01_SOURCE_PRECEDENCE_DEFINED"] = (
        len(registry["source_precedence"]) >= 3
    )

    checks["G02_M4_SOURCE_PRESENT"] = (
        "M4_FEATURE_PREPROCESSING" in registry["sources"]
    )

    checks["G03_M7_SOURCE_PRESENT"] = (
        "M7_MODEL_THRESHOLD" in registry["sources"]
    )

    checks["G04_M8_SOURCE_PRESENT"] = (
        "M8_FROZEN_EVALUATED_SUBJECT" in registry["sources"]
    )

    checks["G05_M9_ARCH_SOURCE_PRESENT"] = (
        "M9_ARCHITECTURE_TRANSITION" in registry["sources"]
    )

    checks["G06_M9_3_RUNTIME_EVIDENCE_PRESENT"] = (
        "M9_3_REPOSITORY_MIGRATION" in registry["sources"]
    )

    checks["G07_MODEL_FINGERPRINT_LOCKED"] = (
        registry["frozen_identity"]["model_sha256"]
        == EXPECTED_MODEL_SHA256
    )

    checks["G08_PREPROCESSING_FINGERPRINT_LOCKED"] = (
        registry["frozen_identity"]["preprocessing_sha256"]
        == EXPECTED_PREPROCESSING_SHA256
    )

    checks["G09_CONTRACT_TARGETS_COMPLETE"] = (
        len(registry["planned_m9_4_contracts"]) == 10
    )

    guardrails = registry["guardrails"]

    checks["G10_NO_SCIENTIFIC_CHANGE"] = all(
        value is False
        for value in guardrails.values()
    )

    return checks


def main() -> None:
    root = detect_project_root()
    m9_3 = load_m9_3_gate(root)

    print("=" * 82)
    print("M9.4.1 — CONTRACT SOURCE AUDIT")
    print("=" * 82)
    print("Project root:", root)
    print("M9.3 decision:", m9_3.get("decision"))
    print("M9.4 authorization:", m9_3.get("m9_4_authorization"))
    print(
        "Mode: governance artifact only — "
        "NO FIT / NO PREDICTION / NO TRANSFORM"
    )

    registry = build_registry(m9_3)
    checks = validate_registry(registry)

    print("\n[1] Frozen identity")
    print(
        " - model:",
        registry["frozen_identity"]["model_sha256"],
    )
    print(
        " - preprocessing:",
        registry["frozen_identity"]["preprocessing_sha256"],
    )

    print("\n[2] Contract source map")
    for source_id, source in registry["sources"].items():
        print(f"\n - {source_id}")
        print("   source:", source["source_document"])
        print("   targets:")
        for target in source["contract_targets"]:
            print("    *", target)

    print("\n[3] M9.4.1 gate")
    for name, passed in checks.items():
        print(f" - {name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "M9.4.1 gate FAIL. Không ghi registry."
        )

    output_dir = (
        root
        / "final_pipeline"
        / "docs"
        / "contracts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "contract_source_registry.json"
    )

    if output_path.exists():
        raise RuntimeError(
            "STOP: contract source registry đã tồn tại; "
            "không ghi đè tự động:\n"
            f"{output_path}"
        )

    output_path.write_text(
        json.dumps(
            {
                **registry,
                "gates": checks,
                "gate_status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("\n[4] Registry written")
    print(" -", output_path.relative_to(root))

    print("\n" + "=" * 82)
    print("M9.4.1 CONTRACT SOURCE AUDIT: PASS")
    print("Contract sources are traceable and frozen.")
    print("No model fit.")
    print("No prediction.")
    print("No preprocessing transform.")
    print("NEXT: M9.4.2 — Transaction Input + Data Contract")
    print("=" * 82)


if __name__ == "__main__":
    main()
