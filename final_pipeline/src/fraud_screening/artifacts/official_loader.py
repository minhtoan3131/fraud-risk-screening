"""Official artifact-manifest loading and identity verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from fraud_screening.artifacts.model_loader import (
    MODEL_ID,
    POSITIVE_CLASS,
    VerifiedModelArtifact,
    load_verified_model,
    sha256_file,
)
from fraud_screening.errors import (
    ArtifactCompatibilityError,
    ArtifactFingerprintError,
    ArtifactNotFoundError,
)
from fraud_screening.preprocessing import (
    FrozenPreprocessor,
    load_frozen_preprocessor,
)


MANIFEST_RELATIVE_PATH = Path(
    "manifest/artifact_manifest.json"
)
EXPECTED_MANIFEST_VERSION = "1.0"
EXPECTED_THRESHOLD = 0.50

_EXPECTED_TOP_LEVEL_KEYS = {
    "manifest_version",
    "model",
    "preprocessing",
    "runtime_policy",
    "screening",
}

_EXPECTED_MODEL_KEYS = {
    "classes",
    "file",
    "model_id",
    "positive_class",
    "positive_class_index",
    "sha256",
}

_EXPECTED_PREPROCESSING_KEYS = {
    "encoded_feature_count",
    "file",
    "matrix_dtype",
    "matrix_format",
    "semantic_feature_count",
    "sha256",
}

_EXPECTED_RUNTIME_POLICY_KEYS = {
    "official_artifacts_read_only",
    "silent_research_fallback",
    "training_may_overwrite_official",
}

_EXPECTED_SCREENING_KEYS = {
    "risk_score_interface",
    "threshold",
    "threshold_comparator",
}


@dataclass(frozen=True, slots=True)
class OfficialArtifactBundle:
    """Verified official model + frozen preprocessing pair."""

    official_root: Path
    manifest_path: Path
    model: VerifiedModelArtifact
    preprocessor: FrozenPreprocessor
    manifest_version: str


def _require_exact_keys(
    value: Any,
    *,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(
            f"{label} must be a JSON object."
        )

    actual = set(value)

    if actual != expected:
        raise ArtifactCompatibilityError(
            f"{label} keys do not match the official contract."
        )

    return value


def _require_nonempty_string(
    value: Any,
    *,
    label: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactCompatibilityError(
            f"{label} must be a non-empty string."
        )

    return value


def _resolve_artifact_path(
    official_root: Path,
    relative_value: Any,
    *,
    label: str,
) -> Path:
    relative_text = _require_nonempty_string(
        relative_value,
        label=label,
    )

    relative = Path(
        relative_text
    )

    if relative.is_absolute():
        raise ArtifactCompatibilityError(
            f"{label} must be relative to the official artifact root."
        )

    candidate = (
        official_root
        / relative
    ).resolve()

    try:
        candidate.relative_to(
            official_root
        )
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            f"{label} escapes the official artifact root."
        ) from exc

    if not candidate.is_file():
        raise ArtifactNotFoundError(
            f"Official artifact not found: {candidate}"
        )

    return candidate


def _load_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    try:
        text = manifest_path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(
            f"Official manifest not found: {manifest_path}"
        ) from exc

    try:
        manifest = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise ArtifactCompatibilityError(
            "Official manifest is not valid JSON."
        ) from exc

    return _require_exact_keys(
        manifest,
        expected=_EXPECTED_TOP_LEVEL_KEYS,
        label="Official manifest",
    )


def load_official_artifacts(
    official_root: str | Path,
) -> OfficialArtifactBundle:
    """Load and verify the immutable official artifact bundle."""

    root = Path(
        official_root
    ).expanduser().resolve()

    if not root.is_dir():
        raise ArtifactNotFoundError(
            f"Official artifact root not found: {root}"
        )

    manifest_path = (
        root
        / MANIFEST_RELATIVE_PATH
    )

    manifest = _load_manifest(
        manifest_path
    )

    if (
        manifest["manifest_version"]
        != EXPECTED_MANIFEST_VERSION
    ):
        raise ArtifactCompatibilityError(
            "Unsupported official manifest version."
        )

    model_spec = _require_exact_keys(
        manifest["model"],
        expected=_EXPECTED_MODEL_KEYS,
        label="model",
    )

    preprocessing_spec = _require_exact_keys(
        manifest["preprocessing"],
        expected=_EXPECTED_PREPROCESSING_KEYS,
        label="preprocessing",
    )

    runtime_policy = _require_exact_keys(
        manifest["runtime_policy"],
        expected=_EXPECTED_RUNTIME_POLICY_KEYS,
        label="runtime_policy",
    )

    screening = _require_exact_keys(
        manifest["screening"],
        expected=_EXPECTED_SCREENING_KEYS,
        label="screening",
    )

    if model_spec["model_id"] != MODEL_ID:
        raise ArtifactCompatibilityError(
            "Manifest model_id mismatch."
        )

    if model_spec["classes"] != [0, 1]:
        raise ArtifactCompatibilityError(
            "Manifest model classes must be exactly [0, 1]."
        )

    if model_spec["positive_class"] != POSITIVE_CLASS:
        raise ArtifactCompatibilityError(
            "Manifest positive class mismatch."
        )

    if model_spec["positive_class_index"] != 1:
        raise ArtifactCompatibilityError(
            "Manifest positive class index must be 1."
        )

    if preprocessing_spec["encoded_feature_count"] != 47:
        raise ArtifactCompatibilityError(
            "Manifest encoded feature count must be 47."
        )

    if preprocessing_spec["semantic_feature_count"] != 10:
        raise ArtifactCompatibilityError(
            "Manifest semantic feature count must be 10."
        )

    if preprocessing_spec["matrix_dtype"] != "float32":
        raise ArtifactCompatibilityError(
            "Manifest matrix dtype must be float32."
        )

    if preprocessing_spec["matrix_format"] != "CSR":
        raise ArtifactCompatibilityError(
            "Manifest matrix format must be CSR."
        )

    expected_runtime_policy = {
        "official_artifacts_read_only": True,
        "silent_research_fallback": False,
        "training_may_overwrite_official": False,
    }

    if runtime_policy != expected_runtime_policy:
        raise ArtifactCompatibilityError(
            "Manifest runtime policy mismatch."
        )

    if (
        screening["risk_score_interface"]
        != "predict_proba positive-class score"
    ):
        raise ArtifactCompatibilityError(
            "Manifest risk-score interface mismatch."
        )

    if screening["threshold"] != EXPECTED_THRESHOLD:
        raise ArtifactCompatibilityError(
            "Manifest threshold mismatch."
        )

    if screening["threshold_comparator"] != ">":
        raise ArtifactCompatibilityError(
            "Manifest threshold comparator must be strict >."
        )

    model_path = _resolve_artifact_path(
        root,
        model_spec["file"],
        label="model.file",
    )

    preprocessing_path = _resolve_artifact_path(
        root,
        preprocessing_spec["file"],
        label="preprocessing.file",
    )

    model_sha = _require_nonempty_string(
        model_spec["sha256"],
        label="model.sha256",
    )

    preprocessing_sha = _require_nonempty_string(
        preprocessing_spec["sha256"],
        label="preprocessing.sha256",
    )

    model = load_verified_model(
        model_path,
        expected_sha256=model_sha,
    )

    if model.model_id != model_spec["model_id"]:
        raise ArtifactCompatibilityError(
            "Verified model ID differs from manifest."
        )

    if list(model.classes) != model_spec["classes"]:
        raise ArtifactCompatibilityError(
            "Verified model classes differ from manifest."
        )

    if model.positive_class != model_spec["positive_class"]:
        raise ArtifactCompatibilityError(
            "Verified positive class differs from manifest."
        )

    if (
        model.positive_class_index
        != model_spec["positive_class_index"]
    ):
        raise ArtifactCompatibilityError(
            "Verified positive class index differs from manifest."
        )

    actual_preprocessing_sha = sha256_file(
        preprocessing_path
    )

    if actual_preprocessing_sha != preprocessing_sha:
        raise ArtifactFingerprintError(
            "Preprocessing artifact SHA-256 mismatch."
        )

    preprocessor = load_frozen_preprocessor(
        preprocessing_path
    )

    if len(preprocessor.feature_names) != 47:
        raise ArtifactCompatibilityError(
            "Loaded preprocessing output width must be 47."
        )

    return OfficialArtifactBundle(
        official_root=root,
        manifest_path=manifest_path.resolve(),
        model=model,
        preprocessor=preprocessor,
        manifest_version=EXPECTED_MANIFEST_VERSION,
    )
