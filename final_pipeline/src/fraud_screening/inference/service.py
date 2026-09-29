"""Reusable end-to-end inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Any

from fraud_screening.artifacts import (
    OfficialArtifactBundle,
    VerifiedModelArtifact,
    load_official_artifacts,
)
from fraud_screening.data import (
    CanonicalTransaction,
    extract_scoring_payload,
    parse_scoring_payload,
)
from fraud_screening.features import (
    SemanticFeatureRow,
    build_semantic_feature_rows,
)
from fraud_screening.inference.scoring import (
    ScreeningBatchResult,
    score_encoded_matrix,
)
from fraud_screening.preprocessing import (
    FrozenPreprocessor,
)


@dataclass(frozen=True, slots=True)
class InferenceService:
    """One stable orchestration surface over the frozen official pipeline."""

    model: VerifiedModelArtifact
    preprocessor: FrozenPreprocessor

    def score_semantic_rows(
        self,
        rows: Sequence[SemanticFeatureRow],
    ) -> ScreeningBatchResult:
        encoded = self.preprocessor.transform(
            rows
        )

        return score_encoded_matrix(
            encoded,
            self.model,
        )

    def score_transactions(
        self,
        transactions: Sequence[CanonicalTransaction],
    ) -> ScreeningBatchResult:
        semantic_rows = build_semantic_feature_rows(
            transactions
        )

        return self.score_semantic_rows(
            semantic_rows
        )

    def score_raw_records(
        self,
        raw_records: Sequence[Mapping[str, Any]],
    ) -> ScreeningBatchResult:
        transactions = tuple(
            parse_scoring_payload(
                extract_scoring_payload(
                    record
                )
            )
            for record in raw_records
        )

        return self.score_transactions(
            transactions
        )


def load_official_inference_service(
    official_root: str | Path,
) -> InferenceService:
    """Load the verified official bundle and expose one inference service."""

    bundle: OfficialArtifactBundle = (
        load_official_artifacts(
            official_root
        )
    )

    return InferenceService(
        model=bundle.model,
        preprocessor=bundle.preprocessor,
    )
