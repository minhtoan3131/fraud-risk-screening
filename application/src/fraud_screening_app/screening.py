"""Application integration with the canonical inference service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from fraud_screening.inference import (
    InferenceService,
    InteractiveScreeningResult,
    load_official_inference_service,
    score_interactive_transaction,
)

from fraud_screening_app.config import (
    resolve_official_root,
)


@dataclass(frozen=True, slots=True)
class ScreeningApplication:
    service: InferenceService

    @classmethod
    def from_official_artifacts(
        cls,
        official_root: str | Path | None = None,
    ) -> "ScreeningApplication":
        resolved = (
            resolve_official_root(
                official_root
            )
        )

        return cls(
            service=(
                load_official_inference_service(
                    resolved
                )
            ),
        )

    def screen(
        self,
        current_record: Mapping[
            str,
            object,
        ],
        *,
        history_records: Sequence[
            Mapping[str, object]
        ] = (),
    ) -> InteractiveScreeningResult:
        return (
            score_interactive_transaction(
                self.service,
                current_record,
                history_records=
                    history_records,
            )
        )
