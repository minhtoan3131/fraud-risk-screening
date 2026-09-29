"""Small deterministic demo suite for the application surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_screening.errors import FraudScreeningError

from fraud_screening_app.presentation import result_payload
from fraud_screening_app.screening import ScreeningApplication


HERE = Path(__file__).resolve().parents[2]
EXAMPLES = HERE / "examples"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scenario(
    application: ScreeningApplication,
    *,
    name: str,
    current_file: str,
    history_file: str | None = None,
) -> dict[str, object]:
    current = _read_json(EXAMPLES / current_file)

    history = []
    if history_file is not None:
        history = _read_json(EXAMPLES / history_file)

    result = application.screen(
        current,
        history_records=history,
    )

    payload = result_payload(
        current,
        result,
        history_count=len(history),
    )

    print(
        f"{name}: "
        f"risk_score={payload['risk_score']:.8f}, "
        f"prediction={payload['screening_prediction']}, "
        f"cold_start={payload['cold_start']}, "
        f"warnings={payload['warnings']}"
    )

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small deterministic fraud-screening demo suite."
    )
    parser.add_argument(
        "--official-root",
        help="Optional official artifact directory.",
    )
    args = parser.parse_args()

    application = ScreeningApplication.from_official_artifacts(
        args.official_root
    )

    print("=" * 72)
    print("FRAUD-RISK SCREENING DEMO")
    print("=" * 72)

    cold = _scenario(
        application,
        name="cold-start",
        current_file="current_transaction.json",
    )

    known = _scenario(
        application,
        name="explicit-history",
        current_file="current_transaction.json",
        history_file="history.json",
    )

    if not cold["cold_start"]:
        raise RuntimeError("Cold-start demo did not report cold_start=True.")

    if known["cold_start"]:
        raise RuntimeError(
            "Explicit-history demo unexpectedly reported cold_start=True."
        )

    print("Demo result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
