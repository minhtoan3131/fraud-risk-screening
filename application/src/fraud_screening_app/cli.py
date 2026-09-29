"""Command-line entry point for the fraud-risk screening application."""

from __future__ import annotations

import argparse
import json
import sys

from fraud_screening.artifacts import MODEL_ID
from fraud_screening.errors import (
    FraudScreeningError,
)

from fraud_screening_app.input_io import (
    load_history_json,
    load_transaction_json,
)
from fraud_screening_app.presentation import (
    format_text_result,
    result_payload,
)
from fraud_screening_app.screening import (
    ScreeningApplication,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-screening-app",
        description="Transaction fraud-risk screening application.",
    )

    parser.add_argument(
        "--version-info",
        action="store_true",
        help="Show the connected final-pipeline model identity.",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    screen_parser = (
        subparsers.add_parser(
            "screen",
            help="Screen one transaction.",
        )
    )

    screen_parser.add_argument(
        "--transaction-json",
        required=True,
        help="Path to one transaction JSON object.",
    )

    screen_parser.add_argument(
        "--history-json",
        help=(
            "Optional path to a JSON list "
            "of strict-prior transactions."
        ),
    )

    screen_parser.add_argument(
        "--official-root",
        help=(
            "Optional official-artifact directory. "
            "If omitted, environment/repository defaults are used."
        ),
    )

    screen_parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of human-readable text.",
    )

    return parser


def _print_shell() -> None:
    print("=" * 72)
    print("TRANSACTION FRAUD-RISK SCREENING")
    print("=" * 72)
    print("Application shell: READY")
    print("Final-pipeline dependency: AVAILABLE")
    print("Model id:", MODEL_ID)
    print(
        "Use the 'screen' command with "
        "--transaction-json to evaluate a transaction."
    )
    print(
        "This program supports transaction screening; "
        "it does not make a final fraud accusation."
    )


def _run_screen(
    args: argparse.Namespace,
) -> int:
    try:
        current_record = (
            load_transaction_json(
                args.transaction_json
            )
        )

        history_records = (
            []
            if args.history_json is None
            else load_history_json(
                args.history_json
            )
        )

        application = (
            ScreeningApplication
            .from_official_artifacts(
                args.official_root
            )
        )

        result = application.screen(
            current_record,
            history_records=
                history_records,
        )

        if args.json:
            print(
                json.dumps(
                    result_payload(
                        current_record,
                        result,
                        history_count=len(
                            history_records
                        ),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(
                format_text_result(
                    current_record,
                    result,
                    history_count=len(
                        history_records
                    ),
                )
            )

        return 0

    except FraudScreeningError as exc:
        print(
            "Screening error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(
            "Input error: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version_info:
        print("Model id:", MODEL_ID)
        print("ML logic duplication in application: NO")
        print("Research runtime dependency: NO")
        return 0

    if args.command == "screen":
        return _run_screen(
            args
        )

    _print_shell()
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
