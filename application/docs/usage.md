# Usage

## Screen one transaction

Cold-start:

    fraud-screening-app screen \
      --transaction-json application/examples/current_transaction.json

With explicit strict-prior history:

    fraud-screening-app screen \
      --transaction-json application/examples/current_transaction.json \
      --history-json application/examples/history.json

Structured output:

    fraud-screening-app screen \
      --transaction-json application/examples/current_transaction.json \
      --history-json application/examples/history.json \
      --json

## Official artifacts

By default the repository-local official artifact directory is used.

A different directory can be supplied with:

    --official-root /absolute/path/to/official

or environment variable:

    FRAUD_SCREENING_OFFICIAL_ROOT=/absolute/path/to/official

## History behavior

History is optional.

When supplied, it must be strict-prior history for the same User+Card.
Invalid history is rejected by the final-pipeline inference API.

The application never invents history and never looks it up silently.

## Interpretation

`risk_score` is the positive-class model score used for screening.

The output is screening support, not a final fraud accusation.
