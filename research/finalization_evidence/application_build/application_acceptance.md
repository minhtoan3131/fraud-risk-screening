# Application Acceptance Evidence

The user-facing application was verified against the canonical final-pipeline
inference API.

Verified behaviors:

- cold-start end-to-end;
- explicit strict-prior history end-to-end;
- application result exactly equals direct final-pipeline inference;
- human-readable output;
- structured JSON output;
- invalid amount returns an explicit validation error;
- same-timestamp history is rejected;
- history from another card is rejected;
- deterministic demo flow;
- application source has no runtime dependency on research;
- application source does not import sklearn/joblib directly;
- application source does not directly call fit/predict/predict_proba;
- application surface contains no development-progress markers.
