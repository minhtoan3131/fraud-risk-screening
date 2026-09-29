"""Stable raw-input schema for transaction screening."""

RAW_SCORING_FIELDS = (
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
)

EXCLUDED_DATASET_FIELDS = (
    "MCC",
    "Errors?",
    "Is Fraud?",
)

SEMANTIC_LOCATION_STATES = (
    "NON_PHYSICAL_OR_ONLINE",
    "PHYSICAL_COMPLETE",
    "PHYSICAL_ZIP_UNAVAILABLE",
)
