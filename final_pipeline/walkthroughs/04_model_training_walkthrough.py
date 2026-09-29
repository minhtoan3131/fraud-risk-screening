#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "final_pipeline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""Explain the isolated rebuild/training path without silently fitting anything."""

import subprocess

from fraud_screening.artifacts.model_loader import (
    EXPECTED_MODEL_PARAMETERS,
    MODEL_ID,
)

TRAIN_SCRIPT = PROJECT_ROOT / "final_pipeline" / "scripts" / "train_model.py"


def main() -> None:
    print("=" * 92)
    print("04 — MODEL TRAINING WALKTHROUGH")
    print("=" * 92)

    print("\n[1] Frozen configuration identity")
    print(" - model/config id:", MODEL_ID)
    print(" - estimator family: RandomForestClassifier")
    for name, value in EXPECTED_MODEL_PARAMETERS.items():
        print(f" - {name}: {value}")

    print("\n[2] Identity distinction")
    print(" - same config id != same fitted estimator bytes")
    print(" - a new fit is a NEW_REBUILD_MODEL")
    print(" - official M8 metrics do not automatically transfer to a rebuild")

    print("\n[3] Rebuild command contract")
    print(" - CLI:", TRAIN_SCRIPT.relative_to(PROJECT_ROOT))
    print(" - default: validate canonical training bundle only")
    print(
        " - --execute: fit preprocessing + model into "
        "final_pipeline/outputs/rebuilds/<run_id>/"
    )
    print(" - official artifact overwrite: FORBIDDEN")

    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "train_model.py --help failed:\n" + completed.stderr
        )

    print("\n[4] CLI help smoke")
    for line in [x for x in completed.stdout.splitlines() if x.strip()][:20]:
        print(" ", line)

    print("\n[5] This walkthrough intentionally does NOT train")
    print(" - preprocessing fit: NO")
    print(" - model.fit(): NO")
    print(" - artifact write: NO")
    print(" - official promotion: NO")
    print("\n04 MODEL TRAINING WALKTHROUGH: PASS")


if __name__ == "__main__":
    main()
