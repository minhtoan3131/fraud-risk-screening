# Data and Artifacts

## Raw dataset

Canonical raw dataset:

    card_transaction.v1.csv

Identity:

    size_bytes = 2354626737
    rows = 24386900
    columns = 15
    SHA256 = 68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de

Raw data không được bundle trong `final_pipeline/` hoặc `application/`.

Kiểm tra nhanh:

    python final_pipeline/scripts/inspect_raw_dataset.py \
      --data-path "../fraud-risk-screening-local-data/ibm_tabformer/card_transaction.v1.csv" \
      --fast

Kiểm tra full SHA-256 + row count:

    python final_pipeline/scripts/inspect_raw_dataset.py \
      --data-path /absolute/path/to/card_transaction.v1.csv

Full inspection đọc toàn bộ file nhiều GB.

## Official model artifact

Path:

    final_pipeline/artifacts/official/model/model.joblib

Model/config identity:

    RF-REF-100-GINI-SQRT-UNPRUNED-CW

SHA-256:

    61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f

## Official preprocessing artifact

Path:

    final_pipeline/artifacts/official/preprocessing/preprocessing_state.json

SHA-256:

    c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98

Semantic input features:

    10

Encoded feature width:

    47

## Official manifest

Path:

    final_pipeline/artifacts/official/manifest/artifact_manifest.json

Runtime loader xác minh manifest, path containment, fingerprint, model identity,
class semantics, preprocessing width và threshold contract trước inference.

## Rebuild behavior

Training/rebuild là luồng riêng với official inference.

Rebuild output:

    final_pipeline/outputs/rebuilds/<run_id>/

Rebuild không được ghi đè:

    final_pipeline/artifacts/official/

Một estimator được fit lại là rebuild artifact mới. Official evaluation metrics
không tự động được kế thừa cho rebuild đó.
