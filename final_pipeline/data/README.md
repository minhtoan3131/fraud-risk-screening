# Raw dataset contract

`final_pipeline` does not bundle the large raw dataset and does not read it from
`research/`.

For dataset replay, provide an explicit external/local path to the canonical
artifact.

Canonical identity:

- filename: `card_transaction.v1.csv`
- size: `2,354,626,737` bytes
- SHA-256: `68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de`
- rows: `24,386,900`
- columns: `15`

Fast read-only inspection:

```bash
python final_pipeline/scripts/inspect_raw_dataset.py \
  --data-path /absolute/path/to/card_transaction.v1.csv \
  --fast
```

Use the full inspection without `--fast` when exact artifact identity must be
re-verified. Full inspection reads the entire multi-gigabyte CSV to compute
SHA-256 and row count.
