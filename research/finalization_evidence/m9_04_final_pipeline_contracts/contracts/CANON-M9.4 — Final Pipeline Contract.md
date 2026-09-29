# CANON-M9.4 — Final Pipeline Contract

**Project:** AI Transaction Fraud Risk Screening  
**Milestone:** M9.4 — Final Pipeline Contracts  
**Decision:** `PASS`  
**M9.5 authorization:** `AUTHORIZED`

---

## 1. Mục tiêu

M9.4 khóa interface và semantics của final pipeline trước khi trích xuất implementation Python ở M9.5.

M9.4 không train model, không refit preprocessing, không chạy inference và không promote official artifact.

---

## 2. Contract artifacts

- `contract_source_registry.json` — SHA-256 `62c7e33809f9d08da819babb9bfde0b3ceac263d5366537449d539d25746b092`
- `transaction_input_data_contract.json` — SHA-256 `9471b406125d22e4d2be8ad0fd4e64052dabb3eb9958cb4ccfd07441013e8457`
- `history_feature_contract.json` — SHA-256 `228d61b9c9ee0832b3e0561ac6333e08bbdb001541b3600b1190f6134949c81f`
- `preprocessing_model_threshold_contract.json` — SHA-256 `0622bc1bcabb939da7f168ddc1b45a57aae473986300158e3f100e7dbcff2dfe`
- `inference_output_artifact_error_contract.json` — SHA-256 `e813ba029b227a67c69466ccbe162cbb811c9a3627cbdd0eaeb0e3401ec57e37`

Registry tổng:

- `m9_04_final_contract_registry.json`

---

## 3. Input / data boundary

Frozen scoring payload sử dụng 12 raw fields:

- `User`
- `Card`
- `Year`
- `Month`
- `Day`
- `Time`
- `Amount`
- `Use Chip`
- `Merchant Name`
- `Merchant City`
- `Merchant State`
- `Zip`

`MCC` không thuộc frozen baseline.

`Errors?` và `Is Fraud?` không được sử dụng làm inference input.

---

## 4. Strict-causal history

Rule bắt buộc:

`Timestamp(history) < Timestamp(current)`

Không cho phép:

- current transaction làm history;
- same-timestamp peer làm history;
- future transaction;
- future-inclusive aggregate;
- target-label history.

State chỉ update sau khi toàn bộ transaction ở cùng prediction timestamp đã được tính.

Cold-start:

- `has_prior_card_history = False`
- `time_since_previous_transaction_min = NA`
- `transactions_last_1h = 0`
- `amount_minus_previous_mean = NA`
- `is_new_merchant = True`

---

## 5. Exact semantic feature interface

1. `amount_numeric`
2. `time_since_previous_transaction_min`
3. `transactions_last_1h`
4. `amount_minus_previous_mean`
5. `is_new_merchant`
6. `has_prior_card_history`
7. `transaction_mode`
8. `location_state`
9. `hour_of_day`
10. `day_of_week`

Pre-encoding feature count: `10`.

---

## 6. Frozen preprocessing

Numeric:

- `amount_numeric`
- `time_since_previous_transaction_min`
- `transactions_last_1h`
- `amount_minus_previous_mean`

Boolean:

- `is_new_merchant`
- `has_prior_card_history`

Categorical:

- `transaction_mode`
- `location_state`
- `hour_of_day`
- `day_of_week`

Frozen output:

- width: `47`
- matrix: `CSR sparse matrix`
- dtype: `float32`

Preprocessing state SHA-256:

`c1dc5486acdcd43f5f75fbd0a11bd7ad3a3b00761d50ae96b2b1fbe5b5d9ef98`

---

## 7. Frozen model

Model ID:

`RF-REF-100-GINI-SQRT-UNPRUNED-CW`

Model:

`RandomForestClassifier`

Classes:

`[0, 1]`

Positive class:

`1`

Positive-class index:

`1`

Frozen estimator SHA-256:

`61bdeeba5fd163cce8a5a9efa028a9cdc2f95c0796822a5222c0604c0302115f`

---

## 8. Risk score và threshold

Risk score:

`predict_proba` positive-class / class-1 score.

Không được gọi là calibrated confidence hoặc real-world fraud probability.

Frozen threshold:

`0.50`

Comparator:

`risk_score > 0.50`

Do đó:

- `0.49` → negative screening
- `0.50` → negative screening
- `0.51` → positive screening

---

## 9. Minimum inference output

- `risk_score`
- `threshold`
- `screening_prediction`
- `model_id`

Optional:

- `cold_start`
- `warnings`

`screening_prediction = 1` có nghĩa giao dịch bị flag bởi screening rule, không phải ground-truth fraud label.

---

## 10. Official artifact policy

Trước M9.6, frozen artifact trong `research/` chỉ là source identity/provenance.

Ở M9.6:

M8 evaluated artifact  
→ byte-preserving copy  
→ SHA-256 verification  
→ `final_pipeline/artifacts/official/`

Sau M9.6:

- application không load artifact từ `research/`;
- final pipeline runtime không fallback sang `research/`;
- training/rebuild không ghi đè official artifact.

---

## 11. Error policy

Fatal categories:

- `InputValidationError`
- `HistoryValidationError`
- `PreprocessingContractError`
- `ArtifactNotFoundError`
- `ArtifactFingerprintError`
- `ArtifactCompatibilityError`
- `InferenceContractError`

Không silent repair model/preprocessing/threshold.

Cold-start là trạng thái hợp lệ, không phải exception.

---

## 12. Architecture boundary

`research/` = historical evidence / provenance.

`final_pipeline/` = canonical ML implementation.

`application/` = user-facing consumer.

Sau M9 Gate:

- application runtime dependency on research = `NONE`;
- final_pipeline runtime dependency on research = `NONE`;
- core ML logic có một source of truth duy nhất.

---

## 13. Cross-contract gate

- `G01_ALL_SUBCONTRACTS_PASS`: **PASS**
- `G02_EXACT_10_FEATURE_ORDER_CONSISTENT`: **PASS**
- `G03_FEATURE_COUNTS_10_TO_47_CONSISTENT`: **PASS**
- `G04_STRICT_CAUSAL_RULE_CONSISTENT`: **PASS**
- `G05_COLD_START_CONSISTENT`: **PASS**
- `G06_MODEL_ID_CONSISTENT`: **PASS**
- `G07_MODEL_FINGERPRINT_CONSISTENT`: **PASS**
- `G08_PREPROCESSING_FINGERPRINT_CONSISTENT`: **PASS**
- `G09_POSITIVE_CLASS_CONSISTENT`: **PASS**
- `G10_THRESHOLD_STRICT_GT_CONSISTENT`: **PASS**
- `G11_OUTPUT_FIELDS_EXACT_MINIMUM`: **PASS**
- `G12_TARGET_EXCLUDED_FROM_INFERENCE`: **PASS**
- `G13_ERRORS_FIELD_EXCLUDED`: **PASS**
- `G14_RAW_IDENTIFIERS_NOT_DIRECT_FEATURES`: **PASS**
- `G15_RUNTIME_RESEARCH_DEPENDENCY_FORBIDDEN`: **PASS**
- `G16_PROMOTION_DEFERRED_AND_PROTECTED`: **PASS**
- `G17_RISK_SCORE_LANGUAGE_CONSISTENT`: **PASS**
- `G18_APPLICATION_BOUNDARY_LOCKED`: **PASS**
- `G19_SOURCE_REGISTRY_HAS_10_TARGET_CONTRACTS`: **PASS**
- `G20_NO_SCIENTIFIC_CHANGE_ACROSS_M9_4`: **PASS**

Overall:

`PASS`

---

## 14. Handoff sang M9.5

M9.5 được phép bắt đầu:

`Canonical Implementation Extraction`

Thứ tự ưu tiên:

1. data + validation;
2. transaction features;
3. behavioral/history features;
4. frozen preprocessing;
5. artifact/config loader;
6. inference primitives;
7. unit/regression tests song song.

M9.5 không được:

- thay semantic đã khóa;
- train/retrain model;
- promote official artifact;
- đổi threshold;
- làm application phụ thuộc `research/`.
