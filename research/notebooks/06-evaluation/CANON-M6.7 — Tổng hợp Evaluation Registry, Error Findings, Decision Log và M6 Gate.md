# CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate

## 0. Trạng thái tài liệu

Milestone:

`M6 — Evaluation + Error Analysis`

Substep:

`M6.7 — Consolidated Evaluation Registry, Findings, Decision Log và M6 Gate`

Loại công việc:

`CANONICAL CONSOLIDATION / MILESTONE GATE`

Câu hỏi trung tâm:

> M6 đã hiểu đủ baseline behavior và error pattern để handoff M7 chưa?

Kết luận:

`YES`

Final M6 state:

`M6 — PASS`

Handoff:

`READY FOR M7 — MODEL SELECTION / ROBUSTNESS / TUNING`

M6.7 là bước tổng hợp evidence đã được runtime-review ở M6.2–M6.6.

M6.7 không tạo metric mới bằng suy đoán, không retrain model và không mở FINAL TEST.

---

# 1. Source hierarchy

M6.7 kế thừa trực tiếp:

- `CANON-Kế hoạch Milestone 6 — Evaluation và Error Analysis.md`;
- `CANON-M6.1 — Khóa Evaluation Charter, scope và guardrails.md`;
- `CANON-M6.2 — Evaluation artifact audit và independent metric reconstruction.md`;
- `CANON-M6.3 — Baseline metric và Confusion Matrix analysis.md`;
- `CANON-M6.4 — Controlled training-window comparison.md`;
- `CANON-M6.5 — Cross-model comparative evaluation.md`;
- `CANON-M6.6 — False Positive / False Negative error analysis.md`.

Upstream modeling / experiment protocol tiếp tục kế thừa:

- M5 baseline registry và official run identity;
- M4.7 baseline-ready representation;
- M3 temporal evaluation / metric / training-window / M7 policy.

Khi có xung đột:

`project-specific CANON > generic ML guidance`.

---

# 2. Vai trò chính thức của M6.7

M6.7 tổng hợp:

- Evaluation Registry;
- six-run canonical metrics;
- Confusion Matrix findings;
- predicted-positive behavior;
- controlled training-window findings;
- same-window cross-model findings;
- false-negative findings;
- false-positive findings;
- probability/risk-score findings;
- computational observations;
- open questions;
- M6 Decision Log;
- M6 Gate;
- M7 handoff.

M6.7 không khóa:

- final model;
- final training window;
- final imbalance strategy;
- final hyperparameters;
- final numerical threshold;
- final-test performance.

M6.7 không phải bước:

`FINAL MODEL SELECTION`.

---

# 3. Evaluation Charter được kế thừa

Problem:

```text
Binary supervised classification

Positive class:
fraud = 1

Prediction point:
transaction screening time
```

Evaluation direction:

`PAST → FUTURE`

Core M6 population:

```text
VALIDATION

2019-01-01 <= Timestamp < 2019-06-01

Rows:
712,458

Fraud:
1,052

Non-fraud:
711,406
```

FINAL TEST:

```text
2019-06-01 <= Timestamp < 2019-11-01

Status:
PROTECTED / NO ACCESS IN M6
```

Metric contract:

```text
Primary:
F1_fraud

Mandatory secondary:
Recall_fraud
Precision_fraud

Mandatory raw counts:
TP
FP
FN
TN

Operational:
predicted_positive_count
predicted_positive_rate

Accuracy:
REFERENCE ONLY
```

Probability:

`DESCRIPTIVE ANALYSIS ALLOWED`

Threshold optimization:

`PROHIBITED IN M6`

Retraining / tuning / imbalance intervention:

`NOT PERFORMED IN M6`

---

# 4. M6 substep completion registry

```text
+------+--------------------------------------------------------+---------------------------+
| Step | Scope                                                  | Final state               |
+------+--------------------------------------------------------+---------------------------+
| M6.1 | Evaluation Charter / scope / guardrails                 | PASS                      |
| M6.2 | Artifact audit + independent metric reconstruction      | PASS                      |
| M6.3 | Baseline metric + Confusion Matrix analysis             | PASS                      |
| M6.4 | Controlled W_SHORT vs W_LONG comparison                 | PASS                      |
| M6.5 | Same-window LR / DT / RF comparison                     | PASS                      |
| M6.6 | Transaction-level FP / FN error analysis                | PASS                      |
| M6.7 | Consolidation / M6 Gate / M7 handoff                    | PASS                      |
+------+--------------------------------------------------------+---------------------------+
```

Blocking issue:

`NONE`

---

# 5. Canonical Evaluation Registry — six official runs

Official subjects:

```text
Logistic Regression
M5-LR-SHORT-B04
M5-LR-LONG-B04

Decision Tree
M5-DT-SHORT-B01
M5-DT-LONG-B01

Random Forest
M5-RF-SHORT-B01
M5-RF-LONG-B01
```

Canonical six-run metric registry:

```text
+--------------------+---------+----------+----------+-----------+-----+-----+------+--------+--------+-------------+----------+
| Run                | Window  | F1       | Recall   | Precision | TP  | FP  | FN   | TN     | Alerts | Alert rate  | Accuracy |
+--------------------+---------+----------+----------+-----------+-----+-----+------+--------+--------+-------------+----------+
| M5-LR-SHORT-B04    | W_SHORT | 0.337475 | 0.245247 | 0.540881  | 258 | 219 | 794  | 711187 | 477    | 0.00066951  | 0.998578 |
| M5-LR-LONG-B04     | W_LONG  | 0.039964 | 0.020913 | 0.448980  | 22  | 27  | 1030 | 711379 | 49     | 0.00006878  | 0.998516 |
| M5-DT-SHORT-B01    | W_SHORT | 0.327056 | 0.313688 | 0.341615  | 330 | 636 | 722  | 710770 | 966    | 0.00135587  | 0.998094 |
| M5-DT-LONG-B01     | W_LONG  | 0.199134 | 0.196768 | 0.201558  | 207 | 820 | 845  | 710586 | 1027   | 0.00144149  | 0.997663 |
| M5-RF-SHORT-B01    | W_SHORT | 0.366467 | 0.290875 | 0.495146  | 306 | 312 | 746  | 711094 | 618    | 0.00086742  | 0.998515 |
| M5-RF-LONG-B01     | W_LONG  | 0.152709 | 0.088403 | 0.560241  | 93  | 73  | 959  | 711333 | 166    | 0.00023300  | 0.998551 |
+--------------------+---------+----------+----------+-----------+-----+-----+------+--------+--------+-------------+----------+
```

Interpretation guardrail:

`Accuracy không được đọc riêng lẻ như fraud-detection quality`.

Dù Accuracy của toàn bộ runs gần 0.998, Recall_fraud thay đổi mạnh và FN nằm trong khoảng:

```text
722 → 1,030
```

---

# 6. Evaluation artifact integrity

M6.2 đã xác minh:

```text
Prediction artifacts:
6 / 6 VERIFIED

Risk-score artifacts:
6 / 6 VERIFIED

Summary artifacts:
6 / 6 VERIFIED

Config locks:
3 / 3 VERIFIED

Pair manifests:
3 / 3 VERIFIED

Independent metric reconstruction:
6 / 6 PASS

Persisted-summary agreement:
6 / 6 PASS

Confusion arithmetic:
PASS

Controlled-pair metadata:
PASS

Evaluation Registry persistence:
PASS

Evaluation Registry round-trip:
PASS

FINAL TEST:
PROTECTED
```

Canonical M6.2 artifacts:

```text
data/processed/m6_02_evaluation_artifact_audit/
    m6_02_evaluation_registry.json
    m6_02_audit_manifest.json
```

M6.2 compatibility note:

M5.3 Logistic Regression summaries không chứa filename metadata theo schema cũ.

M6.2 không coi việc thiếu metadata đó là bằng chứng artifact integrity.

Identity vẫn được xác minh bằng:

- canonical filename convention;
- file existence;
- experiment identity;
- pair-manifest identity;
- SHA-256;
- independent metric reconstruction.

Status:

`SAFE LEGACY-SCHEMA COMPATIBILITY`

---

# 7. Confusion Matrix / operational findings

## M6-F01 — Raw count evidence là bắt buộc

Mọi run đã xác minh:

```text
TP + FN = 1,052

TP + FP = alert count

TN + FP = 711,406

TP + FP + FN + TN = 712,458
```

Status:

`VERIFIED FACT`

---

## M6-F02 — Accuracy không đủ để mô tả fraud screening

Observed:

```text
Accuracy:
≈ 0.9977 → 0.9986

Recall_fraud:
≈ 0.0209 → 0.3137
```

Finding:

Accuracy gần như cùng mức cao trong khi fraud capture khác nhau rất lớn.

Status:

`VERIFIED FACT`

---

## M6-F03 — Alert volume khác nhau mạnh giữa baseline runs

Observed:

```text
Minimum alert count:
49
LR-LONG

Maximum alert count:
1,027
DT-LONG
```

Operational interpretation:

Baseline model families tạo workload screening rất khác nhau.

Status:

`VERIFIED FACT`

---

# 8. Controlled training-window findings

Comparison scope:

```text
LR:
W_SHORT vs W_LONG

DT:
W_SHORT vs W_LONG

RF:
W_SHORT vs W_LONG
```

Controlled rule:

`ONLY TRAINING WINDOW CHANGES`

Delta convention:

`W_LONG − W_SHORT`

Summary:

```text
+---------------------+-----------+------------+-------------+-------+-------+-------+--------------------------+
| Family              | ΔF1       | ΔRecall    | ΔPrecision  | ΔTP   | ΔFP   | ΔFN   | Provisional F1 direction |
+---------------------+-----------+------------+-------------+-------+-------+-------+--------------------------+
| Logistic Regression | -0.297512 | -0.224335  | -0.091901   | -236  | -192  | +236  | W_SHORT                  |
| Decision Tree       | -0.127922 | -0.116920  | -0.140057   | -123  | +184  | +123  | W_SHORT                  |
| Random Forest       | -0.213758 | -0.202471  | +0.065095   | -213  | -239  | +213  | W_SHORT                  |
+---------------------+-----------+------------+-------------+-------+-------+-------+--------------------------+
```

## M6-F04 — Provisional F1 direction nhất quán

Observed:

```text
LR:
W_SHORT

DT:
W_SHORT

RF:
W_SHORT
```

Cross-pair classification:

`CONSISTENT_PROVISIONAL_DIRECTION — W_SHORT`

Status:

`COMPARATIVE EVIDENCE`

Boundary:

`NOT FINAL TRAINING-WINDOW SELECTION`

---

## M6-F05 — LR và DT có W_SHORT advantage trên cả F1 / Recall / Precision

Classification:

```text
LR:
SHORT_DOMINATES_F1_RECALL_PRECISION

DT:
SHORT_DOMINATES_F1_RECALL_PRECISION
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6-F06 — RF có Precision/Recall trade-off

Observed:

```text
W_SHORT:
higher F1
higher Recall

W_LONG:
higher Precision
```

Classification:

`SHORT_F1_RECALL__LONG_PRECISION`

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6-F07 — W_LONG có computational cost cao hơn

Observed fit-time ratio:

```text
LR:
LONG / SHORT ≈ 4.40x

DT:
LONG / SHORT ≈ 17.48x

RF:
LONG / SHORT ≈ 15.42x
```

Status:

`VERIFIED COMPUTATIONAL EVIDENCE`

Interpretation guardrail:

Runtime không phải automatic predictive winner rule.

---

# 9. Same-window cross-model findings

Comparison scope:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Mixed-window comparison:

`NOT USED`

## W_SHORT descriptive extrema

```text
Highest F1:
Random Forest

Highest Recall:
Decision Tree

Highest Precision:
Logistic Regression

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

## W_LONG descriptive extrema

```text
Highest F1:
Decision Tree

Highest Recall:
Decision Tree

Highest Precision:
Random Forest

Lowest FN:
Decision Tree

Lowest FP:
Logistic Regression

Lowest alert count:
Logistic Regression

Lowest fit time:
Logistic Regression
```

## M6-F08 — Không có model family dẫn toàn bộ criteria

Finding:

```text
MODEL TRADE-OFF EXISTS
```

Không có một family cùng lúc tối ưu:

- F1;
- Recall;
- Precision;
- FN;
- FP;
- alert burden;
- computational cost;

trong cả hai same-window groups.

Status:

`VERIFIED COMPARATIVE PATTERN`

Boundary:

`NOT FINAL MODEL RANKING`

---

## M6-F09 — W_SHORT pairwise model trade-off

RF-SHORT vs LR-SHORT:

```text
RF - LR

ΔF1:
+0.028992

ΔRecall:
+0.045627

ΔPrecision:
-0.045735

ΔTP:
+48

ΔFP:
+93
```

RF-SHORT vs DT-SHORT:

```text
RF - DT

ΔF1:
+0.039411

ΔRecall:
-0.022814

ΔPrecision:
+0.153531

ΔTP:
-24

ΔFP:
-324
```

Interpretation:

RF-SHORT có higher F1 trong W_SHORT nhưng vẫn có trade-off với Recall/Precision tùy model đối chiếu.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

## M6-F10 — W_LONG pairwise model trade-off

DT-LONG vs LR-LONG:

```text
ΔTP:
+185

ΔFP:
+793

ΔFN:
-185
```

RF-LONG vs DT-LONG:

```text
ΔF1:
-0.046425

ΔRecall:
-0.108365

ΔPrecision:
+0.358683

ΔFP:
-747

ΔFN:
+114
```

Interpretation:

W_LONG cho trade-off mạnh giữa fraud capture và false-alert burden.

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

---

# 10. Transaction-level lineage integrity

M6.6 đã rebuild canonical semantic validation representation từ raw artifact theo strict-causal contract.

Observed:

```text
Raw rows streamed:
24,386,900

Card blocks:
6,139

Development/history context rows:
23,038,920

Validation semantic rows:
712,458

First validation raw_row_id:
4776

Last validation raw_row_id:
24,385,729
```

Exact gate:

```text
reconstructed validation raw_row_id
==
M4.7 row_id_validation
```

Result:

`PASS`

Canonical strict-causal rule:

`Timestamp(history) < Timestamp(current)`

Same-timestamp peer:

`NOT HISTORY`

Raw semantic pass:

```text
Is Fraud?:
NOT READ

Errors?:
NOT READ
```

Status:

`TRANSACTION-LEVEL SEMANTIC LINEAGE VERIFIED`

---

# 11. FP / FN error findings

Primary groups:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

Six official runs:

`6 / 6 ANALYZED`

## M6-F11 — Error complementarity tồn tại giữa model families

W_SHORT fraud overlap:

```text
Actual fraud:
1,052

All three miss:
618
≈ 58.75%

Exactly one catches:
127

Exactly two catch:
154

All three catch:
153
≈ 14.54%

Caught by at least one:
434
≈ 41.25%
```

W_LONG fraud overlap:

```text
Actual fraud:
1,052

All three miss:
810
≈ 77.00%

Exactly one catches:
166

Exactly two catch:
72

All three catch:
4
≈ 0.38%

Caught by at least one:
242
≈ 23.00%
```

Finding:

LR / DT / RF không tạo identical fraud-error sets.

Status:

`VERIFIED PATTERN`

---

## M6-F12 — Shared-miss burden thấp hơn trong W_SHORT evidence

Observed:

```text
W_SHORT all-three miss:
618

W_LONG all-three miss:
810
```

Status:

`DESCRIPTIVE COMPARATIVE EVIDENCE`

Boundary:

`NOT FINAL WINDOW SELECTION`

---

## M6-F13 — `is_new_merchant` là cross-run error-pattern signal

Observed across all six official runs:

```text
Fraud side:

FN true-rate
<
TP true-rate
```

và:

```text
Non-fraud side:

FP true-rate
>
TN true-rate
```

Interpretation:

Trong synthetic validation dataset:

`is_new_merchant`

liên hệ với stronger fraud flagging nhưng đồng thời liên hệ với false-alert burden.

Status:

`DESCRIPTIVE ASSOCIATION`

Không diễn giải:

`CAUSAL RELATIONSHIP`

---

## M6-F14 — `location_state` không tách FN khỏi TP trên fraud side

Observed:

```text
PHYSICAL_ZIP_UNAVAILABLE

FN rate:
1.0

TP rate:
1.0
```

Finding:

Không có observed FN-vs-TP separation từ dimension này trong fraud population.

Status:

`DESCRIPTIVE`

---

## M6-F15 — `location_state` xuất hiện mạnh ở một số FP groups

Examples:

```text
LR-SHORT

PHYSICAL_ZIP_UNAVAILABLE

FP:
≈ 99.09%

TN:
≈ 0.59%
```

```text
RF-SHORT

PHYSICAL_ZIP_UNAVAILABLE

FP:
≈ 99.68%

TN:
≈ 0.58%
```

Status:

`DATASET-SPECIFIC FALSE-ALERT PATTERN`

M7 handoff:

`FOLLOW-UP HYPOTHESIS`

---

## M6-F16 — W_LONG subgroup interpretation cần small-n caution

Examples:

```text
LR-LONG:
TP = 22
FP = 27

RF-LONG:
TP = 93
FP = 73
```

Finding:

Categorical rate gaps từ các subgroup nhỏ chỉ mang tính mô tả.

Status:

`INTERPRETATION GUARDRAIL`

---

# 12. Probability / risk-score findings

All persisted risk-score artifacts:

`6 / 6 VERIFIED`

Scope:

`DESCRIPTIVE ONLY`

Threshold search:

`NONE`

Calibration claim:

`NONE`

## M6-F17 — Decision Tree error probabilities rất discrete/extreme

Observed:

```text
DT-SHORT FN:
median score = 0

DT-SHORT FP:
median score = 1

DT-LONG FN:
median score = 0

DT-LONG FP:
median score = 1
```

Error-band evidence:

```text
DT SHORT/LONG FN:
100% score < 0.10

DT SHORT/LONG FP:
100% score >= 0.90
```

Status:

`VERIFIED PROBABILITY BEHAVIOR`

Boundary:

`NOT CALIBRATION CLAIM`

---

## M6-F18 — Nhiều LR/RF false negatives nằm xa 0.5

Observed:

```text
LR-SHORT FN score < 0.40:
≈ 92.06%

LR-LONG FN score < 0.40:
≈ 96.99%

RF-SHORT FN score < 0.40:
≈ 88.34%

RF-LONG FN score < 0.40:
≈ 93.22%
```

Finding:

FN burden của LR/RF không chỉ gồm near-boundary cases.

Status:

`M7 THRESHOLD HYPOTHESIS INPUT`

Guardrail:

Không giả định threshold tuning sẽ tự động recover phần lớn FN.

---

# 13. Computational findings

Persisted baseline fit context:

```text
W_SHORT

LR:
~0.82 s

DT:
~6.66 s

RF:
~40.70 s
```

```text
W_LONG

LR:
~3.61 s

DT:
~116.47 s

RF:
~627.71 s
```

Finding:

Computational cost phụ thuộc mạnh vào:

- model family;
- training-window size.

Observed order:

```text
LR:
lowest baseline fit cost

DT:
intermediate

RF:
highest
```

Status:

`VERIFIED COMPUTATIONAL EVIDENCE`

M7 implication:

Robustness / CV / tuning design phải tính đến computational budget, đặc biệt với RF và W_LONG.

---

# 14. Consolidated Findings Registry

Findings được phân biệt rõ theo evidence type.

```text
+---------+--------------------------------------------------------------+-----------------------------------+
| ID      | Finding                                                      | Evidence class                    |
+---------+--------------------------------------------------------------+-----------------------------------+
| M6-F01  | Confusion arithmetic valid for 6 / 6 runs                    | VERIFIED FACT                     |
| M6-F02  | Accuracy không đại diện fraud detection quality              | VERIFIED FACT                     |
| M6-F03  | Alert workload khác mạnh giữa runs                           | VERIFIED FACT                     |
| M6-F04  | Provisional F1 direction = W_SHORT ở 3 / 3 families          | COMPARATIVE EVIDENCE              |
| M6-F05  | LR/DT: SHORT leads F1 + Recall + Precision                    | COMPARATIVE EVIDENCE              |
| M6-F06  | RF: SHORT F1/Recall vs LONG Precision trade-off               | COMPARATIVE EVIDENCE              |
| M6-F07  | W_LONG fit cost cao hơn trong 3 / 3 families                  | COMPUTATIONAL EVIDENCE            |
| M6-F08  | Không model family nào dẫn toàn bộ criteria                   | VERIFIED COMPARATIVE PATTERN      |
| M6-F09  | W_SHORT model families có metric/alert trade-off              | COMPARATIVE EVIDENCE              |
| M6-F10  | W_LONG model families có metric/alert trade-off               | COMPARATIVE EVIDENCE              |
| M6-F11  | LR/DT/RF có error complementarity                             | VERIFIED ERROR PATTERN            |
| M6-F12  | W_SHORT shared-miss burden thấp hơn W_LONG                    | COMPARATIVE ERROR EVIDENCE        |
| M6-F13  | is_new_merchant liên hệ fraud flagging + FP burden            | DESCRIPTIVE ASSOCIATION           |
| M6-F14  | location_state không tách FN/TP fraud                         | DESCRIPTIVE                       |
| M6-F15  | location_state tập trung mạnh ở một số FP groups              | DATASET-SPECIFIC HYPOTHESIS       |
| M6-F16  | Một số W_LONG error groups có small-n uncertainty             | INTERPRETATION GUARDRAIL          |
| M6-F17  | DT error scores discrete/extreme                              | PROBABILITY BEHAVIOR              |
| M6-F18  | Nhiều LR/RF FN nằm xa 0.5                                    | M7 THRESHOLD HYPOTHESIS INPUT     |
+---------+--------------------------------------------------------------+-----------------------------------+
```

---

# 15. Consolidated M6 Decision Log

## M6-D01 — Milestone role

Decision:

M6 là:

`EVALUATION + ERROR ANALYSIS`

không phải final model selection.

Status:

`LOCKED`

---

## M6-D02 — Evaluation population

Decision:

```text
VALIDATION
2019-01-01 <= Timestamp < 2019-06-01
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6-D03 — Positive class

Decision:

`fraud = 1`

Status:

`INHERITED — LOCKED`

---

## M6-D04 — Metric contract

Decision:

```text
Primary:
F1_fraud

Secondary:
Recall_fraud
Precision_fraud

Mandatory:
TP / FP / FN / TN

Operational:
predicted-positive count/rate

Accuracy:
REFERENCE ONLY
```

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6-D05 — Independent metric reconstruction

Decision:

Evaluation metric phải reconstruct từ:

`y_validation + persisted y_pred`

Observed:

`6 / 6 PASS`

Status:

`LOCKED / VERIFIED`

---

## M6-D06 — Probability evidence

Decision:

Use persisted risk score descriptively.

Threshold optimization:

`NO`

Calibration claim:

`NO`

Status:

`LOCKED / VERIFIED`

---

## M6-D07 — Controlled training-window comparison

Decision:

Within-family comparison hợp lệ khi:

`ONLY TRAINING WINDOW CHANGES`

Observed:

`3 / 3 controlled pairs PASS`

Status:

`LOCKED / VERIFIED`

---

## M6-D08 — Training-window comparative evidence

Observed:

```text
LR:
W_SHORT provisional F1 leader

DT:
W_SHORT provisional F1 leader

RF:
W_SHORT provisional F1 leader
```

Status:

`CONSISTENT PROVISIONAL DIRECTION`

Final training-window winner:

`OPEN`

---

## M6-D09 — Cross-model comparison

Decision:

Model-family comparison phải giữ training window cố định.

Observed:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

Status:

`LOCKED / VERIFIED`

---

## M6-D10 — Model-family evidence

Decision:

Record trade-off, không tạo final ranking.

Observed:

`MODEL TRADE-OFF EXISTS`

Model-family winner:

`OPEN`

---

## M6-D11 — Error-analysis primary groups

Decision:

```text
Fraud side:
FN vs TP

Non-fraud side:
FP vs TN
```

Observed:

`6 / 6 COMPLETE`

Status:

`LOCKED / VERIFIED`

---

## M6-D12 — Transaction-level lineage

Decision:

Row-level finding chỉ hợp lệ khi:

```text
reconstructed raw_row_id
==
M4.7 row_id_validation
```

Observed:

`712,458 / 712,458 EXACT`

Status:

`LOCKED / VERIFIED`

---

## M6-D13 — Strict causal semantic history

Decision:

`Timestamp(history) < Timestamp(current)`

Same timestamp:

`NOT HISTORY`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6-D14 — Raw target in semantic rebuild

Decision:

Do not read raw target.

Observed:

`Is Fraud? NOT READ`

Status:

`LOCKED / VERIFIED`

---

## M6-D15 — Error-pattern findings

Decision:

Handoff cross-run patterns và model-specific error behavior sang M7 như evidence / hypotheses.

Status:

`LOCKED`

---

## M6-D16 — Small-n interpretation

Decision:

Không overgeneralize categorical rates từ small TP/FP groups.

Status:

`LOCKED`

---

## M6-D17 — Retraining / tuning / imbalance intervention

Decision:

None in M6.

Observed:

```text
Retraining:
NO

Hyperparameter tuning:
NO

Class weight:
NO

Oversampling:
NO

Undersampling:
NO

SMOTE:
NO
```

Status:

`LOCKED / VERIFIED`

---

## M6-D18 — Threshold

Decision:

No threshold optimization in M6.

Final numerical threshold:

`OPEN — M7`

Status:

`LOCKED BOUNDARY`

---

## M6-D19 — Final model

Decision:

Not selected in M6.

Status:

`OPEN — M7`

---

## M6-D20 — Final training window

Decision:

Not selected in M6.

Status:

`OPEN — M7 / ROBUSTNESS IF REQUIRED`

---

## M6-D21 — FINAL TEST

Decision:

No access in M6.

Observed:

`PROTECTED`

Status:

`INHERITED — VERIFIED — LOCKED`

---

## M6-D22 — M6 completion criterion

Decision:

M6 PASS khi:

`evaluation evidence is complete, coherent and M7-ready`.

Observed:

`SATISFIED`

Status:

`LOCKED`

---

## M6-D23 — M7 handoff

Decision:

Handoff phải gồm:

- comparative evidence;
- error evidence;
- probability behavior;
- computational constraints;
- unresolved selection questions.

Status:

`READY`

---

# 16. Open Questions handoff sang M7

## O01 — Final training-window preference

Status:

`OPEN`

Current M6 evidence:

`CONSISTENT PROVISIONAL W_SHORT DIRECTION`

M7 phải quyết định có cần:

- temporal robustness;
- temporal CV;
- tie-break protocol;

trước khi khóa final training window.

---

## O02 — Final model-family preference

Status:

`OPEN`

M6 evidence cho thấy:

`MODEL TRADE-OFF EXISTS`

M7 chịu trách nhiệm selection.

---

## O03 — Class imbalance intervention

Status:

`OPEN — M7`

Authorized candidates từ upstream:

```text
NONE
CLASS_WEIGHT
RANDOM_OVERSAMPLING
RANDOM_UNDERSAMPLING
SMOTE — CONDITIONAL
```

M6 không thử các intervention này.

---

## O04 — Temporal CV / robustness

Status:

`DEFERRED TO M7`

Primary strategy nếu dùng:

`FORWARD / EXPANDING TEMPORAL VALIDATION`

Evidence cần xem xét:

- W_SHORT provisional direction nhất quán;
- W_LONG performance degradation ở nhiều runs;
- error-overlap difference giữa W_SHORT / W_LONG.

---

## O05 — Hyperparameter tuning

Status:

`DEFERRED TO M7`

Search space phải:

- nhỏ;
- có lý do;
- khai báo trước;
- không dùng FINAL TEST.

---

## O06 — Final numerical threshold

Status:

`DEFERRED TO M7`

M6 handoff evidence:

- LR/RF có nhiều FN nằm xa 0.5;
- DT error scores cực đoan;
- alert workload khác mạnh giữa models.

Implication:

Threshold work phải là explicit controlled experiment.

Không giả định threshold lowering là đủ.

---

## O07 — Probability calibration

Status:

`OPEN / NOT CORE M6`

Decision Tree score behavior cho thấy calibration/ranking behavior đáng được chú ý.

Nhưng M6 không thực hiện calibration experiment.

---

## O08 — Final project performance

Status:

`OPEN`

FINAL TEST chưa mở.

Không có final project performance claim trong M6.

---

## O09 — Dataset-specific false-alert structure

Status:

`OPEN — M7 FOLLOW-UP`

Question:

`location_state / is_new_merchant patterns có ổn định qua temporal folds hoặc selected configurations không?`

Guardrail:

Do dataset synthetic, không suy rộng trực tiếp sang production fraud behavior.

---

## O10 — Shared-miss fraud structure

Status:

`OPEN — M7 FOLLOW-UP`

Observed:

```text
W_SHORT all-three miss:
618

W_LONG all-three miss:
810
```

Question:

Robustness / imbalance / tuning có giảm shared-miss subset hay chỉ dịch chuyển model-specific errors?

---

# 17. M6 Gate

M6 Gate có 15 checks.

## G01 — M5 handoff integrity

Required:

- six official runs đúng identity;
- persisted artifacts đúng source/version;
- không stale experiment.

Evidence:

M6.2 artifact / config / pair-manifest audit PASS.

Result:

`PASS`

---

## G02 — Prediction artifact integrity

Required:

`6 / 6 y_pred` load đúng identity / shape / support.

Evidence:

`6 / 6 VERIFIED`

Result:

`PASS`

---

## G03 — Risk-score artifact integrity

Required:

`6 / 6 risk_score` finite / valid / correct identity.

Evidence:

`6 / 6 VERIFIED`

Result:

`PASS`

---

## G04 — Independent metric reproduction

Required:

Canonical metrics được reconstruct từ:

`y_validation + y_pred`

và match M5 summaries.

Evidence:

```text
6 / 6 independent metric checks:
PASS

6 / 6 summary match:
PASS
```

Result:

`PASS`

---

## G05 — Confusion Matrix reconstruction

Required:

`6 / 6 TP / FP / FN / TN`

và arithmetic consistency.

Evidence:

M6.2 / M6.3 / M6.6 all PASS.

Result:

`PASS`

---

## G06 — Predicted-positive analysis

Required:

Alert count / rate được đọc cho mọi run.

Evidence:

6 / 6 profiles có:

- alert count;
- alert rate;
- TP;
- FP;
- Precision.

Result:

`PASS`

---

## G07 — Controlled training-window analysis

Required:

```text
LR SHORT vs LONG
DT SHORT vs LONG
RF SHORT vs LONG
```

với đúng controlled scope.

Evidence:

```text
3 / 3 controlled pairs:
PASS

12 / 12 controlled fields per pair:
PASS
```

Result:

`PASS`

---

## G08 — Cross-model comparative analysis

Required:

```text
W_SHORT:
LR vs DT vs RF

W_LONG:
LR vs DT vs RF
```

không confound window effect.

Evidence:

```text
2 / 2 same-window groups:
PASS

Mixed-window model comparison:
NONE
```

Result:

`PASS`

---

## G09 — False-negative analysis

Required:

FN map về actual validation transactions và có evidence-based findings.

Evidence:

```text
Transaction-level mapping:
VERIFIED

FN vs TP:
6 / 6 COMPLETE
```

Result:

`PASS`

---

## G10 — False-positive analysis

Required:

FP map về actual validation transactions và có evidence-based findings.

Evidence:

```text
Transaction-level mapping:
VERIFIED

FP vs TN:
6 / 6 COMPLETE
```

Result:

`PASS`

---

## G11 — Probability behavior review

Required:

Persisted score evidence được audit và descriptive findings được tổng hợp mà không threshold-optimize.

Evidence:

```text
Risk scores:
6 / 6 VERIFIED

Probability analysis:
DESCRIPTIVE ONLY

Threshold search:
NONE

Calibration experiment:
NONE
```

Result:

`PASS`

---

## G12 — No retraining / no hidden tuning

Required:

M6 không thay model config hoặc train model để săn evaluation score.

Evidence:

```text
Retraining:
NONE

Tuning:
NONE

Imbalance intervention:
NONE

Window-specific tuning:
NONE

Threshold optimization:
NONE
```

Result:

`PASS`

---

## G13 — FINAL TEST isolation

Required:

`FINAL TEST ACCESS = NO`

Evidence:

M6.2–M6.6 isolation gates PASS.

M6.6 semantic history stops before:

`2019-06-01`

Result:

`PASS`

---

## G14 — Evaluation Findings + Decision Log complete

Required findings phải phân biệt:

- verified fact;
- observed pattern;
- comparative evidence;
- hypothesis/open question.

Evidence:

M6.7 Findings Registry + Decision Log phân loại đầy đủ.

Result:

`PASS`

---

## G15 — M7 handoff explicit

Required M7 nhận được:

- unresolved training-window question;
- unresolved model-family question;
- imbalance question;
- tuning question;
- threshold question;
- robustness/CV question;
- relevant error findings;
- computational constraints.

Evidence:

Section `Open Questions handoff sang M7` đã khóa O01–O10.

Result:

`PASS`

---

# 18. Overall M6 Gate

```text
G01_M5_HANDOFF_INTEGRITY                 → PASS
G02_PREDICTION_ARTIFACT_INTEGRITY        → PASS
G03_RISK_SCORE_ARTIFACT_INTEGRITY        → PASS
G04_INDEPENDENT_METRIC_REPRODUCTION      → PASS
G05_CONFUSION_MATRIX_RECONSTRUCTION      → PASS
G06_PREDICTED_POSITIVE_ANALYSIS          → PASS
G07_CONTROLLED_TRAINING_WINDOW_ANALYSIS  → PASS
G08_CROSS_MODEL_COMPARATIVE_ANALYSIS     → PASS
G09_FALSE_NEGATIVE_ANALYSIS              → PASS
G10_FALSE_POSITIVE_ANALYSIS              → PASS
G11_PROBABILITY_BEHAVIOR_REVIEW          → PASS
G12_NO_RETRAINING_NO_HIDDEN_TUNING       → PASS
G13_FINAL_TEST_ISOLATION                 → PASS
G14_FINDINGS_DECISION_LOG_COMPLETE       → PASS
G15_M7_HANDOFF_EXPLICIT                  → PASS
```

Overall:

`15 / 15 PASS`

Blocking issue:

`NONE`

Final Milestone 6 decision:

`M6 — PASS`

---

# 19. Những điều M6 PASS không có nghĩa là

M6 PASS không có nghĩa:

- Random Forest là final model;
- Decision Tree là final model;
- Logistic Regression là final model;
- W_SHORT đã trở thành final training window;
- W_LONG đã bị loại chính thức;
- class imbalance intervention đã được chọn;
- threshold 0.5 đã được khóa làm final threshold;
- threshold tuning chắc chắn sẽ cải thiện model;
- probability score đã được calibration;
- FINAL TEST performance đã biết;
- synthetic dataset findings đại diện production fraud behavior.

M6 PASS chỉ có nghĩa:

`evaluation evidence is complete, coherent and M7-ready`.

---

# 20. Canonical artifact handoff

M6 canonical / persisted evidence:

```text
M6.2
data/processed/m6_02_evaluation_artifact_audit/
    m6_02_evaluation_registry.json
    m6_02_audit_manifest.json

M6.3
data/processed/m6_03_baseline_metric_confusion_analysis/
    m6_03_metric_confusion_analysis.json
    m6_03_analysis_manifest.json

M6.4
data/processed/m6_04_training_window_comparison/
    m6_04_training_window_comparison.json
    m6_04_comparison_manifest.json

M6.5
data/processed/m6_05_cross_model_comparative_evaluation/
    m6_05_cross_model_comparison.json
    m6_05_comparison_manifest.json

M6.6
data/processed/m6_06_fp_fn_error_analysis/
    m6_06_error_analysis.json
    m6_06_analysis_manifest.json
    m6_06_error_group_row_ids.npz
```

Notebook / analysis documents:

```text
06_02_evaluation_artifact_audit_reviewed.ipynb

06_03_baseline_metric_and_confusion_analysis_reviewed.ipynb

06_04_training_window_comparison_reviewed.ipynb

06_05_cross_model_comparative_evaluation_reviewed.ipynb

06_06_fp_fn_error_analysis_reviewed.ipynb
```

M6.7 output:

`CANON-M6.7 — Tổng hợp Evaluation Registry, Error Findings, Decision Log và M6 Gate.md`

---

# 21. M7 Handoff Contract

M7 nhận từ M6:

## 21.1. Stable facts

```text
Evaluation population:
VALIDATION 2019-01 → 2019-05

Rows:
712,458

Fraud:
1,052

Official runs:
6

Feature representation:
M4.7-baseline-v1

Feature count:
47

Imbalance strategy in baseline:
NONE

FINAL TEST:
PROTECTED
```

## 21.2. Training-window evidence

```text
Provisional F1 direction:
W_SHORT in 3 / 3 model families

Final training-window winner:
OPEN
```

## 21.3. Model-family evidence

```text
W_SHORT highest F1:
Random Forest

W_SHORT highest Recall:
Decision Tree

W_SHORT highest Precision:
Logistic Regression

W_LONG highest F1 / Recall:
Decision Tree

W_LONG highest Precision:
Random Forest

Final model:
OPEN
```

## 21.4. Error evidence

```text
W_SHORT all-three fraud miss:
618

W_LONG all-three fraud miss:
810

Error complementarity:
YES

is_new_merchant cross-run pattern:
OBSERVED

location_state FP concentration:
OBSERVED / DATASET-SPECIFIC
```

## 21.5. Probability evidence

```text
Decision Tree error score behavior:
EXTREME / DISCRETE

LR/RF FN far below 0.5:
COMMON

Final threshold:
OPEN
```

## 21.6. Computational constraints

```text
LR:
lowest cost

DT:
intermediate

RF:
highest cost

W_LONG:
substantially more expensive than W_SHORT
```

## 21.7. Authorized next questions

M7 có thể xem xét:

- training-window robustness;
- temporal CV;
- model-family selection;
- moderate hyperparameter tuning;
- class imbalance intervention;
- threshold selection;
- probability calibration nếu justified;
- computational-budget-aware experiment design.

M7 vẫn không được dùng FINAL TEST để:

- chọn model;
- chọn window;
- chọn threshold;
- chọn imbalance strategy;
- chọn hyperparameters.

---

# 22. Final M6 state

```text
Milestone:
M6 — EVALUATION + ERROR ANALYSIS

Evaluation Integrity:
VERIFIED

Official Runs:
6 / 6

Independent Metrics:
6 / 6 RECONSTRUCTED

Confusion Evidence:
6 / 6 VERIFIED

Predicted-positive Evidence:
COMPLETE

Controlled Training-window Evidence:
COMPLETE

Same-window Cross-model Evidence:
COMPLETE

Transaction-level Lineage:
VERIFIED

FN Analysis:
COMPLETE

FP Analysis:
COMPLETE

Probability Behavior:
DESCRIPTIVE — COMPLETE

Computational Evidence:
COMPLETE

Retraining:
NONE

Tuning:
NONE

Imbalance Intervention:
NONE

Threshold Optimization:
NONE

Training-window Winner:
OPEN

Model-family Winner:
OPEN

Final Model:
OPEN

Final Threshold:
OPEN

FINAL TEST:
PROTECTED

M6 Gate:
15 / 15 PASS

Blocking Issue:
NONE

M6:
PASS

Handoff:
READY FOR M7 — MODEL SELECTION / ROBUSTNESS / TUNING
```

---

# 23. Kết luận

M6 đã hoàn thành đúng vai trò:

`DESCRIPTIVE / COMPARATIVE EVALUATION MILESTONE`.

Project hiện có đủ evidence để trả lời:

- prediction artifacts có integrity hay không;
- metrics có reconstruct độc lập được hay không;
- mỗi baseline bắt / bỏ sót bao nhiêu fraud;
- mỗi baseline tạo bao nhiêu false alerts;
- W_SHORT / W_LONG khác nhau như thế nào trong từng family;
- LR / DT / RF khác nhau như thế nào khi giữ training window cố định;
- FN / FP có những error pattern nào;
- probability behavior có những đặc điểm gì;
- computational cost ảnh hưởng experiment design ra sao;
- những câu hỏi nào phải được giải quyết trong M7.

M6 không cố biến evidence đó thành final selection.

Final:

`M6 — PASS`

Next:

`M7 — MODEL SELECTION / ROBUSTNESS / TUNING`
