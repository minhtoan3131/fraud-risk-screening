# Software Testing Charter

## Scope

Kiểm thử tập trung vào canonical final pipeline và user-facing application.
Không mở lại model selection, feature selection, threshold selection hoặc final
evaluation.

## Test layers

1. Final-pipeline unit/regression
   - parsing/schema;
   - transaction/history features;
   - preprocessing;
   - artifact loading;
   - threshold semantics;
   - rebuild isolation;
   - interactive inference.

2. Raw application integration
   - raw transaction input;
   - cold-start;
   - explicit strict-prior history;
   - invalid input/history;
   - result parity with direct final-pipeline inference.

3. Frozen FINAL TEST ground-truth regression
   - use audited FINAL TEST encoded matrix;
   - use frozen M8.3 row ids, labels, risk scores and predictions;
   - select representative TP, TN, FP and FN rows;
   - score those same encoded rows using the official packaged model;
   - require prediction parity and numerical risk-score parity.

4. Defense-oriented cases
   - one frozen FINAL TEST fraud caught by model (TP);
   - one frozen FINAL TEST non-fraud correctly screened negative (TN);
   - one frozen false positive;
   - one frozen false negative;
   - application cold-start;
   - known card with explicit history/new merchant;
   - invalid Amount;
   - invalid same-timestamp history.

## Boundaries

Testing may read research evidence for audit comparison only.

Runtime dependencies remain:

application -> final_pipeline

Forbidden runtime dependencies:

application -> research
final_pipeline -> research
