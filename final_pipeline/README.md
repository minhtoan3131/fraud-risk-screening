# Final Fraud-Risk Screening Pipeline

`final_pipeline/` chứa implementation Machine Learning cuối dùng cho transaction
fraud-risk screening.

## Thành phần

- `src/fraud_screening/`: canonical Python package;
- `artifacts/official/`: frozen official model và preprocessing state;
- `scripts/`: operational commands;
- `walkthroughs/`: các walkthrough sạch từ dataset đến inference;
- `tests/`: unit/regression tests;
- `data/`: hướng dẫn raw-dataset boundary.

## Runtime flow

Input transaction
→ validation
→ strict-prior card history
→ semantic features
→ frozen preprocessing
→ frozen Random Forest
→ positive-class risk score
→ strict threshold
→ screening result

## Cài đặt local

    python -m pip install -e final_pipeline

## Kiểm thử

    PYTHONPATH=final_pipeline/src \
    python -m unittest discover -s final_pipeline/tests -v

## Raw dataset

Raw dataset không được bundle vào package và không được đọc ngầm từ khu vực
nghiên cứu. Dataset replay luôn yêu cầu một đường dẫn external/local rõ ràng.

## Artifact policy

Official artifacts là read-only đối với runtime và rebuild training.
Rebuild outputs phải nằm ở khu vực output riêng và không ghi đè official artifacts.

## Interpretation

`risk_score` là positive-class model score phục vụ screening.

Không mô tả score này là calibrated real-world fraud probability và không coi
screening output là kết luận gian lận cuối cùng.
