# Runtime Installation and Verification

## Supported runtime

Package metadata yêu cầu Python `>=3.11`.

Exact environment đã dùng cho release smoke được ghi trong:

    runtime_environment.json

Các library runtime exact-version được ghi trong:

    requirements.txt

## Install

Từ project root:

    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e final_pipeline
    python -m pip install --no-deps -e application

## Verify package imports

    python -c "import fraud_screening, fraud_screening_app; print('IMPORT PASS')"

## Verify application

    fraud-screening-app --version-info

    fraud-screening-app screen \
      --transaction-json application/examples/current_transaction.json

## Selected smoke tests

    python -m unittest discover \
      -s final_pipeline/tests \
      -p 'test_inference_service.py' -v

    python -m unittest discover \
      -s application/tests \
      -p 'test_acceptance.py' -v

Không cần rerun toàn bộ research pipeline 24 triệu dòng để chạy application.
