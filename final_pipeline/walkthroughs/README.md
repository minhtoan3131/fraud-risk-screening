# Clean walkthrough layer

Các file này là lớp hướng dẫn chạy trên canonical `final_pipeline`.

- Không sao chép parser/history/preprocessing/model-scoring logic.
- Không phụ thuộc runtime vào `research/`.
- Không ghi đè official artifacts.
- Không tự thay threshold/model/feature semantics.

Thứ tự:

1. `01_dataset_walkthrough.py`
2. `02_feature_pipeline_walkthrough.py`
3. `03_preprocessing_walkthrough.py`
4. `04_model_training_walkthrough.py`
5. `05_inference_walkthrough.py`

`01` mặc định chỉ kiểm tra nhanh; thêm `--full` mới quét SHA-256 và row count toàn bộ dataset.

`04` chỉ giải thích và smoke `train_model.py --help`; không gọi `fit`. Training thật phải được kích hoạt rõ qua operational CLI.
