# Evaluation: YOLO26s on the leak-free split

Model: `yolo_model/yolo26s_clean/weights/best.pt` (YOLO26s, fine-tuned from COCO weights, best epoch 111 of 150)
Software: Ultralytics 8.4.183, PyTorch 2.11, Tesla T4 (Google Colab)
Data: `clean_split` (see `reports/split_report.csv`); 389 train / 111 valid / 55 test images

Metrics are computed by `yolo detect val` at the default evaluation settings unless stated.

## 1. Test set (final score; not used for any decision)

55 images (13 without marks), 61 marks.

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **all** | 55 | 61 | **0.840** | **0.772** | **0.794** | **0.556** |
| mix | 11 | 11 | 0.665 | 0.818 | 0.764 | 0.588 |
| signature | 24 | 33 | 0.867 | 0.793 | 0.875 | 0.478 |
| stamp | 16 | 17 | 0.987 | 0.706 | 0.744 | 0.602 |

Inference on T4: 16.4 ms per image (plus 6.1 ms preprocessing, 4.6 ms postprocessing).

## 2. Validation set (used to pick the best epoch and the confidence threshold)

111 images (33 without marks), 122 marks.

**Best epoch during training (epoch 111):**

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.791 | 0.898 | 0.914 | 0.687 |

**Class-agnostic (`single_cls=True`, any mark regardless of label):**

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.884 | 0.934 | 0.936 | 0.660 |

**Per class at `conf=0.25`** (operating-point check; mAP is lower than at the default because low-confidence predictions are cut off):

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| all | 111 | 122 | 0.791 | 0.895 | 0.860 | 0.651 |
| mix | 23 | 25 | 0.701 | 0.920 | 0.870 | 0.662 |
| signature | 52 | 66 | 0.934 | 0.864 | 0.853 | 0.569 |
| stamp | 28 | 31 | 0.737 | 0.902 | 0.857 | 0.722 |

## 3. Confidence threshold

Best mean F1 over all classes: **0.83 at confidence 0.295** (`yolo_model/yolo26s_clean/BoxF1_curve.png`). F1 is nearly flat between about 0.25 and 0.7, so the API uses **0.3**.

## 4. Error analysis (validation confusion matrix)

`yolo_model/yolo26s_clean/confusion_matrix.png`

| True class | Correct | Labelled mix | Labelled signature | Labelled stamp | Missed |
|---|---|---|---|---|---|
| mix (25) | 20 | – | 3 | 2 | 0 |
| signature (66) | 47 | 18 | – | 0 | 1 |
| stamp (31) | 23 | 4 | 4 | – | 0 |

- Only **1 of 122** marks was missed entirely. Most errors are **wrong labels**, chiefly signatures labelled "mix" (signature over stamp).
- The matrix also counts many background detections; these come from low-confidence predictions and are largely removed by the 0.3 threshold.

## 5. Comparison with the original model

| Model | Training split | Evaluated on | mAP@50 |
|---|---|---|---|
| YOLOv8s, 800 px, 250 epochs | leaky | leaky validation | 0.88 (inflated by duplicates) |
| YOLO26s, 640 px, 150 epochs | clean | clean validation | 0.91 |
| YOLO26s, 640 px, 150 epochs | clean | **clean test** | **0.79** |

The two models differ in both architecture and data split, so these rows are not a like-for-like comparison. The original model cannot simply be evaluated on the new test set either, because its training set contained copies of some new test images.

## 6. Caveats

- Small test set: one stamp is about 6 recall points. Treat differences under ~5 points as noise.
- All images were resized to 640×640 (stretched) with adaptive contrast equalisation by Roboflow before training.
- Real certificates in the dataset are private and not included in this repository.
