# Tooth Numbering (FDI) – YOLO Training

Overview

End-to-end pipeline for training a YOLO detector to localize and identify teeth using the FDI numbering system on panoramic dental images. The repository/script prepares the dataset, trains with pretrained weights at 640, evaluates on val/test, saves metrics and confusion matrices, generates sample predictions, and optionally post-processes predictions for anatomical consistency.

# Environment

Python 3.8+ recommended (CUDA optional).

Install core packages:

Option A: pip quick start

pip install -U ultralytics opencv-python pyyaml tqdm scikit-learn pandas numpy matplotlib

Option B: conda example (GPU optional)

conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics

Notes:

For headless/servers, opencv-python-headless can be substituted.

Ultralytics provides both Python API and CLI. This project uses the Python API; equivalent CLI examples are linked below.

# Dataset

Input archive: ToothNumber_TaskDataset.zip containing:

images/ with panoramic radiographs

labels/ with YOLO-format annotations using FDI classes (0–31) per provided mapping.

The script auto-extracts to /content/dataset_raw, finds images, pairs labels, and creates train/val/test splits (80/10/10) under /content/dataset_prepared.

# Quick Start (Python API)

Place ToothNumber_TaskDataset.zip at /content/ToothNumber_TaskDataset.zip (or update ZIP_PATH in the script).

Run the provided single script (dataset prep → train → eval → predictions → post-processing).

# Key defaults:

Model: yolov8s.pt (pretrained)

Image size: 640

Epochs: 10 (adjustable)

Batch: 16 (adjustable based on GPU memory)

Equivalent CLI (optional)

If preferring CLI, an equivalent training command is:

yolo detect train model=yolov8s.pt data=/content/dataset_prepared/data.yaml epochs=10 imgsz=640

Validation/Test: yolo detect val model=PATH/TO/best.pt data=/content/dataset_prepared/data.yaml

Predictions: yolo predict model=PATH/TO/best.pt source=/content/dataset_prepared/images/test imgsz=640 conf=0.25 iou=0.7

Refer to Ultralytics training docs for argument variants and devices.

# Outputs

After running the full pipeline:

# Training artifacts:

/content/runs/train with logs, plots, and weights (best.pt under weights/).

# Evaluation artifacts:

Metrics CSV/JSON and plots copied to:

/content/runs/submission_artifacts/metrics_val.csv, metrics_test.csv

/content/runs/submission_artifacts/val_plots, test_plots (confusion_matrix.png, PR/F1 curves, etc.).

# Sample predictions:

Raw annotated images: /content/runs/predict_vis

YOLO .txt predictions: /content/runs/predict_vis/labels (when save_txt=True).

Optional post-processing:

Post-processed visuals: /content/runs/postproc_vis

Post-processed labels: /content/runs/postproc_labels

# Manifest:

/content/runs/submission_artifacts/manifest.json summarizing paths and configuration.

Training/Evaluation Snippet (Python)

Minimal Python example using Ultralytics:

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(data="/content/dataset_prepared/data.yaml", imgsz=640, epochs=10)

metrics = model.val(data="/content/dataset_prepared/data.yaml", split="val")

preds = model.predict(source="/content/dataset_prepared/images/test", imgsz=640, conf=0.25, iou=0.7)

See Ultralytics docs for full argument references and device configuration.

# Tips

Increase epochs for higher mAP if time permits.

Monitor precision/recall, mAP@50, mAP@50–95; confusion matrices help identify class confusions.

For reproducibility, set seeds and log versions.
