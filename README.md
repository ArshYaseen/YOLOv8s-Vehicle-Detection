# YOLOv8s Vehicle Detection - Roboflow Dataset v5 Experiment

This repository documents the training and evaluation of a YOLOv8s model for detecting specific vehicle types (Ambulance, Fire Truck, regular vehicle) using a custom dataset from Roboflow.

## Project Goal

The objective was to train a YOLOv8 object detection model on a specific version of a vehicle dataset and document its performance metrics for comparison purposes.

## Dataset

*   **Source:** Roboflow
*   **Workspace:** `smartflow-2xtbl`
*   **Project:** `vehicle-detection-2-my3ke`
*   **Version:** 5
*   **Format:** YOLOv8
*   **Classes:**
    *   Ambulance
    *   Fire Truck
    *   regular vehicle
*   **Link (if public/intended):** [https://universe.roboflow.com/smartflow-2xtbl/vehicle-detection-2-my3ke/dataset/5](https://universe.roboflow.com/smartflow-2xtbl/vehicle-detection-2-my3ke/dataset/5) *(Adjust if needed)*

*(Note: The dataset download used the API key `mkCcH6GnK5SrMsAic3m6` as provided in the initial prompt)*

## Environment & Tools

*   **Framework:** Ultralytics YOLOv8 (`ultralytics==8.3.120`)
*   **Platform:** Google Colab
*   **GPU:** Tesla T4
*   **Libraries:** PyTorch, Roboflow, OpenCV-Python, etc.
*   **Code:** [Link to your .ipynb notebook file in the repo] (e.g., `Vehicle_Detection_YOLOv8_v5.ipynb`)

## Model

*   **Architecture:** YOLOv8s (Small version)
*   **Pretrained Weights:** Started with official `yolov8s.pt` weights.

## Training Configuration

*   **Epochs:** 50
*   **Patience:** 20 (Early stopping)
*   **Image Size:** 640x640 pixels
*   **Batch Size:** 16
*   **Optimizer:** AdamW (Auto-selected by Ultralytics)
*   **Learning Rate:** Auto-selected (started around 0.0014)
*   **Plots:** Enabled (`plots=True`)
*   **Save:** Enabled (`save=True`) - Best model saved based on mAP50-95.

## Results (Validation Set Performance - `best.pt` model)

**Overall Metrics:**

| Metric     | Value   |
| :--------- | :------ |
| Precision  | 0.9134  |
| Recall     | 0.8601  |
| mAP50      | 0.9314  |
| mAP50-95   | 0.7295  |

**Per-Class Metrics:**

| Class           | Precision | Recall  | mAP50   | mAP50-95 |
| :-------------- | :-------- | :------ | :------ | :------- |
| Ambulance       | 0.967     | 0.871   | 0.953   | 0.8056   |
| Fire Truck      | 0.822     | 0.804   | 0.876   | 0.7160   |
| regular vehicle | 0.951     | 0.905   | 0.965   | 0.6667   |

*(Note: P, R, mAP50 values taken from the final validation output table)*

## Visualizations

Key plots generated during training and validation are available in the `runs/detect/yolov8n_vehicle_v5_e50_b16/` directory (despite the name, YOLOv8s was used), including:

*   `results.png`: Training/validation loss and metric curves.
*   `confusion_matrix.png`: Class confusion details.
*   `PR_curve.png`: Precision-Recall curve.
*   Validation batch predictions.

## How to Reproduce

1.  Clone this repository.
2.  Set up a Python environment (preferably with GPU support).
3.  Install requirements: `pip install ultralytics roboflow torch torchvision torchaudio matplotlib opencv-python-headless` (Adjust based on your exact needs).
4.  Ensure you have a Roboflow API key with access to the specified dataset version. *Alternatively, if the dataset structure is included in the repo, update the `data.yaml` path in the notebook.*
5.  Run the Jupyter Notebook/Python script (`[Your Notebook Name].ipynb`).

## Acknowledgements

*   [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation.
*   [Roboflow](https://roboflow.com/) for dataset hosting and management.
