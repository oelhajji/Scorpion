# 🦂 Scorpion Detection using YOLO (Ultralytics)

This repository contains a YOLO-based object detection project for detecting scorpions in videos using the Ultralytics YOLO framework.

> [!WARNING]  
> **Dataset Not Included:** The dataset is not included in this repository and must be added manually before training.

---

## Repository Structure

```
Scorpion/
├── dataset/                # (Add your dataset here)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
├── data.yaml               # Dataset configuration
├── train.py                # Training script
├── demo.py                 # Inference/Detection script
└── best.pt                 # Trained model weights
```
## Dataset Setup
Place your dataset inside the following directory:
`Scorpion/dataset/`

### Download
You can obtain the labeled scorpion dataset from the following link:
* **[Download Dataset Here]https://universe.roboflow.com/truong-do/worm-xyvu0**

The dataset must follow the **YOLO format**:
* Images stored in `images/`
* Labels stored in `labels/`
* One `.txt` label file per image (normalized `class x_center y_center width height`)

---

## Dataset Configuration
The dataset configuration is defined in `data.yaml`:

| Key | Description |
| :--- | :--- |
| **path** | Root dataset directory |
| **train** | Training images path |
| **val** | Validation images path |
| **nc** | Number of classes (1) |
| **names** | Class names (`['scorpion']`) |

---

## Training the Model
To start the training process, run:

```bash
python train.py
```
## Running Inference on Video

1. Place your input video in the project root and rename it to `video.mp4`.
2. Run the demo script:

```bash
python demo.py
```
## 🛠 Requirements

* **Python:** 3.8+
* **Frameworks:** `ultralytics`, `opencv-python`, `numpy`
* **Hardware:** CUDA-enabled GPU (recommended for real-time performance)

### Installation

To install the necessary dependencies, run the following command in your terminal:

```bash
pip install ultralytics opencv-python numpy
