# Complete Step-by-Step Guide: Self-Driving Car Simulation Project

This guide walks you through every step from initial setup to testing your autonomous car in the simulator.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Installing Dependencies](#2-installing-dependencies)
3. [Downloading the Simulator](#3-downloading-the-simulator)
4. [Collecting Training Data](#4-collecting-training-data)
5. [Visualizing & Exploring Data](#5-visualizing--exploring-data)
6. [Training the Model](#6-training-the-model)
7. [Testing in the Simulator](#7-testing-in-the-simulator)
8. [Expected Outputs](#8-expected-outputs)
9. [Troubleshooting](#9-troubleshooting)
10. [Git Workflow](#10-git-workflow)

---

## 1. Environment Setup

### 1.1 Prerequisites

- **Python 3.8–3.10** (tested; 3.11+ may have TensorFlow compatibility issues)
- **Git** installed
- **Udacity Self-Driving Car Simulator** (see Step 3)
- **GPU (recommended)**: NVIDIA GPU with CUDA support for faster training

### 1.2 Create a Virtual Environment

```bash
# Navigate to the project directory
cd final_project

# Create a virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

---

## 2. Installing Dependencies

```bash
# Make sure your virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import socketio; print(f'SocketIO: {socketio.__version__}')"
```

**Expected output:**
```
TensorFlow: 2.x.x
GPU available: True (or False if CPU only)
OpenCV: 4.x.x
SocketIO: 5.x.x
```

---

## 3. Downloading the Simulator

### Windows
1. Download from: [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim/releases)
2. Extract the ZIP file
3. Run `beta_simulator.exe`

### Mac
1. Search for "Udacity self-driving car simulator Mac" or download from the same GitHub releases page
2. Download the Mac version
3. Run the application

### Linux
1. Download the Linux version from the GitHub releases
2. Make it executable: `chmod +x beta_simulator.x86_64`
3. Run: `./beta_simulator.x86_64`

### Simulator Settings
- **Screen resolution**: 840 × 480
- **Graphics quality**: Fastest (for performance)
- **Windowed**: Checked
- Click **"Play!"**

---

## 4. Collecting Training Data

### 4.1 Launch in Training Mode

1. Open the simulator
2. Configure settings (840x480, Fastest, Windowed)
3. Click **"Play!"**
4. Select **"TRAINING MODE"**

### 4.2 Set Up Recording

1. Click the **"Recording"** button (red circle) at the top of the simulator
2. Select a folder path — create a folder called `data` inside your project directory
3. Click **"Select"**

### 4.3 Drive the Car

- **Use the mouse** for smooth, continuous steering (better than keyboard)
- **Keyboard controls**: Arrow keys or WASD
- Drive the car along the **left track** (simpler track)
- Drive carefully, staying in the center of the lane

### 4.4 Data Collection Tips

- **Drive 5 laps forward** along the track
- **Drive 5 laps in reverse** (opposite direction) for balanced data
- **Recovery driving**: Intentionally steer toward the edge, then correct back to center — this teaches the model how to recover
- Keep a **smooth, steady speed**
- Click **"Recording"** again to stop recording

### 4.5 Verify Data

After recording, check your `data/` folder:

```
data/
├── IMG/                      # Should contain thousands of .jpg images
│   ├── center_2024_...jpg   # Center camera
│   ├── left_2024_...jpg     # Left camera
│   └── right_2024_...jpg    # Right camera
└── driving_log.csv           # CSV with columns: center,left,right,steering,throttle,brake,speed
```

**Quick check:**
```bash
# Count images
ls data/IMG/ | wc -l

# View first few lines of CSV
head -5 data/driving_log.csv
```

You should have at least **5,000–10,000+ images** for good results.

---

## 5. Visualizing & Exploring Data

```bash
python visualize_data.py --data_dir ./data
```

**This will:**
- Show the steering angle distribution (histogram)
- Display sample images from the dataset
- Show the preprocessing pipeline step by step
- Show examples of data augmentation
- Save visualization plots as PNG files

**Expected output:**
```
[INFO] Loaded 8000 samples from ./data/driving_log.csv

Dataset shape: (8000, 7)
Steering angle range: [-1.000, 1.000]
Mean steering angle: 0.005

[INFO] Balanced data: 8000 -> 6500 samples
[INFO] All visualizations saved!
```

---

## 6. Training the Model

### 6.1 Basic Training

```bash
python train.py --data_dir ./data
```

### 6.2 Custom Training Parameters

```bash
python train.py \
    --data_dir ./data \
    --model_output model.h5 \
    --epochs 30 \
    --batch_size 100 \
    --learning_rate 0.001 \
    --samples_per_bin 400 \
    --steps_per_epoch 300 \
    --validation_steps 200
```

### 6.3 Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Path to data folder with driving_log.csv and IMG/ |
| `--model_output` | `model.h5` | Where to save the trained model |
| `--epochs` | `30` | Number of training passes through the data |
| `--batch_size` | `100` | Number of samples per batch |
| `--learning_rate` | `0.001` | Adam optimizer learning rate |
| `--samples_per_bin` | `400` | Max samples per steering angle bin |
| `--steps_per_epoch` | `300` | Batches per training epoch |
| `--validation_steps` | `200` | Batches per validation epoch |

### 6.4 Expected Training Output

```
============================================================
STEP 1: Loading driving data
============================================================
[INFO] Loaded 8000 samples from ./data/driving_log.csv

============================================================
STEP 2: Balancing data distribution
============================================================
[INFO] Balanced data: 8000 -> 6500 samples

============================================================
STEP 5: Building NVIDIA CNN model
============================================================
[INFO] NVIDIA Self-Driving Car Model Summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 31, 98, 24)        1824
 conv2d_1 (Conv2D)           (None, 14, 47, 36)        21636
 conv2d_2 (Conv2D)           (None, 5, 22, 48)         43248
 conv2d_3 (Conv2D)           (None, 3, 20, 64)         27712
 conv2d_4 (Conv2D)           (None, 1, 18, 64)         36928
 flatten (Flatten)           (None, 1152)              0
 dense (Dense)               (None, 1164)              1342092
 dropout (Dropout)           (None, 1164)              0
 dense_1 (Dense)             (None, 100)               116500
 dropout_1 (Dropout)         (None, 100)               0
 dense_2 (Dense)             (None, 50)                5050
 dropout_2 (Dropout)         (None, 50)                0
 dense_3 (Dense)             (None, 10)                510
 dense_4 (Dense)             (None, 1)                 11
=================================================================
Total params: 1,595,511
...

Epoch 1/30
300/300 - loss: 0.0832 - val_loss: 0.0456
Epoch 2/30
300/300 - loss: 0.0412 - val_loss: 0.0389
...
Epoch 30/30
300/300 - loss: 0.0178 - val_loss: 0.0201

[INFO] Model saved to model.h5
[INFO] Training loss plot saved to training_loss.png

TRAINING COMPLETE!
```

### 6.5 Evaluating Training

After training, check `training_loss.png`:
- **Both curves should decrease** over epochs
- **Validation loss should be close to training loss** (not diverging)
- If validation loss increases while training loss decreases → **overfitting**
  - Try: more augmentation, higher dropout, less epochs, more data

---

## 7. Testing in the Simulator

### 7.1 Run the Test Script

```bash
python TestSimulation.py --model model.h5
```

**Expected output:**
```
============================================================
SELF-DRIVING CAR - AUTONOMOUS MODE
============================================================
[INFO] Loading model from: model.h5
[INFO] Model loaded successfully!
[INFO] Max speed: 25

[INFO] Waiting for simulator connection...
[INFO] Please launch the simulator and select 'Autonomous Mode'
============================================================
```

### 7.2 Launch the Simulator

1. Open the simulator with the **same settings** used for data collection
2. Click **"Play!"**
3. Select **"AUTONOMOUS MODE"**

### 7.3 Observe the Results

You should see:
- The terminal printing steering angles and speeds in real-time
- The car driving itself along the track!

```
[INFO] Simulator connected!
Steering: 0.0123 | Throttle: 0.8500 | Speed: 12.34
Steering: -0.0456 | Throttle: 0.7200 | Speed: 15.67
Steering: 0.1234 | Throttle: 0.6100 | Speed: 18.90
...
```

### 7.4 Recording Results

- Use screen recording software (OBS Studio, Windows Game Bar, or macOS screen recording) to capture the autonomous driving session
- This recording is a required deliverable

---

## 8. Expected Outputs

| File | Description |
|------|-------------|
| `model.h5` | Trained Keras model (≈19 MB) |
| `training_loss.png` | Training/validation loss curves |
| `histogram_before.png` | Steering distribution before balancing |
| `histogram_after.png` | Steering distribution after balancing |
| `sample_images.png` | Sample images from dataset |
| `preprocessing_pipeline.png` | Visualization of preprocessing steps |
| `augmentation_examples.png` | Examples of data augmentation |

---

## 9. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | TF not installed | `pip install tensorflow` |
| `No images found in IMG folder` | Wrong data path | Check `--data_dir` points to folder with `IMG/` and `driving_log.csv` |
| `Model predicts constant steering` | Not enough training data | Collect more data (10+ laps) |
| Car drives off road immediately | Poor data quality | Re-collect data with smoother driving |
| `OSError: [Errno 98] Address already in use` | Port 4567 in use | Kill existing process: `lsof -i :4567` then `kill <PID>` |
| Simulator not connecting | Script not running | Ensure `TestSimulation.py` is running BEFORE launching simulator |
| `CUDA out of memory` | GPU memory full | Reduce `--batch_size` to 50 or 32 |
| High validation loss | Overfitting | Add more data, increase augmentation |
| Car wobbles/swerves | Noisy data collection | Use mouse instead of keyboard; drive smoothly |
| `driving_log.csv` has wrong paths | Simulator saved absolute paths | The code auto-fixes this with `fix_image_paths()` |

### Tips for Better Results

1. **More data is better** – aim for 10,000+ images
2. **Drive in both directions** – prevents left/right bias
3. **Include recovery data** – steer from edges back to center
4. **Use mouse for steering** – smoother than keyboard
5. **Balance your data** – the balancing step prevents "always go straight" bias
6. **Monitor loss curves** – stop training when validation loss plateaus

---

## 10. Git Workflow

### Initial Setup

```bash
cd final_project
git init
git add .
git commit -m "Initial commit: project structure and core scripts"
```

### Recommended Commit History

```bash
# After creating project structure
git commit -m "Add project structure: utils, model, train, test scripts"

# After collecting data
git commit -m "Add data collection notes and visualization script"

# After successful training
git commit -m "Train model with 30 epochs, validation loss: 0.02"

# After testing
git commit -m "Verify autonomous driving in simulator - working"

# Documentation
git commit -m "Add README and complete guide documentation"
```

### Important: .gitignore

The `.gitignore` file excludes:
- `data/` folder (images are too large for Git)
- `*.h5` model files (too large)
- `*.png` output plots
- Python cache files

If you need to share the model, use Google Drive or a file sharing service.

---

## Summary Workflow

```
1. Setup environment     →  python -m venv venv && pip install -r requirements.txt
2. Collect data          →  Use simulator in Training Mode, save to ./data/
3. Visualize data        →  python visualize_data.py --data_dir ./data
4. Train model           →  python train.py --data_dir ./data --epochs 30
5. Test model            →  python TestSimulation.py --model model.h5
6. Record results        →  Screen record the autonomous driving
7. Commit to git         →  git add . && git commit -m "Final submission"
```

Good luck! 🚗💨
