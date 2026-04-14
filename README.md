# Self-Driving Car Simulation Project Using CNN

**CVI620 Final Project – Winter 2026**

## Overview

This project implements a neural network model to autonomously control a self-driving car in a simulator. The model predicts the appropriate steering angle using images captured from the car’s front camera, enabling the vehicle to stay on the road.

The project uses the **NVIDIA End-to-End Deep Learning Architecture** for self-driving cars, trained on data collected from the **Udacity Self-Driving Car Simulator**.

## Project Structure

```
final_project/
├── README.md                 # This file - project overview
├── COMPLETE_GUIDE.md         # Step-by-step guide from setup to testing
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── utils.py                  # Data loading, preprocessing, augmentation, batch generator
├── model.py                  # NVIDIA CNN model definition
├── train.py                  # Training script
├── TestSimulation.py         # Inference / autonomous driving script
├── visualize_data.py         # Data visualization & exploration utility
└── data/                     # (You create this) Collected driving data
    ├── IMG/                  # Camera images from simulator
    └── driving_log.csv       # Driving log with steering angles
```

## Architecture

The model follows the **NVIDIA End-to-End Learning** architecture:

| Layer | Type | Filters/Neurons | Kernel | Stride |
|-------|------|-----------------|--------|--------|
| 1 | Conv2D | 24 | 5×5 | 2×2 |
| 2 | Conv2D | 36 | 5×5 | 2×2 |
| 3 | Conv2D | 48 | 5×5 | 2×2 |
| 4 | Conv2D | 64 | 3×3 | 1×1 |
| 5 | Conv2D | 64 | 3×3 | 1×1 |
| 6 | Flatten | - | - | - |
| 7 | Dense | 1164 | - | - |
| 8 | Dense | 100 | - | - |
| 9 | Dense | 50 | - | - |
| 10 | Dense | 10 | - | - |
| 11 | Dense (Output) | 1 | - | - |

**Input**: 66 × 200 × 3 (YUV image)  
**Output**: 1 value (predicted steering angle)

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Collect data using the Udacity Simulator (Training Mode)
#    Save to ./data/ directory

# 4. Train the model
python train.py --data_dir ./data --epochs 30

# 5. Test in simulator
python TestSimulation.py --model model.h5
#    Then launch simulator → Autonomous Mode
```

For detailed instructions, see **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)**.

## Preprocessing Pipeline

1. **Crop** – Extract road region (rows 60–135) to remove sky and car hood
2. **YUV Conversion** – Convert from BGR to YUV color space
3. **Gaussian Blur** – Reduce noise with a 3×3 kernel
4. **Resize** – Scale to 200×66 pixels (NVIDIA model input size)
5. **Normalize** – Scale pixel values to [0, 1]

## Data Augmentation (Training Only)

- **Flipping** – Horizontal flip with negated steering angle
- **Brightness** – Random brightness adjustment
- **Zoom** – Random crop and resize
- **Pan** – Random horizontal/vertical translation with steering correction

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Imbalanced data (bias toward straight) | Histogram-based balancing to cap samples per bin |
| Overfitting | Dropout layers (0.5, 0.3, 0.2) + data augmentation |
| Car drifting off road | Driving in both directions during data collection |
| Smooth steering | Using mouse for data collection instead of keyboard |

## Team

- CVI620 Winter 2026 Project Team
