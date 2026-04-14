"""
utils.py - Utility functions for data loading, preprocessing, and augmentation.

This module provides all helper functions used across the self-driving car
project including image preprocessing, data augmentation, histogram balancing,
and batch generation for training.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random

# =============================================================================
# DATA LOADING
# =============================================================================

def load_driving_data(data_dir):
    """
    Load driving log CSV and return a DataFrame with image paths and steering angles.

    The Udacity simulator saves a driving_log.csv with columns:
    Center, Left, Right, Steering, Throttle, Brake, Speed

    We only use Center image paths and Steering values for this project.

    Args:
        data_dir (str): Path to the directory containing driving_log.csv and IMG folder.

    Returns:
        pd.DataFrame: DataFrame with columns ['center', 'steering'].
    """
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    csv_path = os.path.join(data_dir, 'driving_log.csv')

    df = pd.read_csv(csv_path, names=columns, header=None)

    # Strip whitespace from path strings
    df['center'] = df['center'].str.strip()
    df['left'] = df['left'].str.strip()
    df['right'] = df['right'].str.strip()

    # Convert steering to float
    df['steering'] = df['steering'].astype(float)

    print(f"[INFO] Loaded {len(df)} samples from {csv_path}")
    return df


def fix_image_paths(df, data_dir):
    """
    Fix image paths in the DataFrame to point to actual file locations.
    The simulator may save absolute paths; we convert them to relative paths
    under data_dir/IMG/.

    Args:
        df (pd.DataFrame): DataFrame with 'center' column.
        data_dir (str): Root data directory containing the IMG folder.

    Returns:
        pd.DataFrame: DataFrame with corrected paths.
    """
    def fix_path(path):
        # Extract just the filename from potential absolute paths
        filename = os.path.basename(path)
        return os.path.join(data_dir, 'IMG', filename)

    df['center'] = df['center'].apply(fix_path)
    df['left'] = df['left'].apply(fix_path)
    df['right'] = df['right'].apply(fix_path)
    return df


# =============================================================================
# DATA BALANCING
# =============================================================================

def plot_steering_histogram(steering_values, title="Steering Angle Distribution", bins=25,
                            save_path=None):
    """
    Plot a histogram of steering angle values to check data balance.

    A balanced dataset should have a roughly uniform distribution around 0.

    Args:
        steering_values (array-like): Steering angle values.
        title (str): Plot title.
        bins (int): Number of histogram bins.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(steering_values, bins=bins, color='steelblue', edgecolor='black')
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Histogram saved to {save_path}")
    plt.show()


def balance_data(df, bins=25, samples_per_bin=400):
    """
    Balance the dataset by limiting the number of samples per steering angle bin.

    Prevents the model from being biased toward going straight (steering ≈ 0).

    Args:
        df (pd.DataFrame): DataFrame with 'steering' column.
        bins (int): Number of bins for the histogram.
        samples_per_bin (int): Maximum number of samples allowed per bin.

    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    hist, bin_edges = np.histogram(df['steering'], bins=bins)
    remove_indices = []

    for i in range(bins):
        bin_data = df[(df['steering'] >= bin_edges[i]) & (df['steering'] < bin_edges[i + 1])]
        if len(bin_data) > samples_per_bin:
            remove_indices.extend(
                bin_data.sample(n=len(bin_data) - samples_per_bin, random_state=42).index.tolist()
            )

    df_balanced = df.drop(remove_indices).reset_index(drop=True)
    print(f"[INFO] Balanced data: {len(df)} -> {len(df_balanced)} samples")
    return df_balanced


# =============================================================================
# IMAGE PREPROCESSING (Applied to ALL images — training and testing)
# =============================================================================

def preprocess_image(img):
    """
    Preprocess an image for the NVIDIA self-driving car model.

    Steps:
        1. Crop the road region (rows 60 to 135 out of 160)
        2. Convert BGR to YUV color space
        3. Apply Gaussian blur
        4. Resize to 200x66 (NVIDIA model input size)
        5. Normalize pixel values to [0, 1]

    Args:
        img (np.ndarray): Input BGR image from the simulator (160x320x3).

    Returns:
        np.ndarray: Preprocessed image (66x200x3) with values in [0, 1].
    """
    # 1. Crop: keep only the road area (remove sky and car hood)
    img = img[60:135, :, :]

    # 2. Convert to YUV color space (used by NVIDIA model)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 3. Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4. Resize to NVIDIA model input dimensions: 200 wide x 66 tall
    img = cv2.resize(img, (200, 66))

    # 5. Normalize to [0, 1]
    img = img / 255.0

    return img


# =============================================================================
# DATA AUGMENTATION (Applied ONLY to training data, randomly)
# =============================================================================

def augment_zoom(image, zoom_range=(1.0, 1.3)):
    """
    Apply a random zoom to the image by cropping and resizing.

    Args:
        image (np.ndarray): Input image.
        zoom_range (tuple): Min and max zoom factors.

    Returns:
        np.ndarray: Zoomed image.
    """
    zoom = random.uniform(*zoom_range)
    h, w = image.shape[:2]
    zh, zw = int(h / zoom), int(w / zoom)
    top = random.randint(0, h - zh)
    left = random.randint(0, w - zw)
    cropped = image[top:top + zh, left:left + zw]
    return cv2.resize(cropped, (w, h))


def augment_pan(image, steering, pan_range=100):
    """
    Apply a random horizontal/vertical pan (translation) to the image.
    Adjusts steering angle proportionally for horizontal shifts.

    Args:
        image (np.ndarray): Input image.
        steering (float): Current steering angle.
        pan_range (int): Maximum pixel shift.

    Returns:
        tuple: (panned image, adjusted steering angle).
    """
    pan_x = random.randint(-pan_range, pan_range)
    pan_y = random.randint(-pan_range, pan_range)
    M = np.float32([[1, 0, pan_x], [0, 1, pan_y]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # Adjust steering angle based on horizontal shift
    steering += pan_x * 0.002
    return image, steering


def augment_brightness(image):
    """
    Randomly adjust the brightness of the image.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Brightness-adjusted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Random brightness factor between 0.2 and 1.2
    brightness = random.uniform(0.2, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_flip(image, steering):
    """
    Flip the image horizontally and reverse the steering angle.

    Args:
        image (np.ndarray): Input image.
        steering (float): Current steering angle.

    Returns:
        tuple: (flipped image, negated steering angle).
    """
    image = cv2.flip(image, 1)
    steering = -steering
    return image, steering


def random_augment(image, steering):
    """
    Apply a random combination of augmentation techniques to an image.

    Each augmentation is applied with 50% probability.
    This ensures diversity in the training data without uniformly
    transforming all samples.

    Args:
        image (np.ndarray): Input BGR image.
        steering (float): Current steering angle.

    Returns:
        tuple: (augmented image, adjusted steering angle).
    """
    if random.random() < 0.5:
        image, steering = augment_pan(image, steering)
    if random.random() < 0.5:
        image = augment_zoom(image)
    if random.random() < 0.5:
        image = augment_brightness(image)
    if random.random() < 0.5:
        image, steering = augment_flip(image, steering)
    return image, steering


# =============================================================================
# BATCH GENERATOR
# =============================================================================

def batch_generator(image_paths, steering_angles, batch_size, is_training=True):
    """
    Generator that yields batches of preprocessed images and steering angles.

    For training data, random augmentation is applied before preprocessing.
    For validation data, only preprocessing is applied (no augmentation).

    Args:
        image_paths (array-like): List of image file paths.
        steering_angles (array-like): Corresponding steering angles.
        batch_size (int): Number of samples per batch.
        is_training (bool): If True, apply data augmentation.

    Yields:
        tuple: (batch_images, batch_steerings) as numpy arrays.
    """
    while True:
        batch_images = []
        batch_steerings = []

        for _ in range(batch_size):
            # Random sample from the dataset
            idx = random.randint(0, len(image_paths) - 1)
            img_path = image_paths[idx]
            steering = steering_angles[idx]

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Apply augmentation only during training
            if is_training:
                img, steering = random_augment(img, steering)

            # Apply preprocessing (crop, YUV, blur, resize, normalize)
            img = preprocess_image(img)

            batch_images.append(img)
            batch_steerings.append(steering)

        yield np.array(batch_images), np.array(batch_steerings)
