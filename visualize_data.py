"""
visualize_data.py - Script to visualize and explore the collected driving data.

This script helps you:
    1. Verify the data was collected correctly
    2. View sample images from the dataset
    3. Plot steering angle distribution
    4. Check data balance before and after balancing

Usage:
    python visualize_data.py --data_dir ./data
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_driving_data,
    fix_image_paths,
    balance_data,
    plot_steering_histogram,
    preprocess_image,
    random_augment
)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Driving Data')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    return parser.parse_args()


def show_sample_images(df, num_samples=5):
    """Display sample images from the dataset with their steering angles."""
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    samples = df.sample(n=num_samples, random_state=42)

    for idx, (_, row) in enumerate(samples.iterrows()):
        img = cv2.imread(row['center'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Steering: {row['steering']:.3f}")
            axes[idx].axis('off')

    plt.suptitle('Sample Images from Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_preprocessing_pipeline(df):
    """Show the preprocessing steps applied to an image."""
    sample = df.sample(n=1, random_state=42).iloc[0]
    img = cv2.imread(sample['center'])

    if img is None:
        print("[ERROR] Could not read image.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original')

    # Cropped
    cropped = img[60:135, :, :]
    axes[0, 1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. Cropped (rows 60-135)')

    # YUV
    yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    axes[0, 2].imshow(yuv)
    axes[0, 2].set_title('3. YUV Color Space')

    # Blurred
    blurred = cv2.GaussianBlur(yuv, (3, 3), 0)
    axes[1, 0].imshow(blurred)
    axes[1, 0].set_title('4. Gaussian Blur')

    # Resized
    resized = cv2.resize(blurred, (200, 66))
    axes[1, 1].imshow(resized)
    axes[1, 1].set_title('5. Resized (200x66)')

    # Normalized
    normalized = resized / 255.0
    axes[1, 2].imshow(normalized)
    axes[1, 2].set_title('6. Normalized [0,1]')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Image Preprocessing Pipeline', fontsize=14)
    plt.tight_layout()
    plt.savefig('preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_augmentations(df):
    """Show examples of data augmentation applied to an image."""
    sample = df.sample(n=1, random_state=42).iloc[0]
    img = cv2.imread(sample['center'])
    steering = sample['steering']

    if img is None:
        print("[ERROR] Could not read image.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original (steer={steering:.3f})')

    # Generate 5 random augmentations
    for i in range(5):
        row, col = (i + 1) // 3, (i + 1) % 3
        aug_img, aug_steer = random_augment(img.copy(), steering)
        axes[row, col].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Augmented (steer={aug_steer:.3f})')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    args = parse_args()

    # Load data
    df = load_driving_data(args.data_dir)
    df = fix_image_paths(df, args.data_dir)

    print(f"\nDataset shape: {df.shape}")
    print(f"Steering angle range: [{df['steering'].min():.3f}, {df['steering'].max():.3f}]")
    print(f"Mean steering angle: {df['steering'].mean():.3f}")

    # Plot histograms
    plot_steering_histogram(df['steering'], title='Before Balancing', save_path='hist_before.png')
    df_balanced = balance_data(df)
    plot_steering_histogram(df_balanced['steering'], title='After Balancing', save_path='hist_after.png')

    # Show sample images
    show_sample_images(df)

    # Show preprocessing pipeline
    show_preprocessing_pipeline(df)

    # Show augmentation examples
    show_augmentations(df)

    print("\n[INFO] All visualizations saved!")


if __name__ == '__main__':
    main()
