"""
train.py - Training script for the Self-Driving Car CNN model.

This script:
    1. Loads the driving data (driving_log.csv)
    2. Balances the steering angle distribution
    3. Splits data into training and validation sets
    4. Creates batch generators with augmentation (training only)
    5. Builds the NVIDIA CNN model
    6. Trains the model and saves it
    7. Plots training/validation loss curves

Usage:
    python train.py --data_dir <path_to_data_folder>

Example:
    python train.py --data_dir ./data
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (
    load_driving_data,
    fix_image_paths,
    balance_data,
    plot_steering_histogram,
    batch_generator
)
from model import build_nvidia_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Self-Driving Car Model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory containing driving_log.csv and IMG folder')
    parser.add_argument('--model_output', type=str, default='model.h5',
                        help='Output path for the trained model (default: model.h5)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for training (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--samples_per_bin', type=int, default=400,
                        help='Max samples per bin for data balancing (default: 400)')
    parser.add_argument('--steps_per_epoch', type=int, default=300,
                        help='Steps per training epoch (default: 300)')
    parser.add_argument('--validation_steps', type=int, default=200,
                        help='Steps per validation epoch (default: 200)')
    return parser.parse_args()


def plot_training_history(history, save_path='training_loss.png'):
    """
    Plot and save the training and validation loss curves.

    Args:
        history: Keras training history object.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Training loss plot saved to {save_path}")
    plt.show()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load driving data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading driving data")
    print("=" * 60)
    df = load_driving_data(args.data_dir)
    df = fix_image_paths(df, args.data_dir)

    # ------------------------------------------------------------------
    # 2. Visualize & Balance steering angle distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Balancing data distribution")
    print("=" * 60)
    plot_steering_histogram(
        df['steering'],
        title='Steering Angle Distribution (Before Balancing)',
        save_path='histogram_before.png'
    )

    df = balance_data(df, samples_per_bin=args.samples_per_bin)

    plot_steering_histogram(
        df['steering'],
        title='Steering Angle Distribution (After Balancing)',
        save_path='histogram_after.png'
    )

    # ------------------------------------------------------------------
    # 3. Split into training and validation sets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Splitting data into train/validation sets")
    print("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        df['center'].values,
        df['steering'].values,
        test_size=0.2,
        random_state=42
    )
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")

    # ------------------------------------------------------------------
    # 4. Create batch generators
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Creating batch generators")
    print("=" * 60)
    train_gen = batch_generator(
        X_train, y_train,
        batch_size=args.batch_size,
        is_training=True  # Augmentation ON
    )
    val_gen = batch_generator(
        X_val, y_val,
        batch_size=args.batch_size,
        is_training=False  # Augmentation OFF
    )
    print("[INFO] Batch generators created.")

    # ------------------------------------------------------------------
    # 5. Build the NVIDIA model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Building NVIDIA CNN model")
    print("=" * 60)
    model = build_nvidia_model(learning_rate=args.learning_rate)

    # ------------------------------------------------------------------
    # 6. Train the model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Training the model")
    print("=" * 60)
    history = model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=args.validation_steps,
        verbose=1
    )

    # ------------------------------------------------------------------
    # 7. Save the trained model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Saving the model")
    print("=" * 60)
    model.save(args.model_output)
    print(f"[INFO] Model saved to {args.model_output}")

    # ------------------------------------------------------------------
    # 8. Plot training history
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Plotting training curves")
    print("=" * 60)
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved at: {args.model_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
