"""
model.py - NVIDIA Self-Driving Car CNN Model Architecture.

This module defines the NVIDIA end-to-end deep learning model for
self-driving cars. The architecture follows the paper:
"End to End Learning for Self-Driving Cars" by NVIDIA (2016).

Architecture (from the project specification - Figure 7):
    Input: 66x200x3 (YUV image)
    - Normalization layer
    - Conv2D: 24 filters, 5x5 kernel, 2x2 stride
    - Conv2D: 36 filters, 5x5 kernel, 2x2 stride
    - Conv2D: 48 filters, 5x5 kernel, 2x2 stride
    - Conv2D: 64 filters, 3x3 kernel
    - Conv2D: 64 filters, 3x3 kernel
    - Flatten
    - Dense: 1164 neurons
    - Dense: 100 neurons
    - Dense: 50 neurons
    - Dense: 10 neurons
    - Output: 1 neuron (steering angle)
"""

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, Dense, Flatten, Dropout, Input
    )
    from tensorflow.keras.optimizers import Adam
except (ImportError, ModuleNotFoundError):
    # TensorFlow 2.16+ / Keras 3 standalone imports
    from keras.models import Sequential
    from keras.layers import (
        Conv2D, Dense, Flatten, Dropout, Input
    )
    from keras.optimizers import Adam


def build_nvidia_model(input_shape=(66, 200, 3), learning_rate=1e-3):
    """
    Build and compile the NVIDIA self-driving car CNN model.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tensorflow.keras.Model: Compiled Keras model ready for training.
    """
    model = Sequential([
        # Input layer
        Input(shape=input_shape),

        # ---- Convolutional Layers ----

        # Conv Layer 1: 24 filters, 5x5 kernel, stride 2x2
        # Output: 31x98x24
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),

        # Conv Layer 2: 36 filters, 5x5 kernel, stride 2x2
        # Output: 14x47x36
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),

        # Conv Layer 3: 48 filters, 5x5 kernel, stride 2x2
        # Output: 5x22x48
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),

        # Conv Layer 4: 64 filters, 3x3 kernel
        # Output: 3x20x64
        Conv2D(64, (3, 3), activation='elu'),

        # Conv Layer 5: 64 filters, 3x3 kernel
        # Output: 1x18x64
        Conv2D(64, (3, 3), activation='elu'),

        # ---- Flatten ----
        Flatten(),

        # ---- Fully Connected Layers ----

        # FC Layer 1: 1164 neurons
        Dense(1164, activation='elu'),
        Dropout(0.5),  # Dropout for regularization

        # FC Layer 2: 100 neurons
        Dense(100, activation='elu'),
        Dropout(0.3),

        # FC Layer 3: 50 neurons
        Dense(50, activation='elu'),
        Dropout(0.2),

        # FC Layer 4: 10 neurons
        Dense(10, activation='elu'),

        # Output Layer: 1 neuron (predicted steering angle)
        Dense(1)
    ])

    # Compile with Adam optimizer and MSE loss (regression task)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )

    print("[INFO] NVIDIA Self-Driving Car Model Summary:")
    model.summary()

    return model


if __name__ == '__main__':
    # Quick test: build and display the model
    model = build_nvidia_model()
