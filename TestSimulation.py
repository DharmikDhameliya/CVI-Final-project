"""
TestSimulation.py - Inference script for testing the trained model in the Udacity Simulator.

This script connects to the Udacity self-driving car simulator via SocketIO.
It receives real-time images from the car's front camera, preprocesses them,
predicts the steering angle using the trained CNN model, and sends the
steering command back to the simulator.

Usage:
    1. Run this script:  python TestSimulation.py
    2. Launch the Udacity simulator
    3. Select "Autonomous Mode"
    4. Watch the car drive itself!

The script will:
    - Listen for telemetry data from the simulator (camera images + speed)
    - Preprocess each frame using the same pipeline as training
    - Predict the steering angle using the trained model
    - Send steering angle and throttle back to the simulator
"""

import os
import argparse
import base64
import numpy as np
from io import BytesIO
from PIL import Image

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

try:
    from tensorflow.keras.models import load_model
except (ImportError, ModuleNotFoundError):
    from keras.models import load_model
from utils import preprocess_image

# SocketIO server and Flask app
sio = socketio.Server()
app = Flask(__name__)

# Global variables
model = None
max_speed = 25  # Maximum speed limit


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test Self-Driving Car Model in Simulator')
    parser.add_argument('--model', type=str, default='model.h5',
                        help='Path to the trained model file (default: model.h5)')
    parser.add_argument('--max_speed', type=int, default=25,
                        help='Maximum speed for the car (default: 25)')
    return parser.parse_args()


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handle telemetry events from the simulator.

    This function is called each time the simulator sends a new frame.
    It processes the image, predicts the steering angle, and sends
    control commands back.

    Args:
        sid: Session ID.
        data (dict): Telemetry data including:
            - 'image': Base64 encoded front camera image
            - 'speed': Current speed of the car
    """
    if data:
        # Get current speed from simulator
        speed = float(data['speed'])

        # Decode the base64 image from the simulator
        image_data = data['image']
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Convert PIL image to numpy array (RGB -> BGR for OpenCV compatibility)
        image = np.asarray(image)
        image = image[:, :, ::-1].copy()  # RGB to BGR

        # Preprocess the image (same pipeline as training)
        processed_image = preprocess_image(image)

        # Add batch dimension: (66, 200, 3) -> (1, 66, 200, 3)
        processed_image = np.expand_dims(processed_image, axis=0)

        # Predict steering angle
        steering_angle = float(model.predict(processed_image, verbose=0)[0][0])

        # Calculate throttle based on speed
        # Slow down if too fast, speed up if too slow
        throttle = 1.0 - (speed / max_speed) - abs(steering_angle) * 0.3
        throttle = max(0.1, throttle)  # Minimum throttle to keep moving

        print(f"Steering: {steering_angle:.4f} | Throttle: {throttle:.4f} | Speed: {speed:.2f}")

        # Send control commands back to the simulator
        send_control(steering_angle, throttle)
    else:
        # If no data, send neutral commands
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """Handle new client connections."""
    print("[INFO] Simulator connected!")
    send_control(0, 0)  # Send initial zero commands


def send_control(steering_angle, throttle):
    """
    Send steering and throttle commands to the simulator.

    Args:
        steering_angle (float): Steering angle prediction (-1 to 1).
        throttle (float): Throttle value (0 to 1).
    """
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)


def main():
    """Main entry point for the testing script."""
    global model, max_speed

    args = parse_args()
    max_speed = args.max_speed

    # Load the trained model
    print("=" * 60)
    print("SELF-DRIVING CAR - AUTONOMOUS MODE")
    print("=" * 60)
    print(f"[INFO] Loading model from: {args.model}")
    model = load_model(args.model, compile=False)  # compile=False fixes Keras version mismatch
    print("[INFO] Model loaded successfully!")
    print(f"[INFO] Max speed: {max_speed}")
    print("\n[INFO] Waiting for simulator connection...")
    print("[INFO] Please launch the simulator and select 'Autonomous Mode'")
    print("=" * 60)

    # Wrap Flask app with SocketIO middleware
    app_with_sio = socketio.WSGIApp(sio, app)

    # Start the server on port 4567 (Udacity simulator default)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app_with_sio)


if __name__ == '__main__':
    main()
    