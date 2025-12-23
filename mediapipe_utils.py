import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import numpy as np
import cv2

# Model URLs
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def download_model(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

# Download models
hand_model_path = "hand_landmarker.task"
pose_model_path = "pose_landmarker_lite.task"
download_model(HAND_MODEL_URL, hand_model_path)
download_model(POSE_MODEL_URL, pose_model_path)

# Create landmarkers
base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose)
pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

HAND_SIZE = 21 * 3 * 2
POSE_SIZE = 33 * 4

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)

    landmarks = []

    # Hands
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks[:2]:
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])
    while len(landmarks) < HAND_SIZE:
        landmarks.append(0.0)

    # Pose
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0] * POSE_SIZE)

    return np.array(landmarks, dtype=np.float32)




