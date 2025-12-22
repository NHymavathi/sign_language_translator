import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(max_num_hands=2)
pose = mp_pose.Pose()

HAND_SIZE = 21 * 3 * 2
POSE_SIZE = 33 * 4

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    landmarks = []

    # Hands
    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    while len(landmarks) < HAND_SIZE:
        landmarks.extend([0.0, 0.0, 0.0])

    # Pose
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0] * POSE_SIZE)

    return np.array(landmarks, dtype=np.float32)




