#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Urdu Alphabet Recognition & Dataset Collection Tool (38 Classes)
================================================================
Real-time recognition using trained TFLite model and data collection.

CONTROLS:
  z           → Enter logging mode (for data collection)
  n           → Normal mode (inference only)
  ESC         → Quit

LABEL KEYS (press AFTER entering logging mode with 'z'):
  1-9   → Labels 0-8   (Ain, Alif or Alif Madda, aRay, Bari yeh, Bay, Cahy, chotti yeh, chotti Hay, Daal)
  a-y   → Labels 9-33  (Dal, Do Chashmi Hay, Fay, Gaaf, Ghain, Hamza, hey, jeem, Kaaf, khay, 
                          Laam, Meem, Noon, pay, Qaaf, Ray, Say, Seen, Sheen, Suaad, 
                          Tay, Toayn, Wow, Zaal, Zaal)
  0     → Label 34     (Zay)
  -     → Label 35     (Zoayn)
  =     → Label 36     (Zuaad)
  \\    → Label 37     (Zzay)

Inference uses: model/urdu_alphabet/keypoint_classifier.tflite
Labels from:   model/urdu_alphabet/keypoint_classifier_label.csv
Data logs to:  model/urdu_alphabet/keypoint.csv
"""
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

# ─── Load Configuration ──────────────────────────────────────────────────
URDU_LABELS_PATH = 'model/urdu_alphabet/keypoint_classifier_label.csv'
URDU_MODEL_PATH = 'model/urdu_alphabet/keypoint_classifier.tflite'
KEYPOINT_CSV_PATH = 'model/urdu_alphabet/keypoint.csv'

# Load label names from CSV
try:
    with open(URDU_LABELS_PATH, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f) if row]
except Exception as e:
    print(f"Error loading labels: {e}")
    keypoint_classifier_labels = [f"Label {i}" for i in range(38)]

NUM_CLASSES = len(keypoint_classifier_labels)

# ─── Key-code → Label-ID mapping ─────────────────────────────────────────────
KEY_TO_LABEL = {}
# Keys '1'-'9' → labels 0-8 (ID 0-8)
for _i in range(1, 10):
    KEY_TO_LABEL[48 + _i] = _i - 1
# Keys 'a'-'y' → labels 9-33 (ID 9-33)
# Note: we skip 'z' (122) as it's the mode trigger
for _i, _c in enumerate(range(ord('a'), ord('y') + 1)):
    KEY_TO_LABEL[_c] = 9 + _i
# Additional keys for remaining classes (IDs 34-37)
KEY_TO_LABEL[48] = 34  # '0' -> Zay
KEY_TO_LABEL[45] = 35  # '-' -> Zoayn
KEY_TO_LABEL[61] = 36  # '=' -> Zuaad
KEY_TO_LABEL[92] = 37  # '\' -> Zzay

# Reverse map for display: label_id → key name
LABEL_TO_KEY = {v: chr(k) if 32 <= k <= 126 else str(k) for k, v in KEY_TO_LABEL.items()}
LABEL_TO_KEY[48] = '0' # fix digit keys
for i in range(1, 10): LABEL_TO_KEY[i-1] = str(i)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", help='min_tracking_confidence', type=float, default=0.5)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # Load Urdu Alphabet Classifier
    keypoint_classifier = KeyPointClassifier(model_path=URDU_MODEL_PATH)
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # State
    mode = 0          # 0 = inference, 1 = logging
    number = -1       # selected label ID

    print("=" * 60)
    print("  URDU ALPHABET RECOGNITION & COLLECTION (38 CLASSES)")
    print("=" * 60)
    print("  Press 'z' to START logging (Dataset Creation)")
    print("  Press 'n' for NORMAL mode (Inference Only)")
    print("  Press ESC to QUIT")
    print("=" * 60)

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode, number)

        ret, image = cap.read()
        if not ret: break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Prediction (Real-time inference)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                label_text = keypoint_classifier_labels[hand_sign_id] if 0 <= hand_sign_id < NUM_CLASSES else "Unknown"

                # Logging (Data collection)
                logging_csv(number, mode, pre_processed_landmark_list)

                # Draw UI
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, label_text, number if mode == 1 else -1)

        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Urdu Alphabet Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode, current_number):
    number = current_number
    if key == ord('z'): # 122
        mode = 1
    elif key == ord('n'): # 110
        mode = 0
    if key in KEY_TO_LABEL:
        number = KEY_TO_LABEL[key]
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [np.array((lx, ly))], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    bx, by = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= bx
        temp_landmark_list[i][1] -= by
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(list(map(abs, temp_landmark_list)))
    return [n / max_val for n in temp_landmark_list]

def logging_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number < NUM_CLASSES):
        with open(KEYPOINT_CSV_PATH, 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])

def draw_landmarks(image, lp):
    if len(lp) > 0:
        for i, j in [(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20),(0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0)]:
            cv.line(image, tuple(lp[i]), tuple(lp[j]), (0,0,0), 6)
            cv.line(image, tuple(lp[i]), tuple(lp[j]), (255,255,255), 2)
    for i, p in enumerate(lp):
        size = 8 if i in (4,8,12,16,20) else 5
        cv.circle(image, (p[0], p[1]), size, (255,255,255), -1)
        cv.circle(image, (p[0], p[1]), size, (0,0,0), 1)
    return image

def draw_bounding_rect(use, image, brect):
    if use: cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)
    return image

def draw_info_text(image, brect, handedness, label_text, logging_id):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0,0,0), -1)
    info = handedness.classification[0].label + ":" + label_text
    if logging_id != -1:
        info += f" [REC ID:{logging_id}]"
    cv.putText(image, info, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    if mode == 1:
        cv.putText(image, "MODE: LOGGING (press n for Normal)", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
        if 0 <= number < NUM_CLASSES:
            key_name = LABEL_TO_KEY.get(number, "?")
            cv.putText(image, f"LOGGING: [{key_name}] {keypoint_classifier_labels[number]}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    else:
        cv.putText(image, "MODE: INFERENCE (press z for Logging)", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

if __name__ == '__main__': main()
