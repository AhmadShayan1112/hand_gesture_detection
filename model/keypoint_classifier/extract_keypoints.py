#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import os
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from tqdm import tqdm

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def main():
    # MediaPipe Hands Setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    # Dataset path (one level up from hand-gesture-recognition-mediapipe)
    # The current file is model/keypoint_classifier/extract_keypoints.py
    # Dataset is in ../../../digits/
    dataset_path = 'digits'
    csv_path = 'hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint.csv'

    # Clear/Initialize CSV
    with open(csv_path, 'w', newline="") as f:
        pass

    # Mapping clarification:
    # Index 0 --> Label "1" (Folder "1")
    # Index 1 --> Label "2" (Folder "2")
    # ...
    # Index 9 --> Label "10" (Folder "10")
    digit_labels = [str(i) for i in range(1, 11)]

    for index, label in enumerate(digit_labels):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue

        print(f"Processing folder: {label}")
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_name in tqdm(images):
            image_path = os.path.join(folder_path, image_name)
            image = cv.imread(image_path)
            if image is None:
                continue

            # Detection
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Landmark calculation
                    landmark_list = calc_landmark_list(image, hand_landmarks)

                    # Pre-processing
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Write to CSV
                    with open(csv_path, 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([index, *pre_processed_landmark_list])

    hands.close()
    print("Keypoint extraction complete.")


if __name__ == '__main__':
    main()
