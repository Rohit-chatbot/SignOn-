import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Argument parsing for flexibility
parser = argparse.ArgumentParser(description='Process hand gesture images.')
parser.add_argument('--data_dir', type=str, default='./data', help='Directory of the data')
args = parser.parse_args()

DATA_DIR = args.data_dir
data = []
labels = []

# Initialize Mediapipe hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Iterate over directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            logging.info(f'Processing image: {img_full_path}')
            try:
                img = cv2.imread(img_full_path)
                if img is None:
                    logging.warning(f'Image not found or unable to read: {img_full_path}')
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux = []
                        x_ = []
                        y_ = []

                        # Extract landmark coordinates
                        for landmark in hand_landmarks.landmark:
                            x = landmark.x
                            y = landmark.y
                            x_.append(x)
                            y_.append(y)

                        # Normalize coordinates
                        min_x, min_y = min(x_), min(y_)
                        normalized_coords = [(x - min_x, y - min_y) for x, y in zip(x_, y_)]
                        data_aux.extend(normalized_coords)

                        data.append(data_aux)
                        labels.append(dir_)

                        # Visualization (optional)
                        mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        plt.imshow(img_rgb)
                        plt.title(f'Processed: {dir_}/{img_path}')
                        plt.axis('off')
                        plt.show()
                else:
                    logging.warning(f'No hand landmarks found in image: {img_full_path}')

            except Exception as e:
                logging.error(f'Error processing image {img_full_path}: {e}')

# Save the processed data and labels into a pickle file
try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    logging.info('Data saved to data.pickle')
except Exception as e:
    logging.error(f'Error saving data to pickle file: {e}')
