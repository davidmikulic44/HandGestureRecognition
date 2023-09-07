import os
import csv
import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode = False, 
                       max_num_hands=1, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)


DATA_DIR = './data'
file = open('data.csv', 'w')
fields = ('label','data')
writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')

for directory in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, directory)):
        hand_data = []
        x_ = []
        y_ = []

        frame = cv2.imread(os.path.join(DATA_DIR, directory, img_path))
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(framergb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    hand_data.append(x - min(x_))
                    hand_data.append(y - min(y_))

            writer.writerow({'label' : directory, 'data' : hand_data})


file.close()