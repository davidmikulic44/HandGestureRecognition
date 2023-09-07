import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode = False, 
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

model = load_model('model')

class_names = ['Bok','Peace','Like','Dislike','Okej']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    

    confidence = ''
    class_name = ''
    hand_data = []
    x_ = []
    y_ = []
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
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


            mp_draw.draw_landmarks(frame, 
                                    hand_landmarks, 
                                    mp_hands.HAND_CONNECTIONS)
            
            hand_data = np.array(hand_data)
            hand_data = hand_data.reshape(-1, 42)
            
            prediction = model.predict([hand_data], verbose=0)
            
            classID = np.argmax(prediction)
            
            confidenceList = list(prediction[0])
            confidence = str(round(confidenceList[classID]*100, 2))+'%'
            
            class_name = class_names[classID]
    
    cv2.putText(frame, 
                class_name, 
                (10, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1, (0,0,0), 8, cv2.LINE_AA)
    cv2.putText(frame, 
                class_name, 
                (10, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, 
                confidence, 
                (10, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, 
                confidence,
                (10, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.7, (255,255,255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, 
                'Press ESC to exit', 
                (490, 460), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, 
                'Press ESC to exit', 
                (490, 460), 
                cv2.FONT_HERSHEY_DUPLEX, 
                0.5, (255,255,255), 1, cv2.LINE_AA)


    cv2.imshow('Output', frame) 
    
    key = cv2.waitKey(10) #ms
    if key == 27:  # ESC
        break
    

cap.release()
cv2.destroyAllWindows()