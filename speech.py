import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# Initialize TTS engine
engine = pyttsx3.init()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'Good', 1: 'Morning', 2: 'Judges', 3: "I", 4: "Wish", 
               5: "You", 6: "All", 7: "Have", 8: "Great", 9: "Day", 
               10: "Ahead", 11: "A"}

last_prediction = None
last_prediction_time = 0
prediction_interval = 1.0
speak_on_keypress = False  # Flag to control when to speak

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        current_time = time.time()
        if current_time - last_prediction_time > prediction_interval:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character != last_prediction:
                last_prediction = predicted_character
                last_prediction_time = current_time

                # Only speak if speak_on_keypress is True
                if speak_on_keypress:
                    print(f"Predicted gesture: {predicted_character}")
                    engine.say(predicted_character)
                    engine.runAndWait()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Key event handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('x'):  # Press 'x' to toggle speech on keypress
        speak_on_keypress = not speak_on_keypress
        if speak_on_keypress:
            print("TTS is now enabled. Press 'x' again to disable.")
        else:
            print("TTS is now disabled. Press 'x' again to enable.")

cap.release()
cv2.destroyAllWindows()
        