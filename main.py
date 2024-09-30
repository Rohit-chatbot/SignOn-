import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Load gesture recognition model
model_dict = pickle.load(open('./model.p', 'rb'))  # Ensure correct filename
model = model_dict['model']

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Set lower resolution to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Hands with dynamic mode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Gesture labels
labels_dict = {0: 'Good', 1: 'Morning', 2: 'Judges', 3: "I", 4: "Wish", 
               5: "You", 6: "All", 7: "Have", 8: "Great", 9: "Day", 
               10: "Ahead", 11: ""}

# Store the last prediction to avoid repeating TTS for the same gesture
last_prediction = None
frame_counter = 0

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame only every 3rd frame to reduce lag
    if frame_counter % 3 == 0:
        results = hands.process(frame_rgb)
    frame_counter += 1

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions for prediction
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmark positions
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Predict gesture
        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # If the prediction is new, perform text-to-speech
            if predicted_character != last_prediction:
                last_prediction = predicted_character
                print(f"Predicted gesture: {predicted_character}")
                engine.say(predicted_character)
                engine.runAndWait()

            # Get bounding box coordinates for drawing
            H, W, _ = frame.shape
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            # Draw a rectangle and display the predicted gesture on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Gesture Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

