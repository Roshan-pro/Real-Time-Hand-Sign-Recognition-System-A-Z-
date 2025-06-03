import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your model
model = load_model("my_cnn_model.h5")  

# Class labels
class_labels = ["A","B","C","D","E","F","G","H",
                "I","J","K","L","M","N","O","P","Q",
                "R","S","T","U","V","W","X","Y","Z","DEL","NOTHING","SPACE"]

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box of the hand
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Adding some padding around the hand
            offset = 20
            x_min = max(x_min - offset, 0)
            y_min = max(y_min - offset, 0)
            x_max = min(x_max + offset, w)
            y_max = min(y_max + offset, h)

            hand_img = frame[y_min:y_max, x_min:x_max]

            try:
                resized = cv2.resize(hand_img, (128, 128))
                input_img = np.reshape(resized, (1, 128, 128, 3))
                prediction = model.predict(input_img)
                label_index = np.argmax(prediction)
                label = class_labels[label_index]

                # Draw the prediction
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            except Exception as e:
                print("Error processing hand image:", e)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Sign Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
