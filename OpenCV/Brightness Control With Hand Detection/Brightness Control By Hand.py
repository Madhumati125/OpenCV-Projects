import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import time

# Initialize the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Set timer for auto-close (5 seconds)
start_time = time.time()
duration = 5  # Change this value if you want a longer duration

try:
    while (time.time() - start_time) < duration:
        # Read video frame by frame
        success, frame = cap.read()
        if not success:
            break

        # Flip image horizontally
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image
        Process = hands.process(frameRGB)

        landmarkList = []
        # If hands are detected
        if Process.multi_hand_landmarks:
            for handlm in Process.multi_hand_landmarks:
                for _id, landmarks in enumerate(handlm.landmark):
                    height, width, _ = frame.shape
                    x, y = int(landmarks.x * width), int(landmarks.y * height)
                    landmarkList.append([_id, x, y])

                # Draw hand landmarks
                Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

        if landmarkList != []:
            # Thumb tip
            x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
            # Index finger tip
            x_2, y_2 = landmarkList[8][1], landmarkList[8][2]

            # Draw circles on tips
            cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

            # Draw line between thumb and index finger
            cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

            # Calculate distance between tips
            L = hypot(x_2 - x_1, y_2 - y_1)

            # Adjust brightness based on distance
            b_level = np.interp(L, [15, 220], [0, 100])
            sbc.set_brightness(int(b_level))

        # Display frame
        cv2.imshow('Image', frame)

        # Optional: Close on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Closing...")

finally:
    # Release resources and close window
    cap.release()
    cv2.destroyAllWindows()