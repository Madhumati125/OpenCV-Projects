import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import os
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open webcam for live detection
cap = cv2.VideoCapture(0)

# Load dlib's face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
try:
    dlib_facelandmark = dlib.shape_predictor(r"C:\\models\\shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    cap.release()
    cv2.destroyAllWindows()
    os._exit(1)

# Function to calculate Eye Aspect Ratio (EAR)
def detect_eye(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2 * C)

# Drowsiness parameters
EAR_THRESHOLD = 0.25
CLOSED_FRAMES_THRESHOLD = 15
frame_counter = 0
alert_triggered = False
alert_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            landmarks = dlib_facelandmark(gray, face)

            # Get coordinates for left and right eye
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            # Compute EAR for both eyes
            left_EAR = detect_eye(left_eye)
            right_EAR = detect_eye(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2

            # Display "Eyes Open - Monitoring" at the top
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Check for drowsiness
            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CLOSED_FRAMES_THRESHOLD:
                    if not alert_triggered:
                        alert_time = time.time()
                        print("[ALERT] DROWSINESS DETECTED! Wake up dude!")
                        engine.say("DROWSINESS DETECTED! Wake up dude!")
                        engine.runAndWait()
                        alert_triggered = True
            else:
                frame_counter = 0

            # Keep alert at the bottom for 10 seconds even if eyes open
            if alert_triggered and time.time() - alert_time <= 10:
                text_position = (50, frame.shape[0] - 50)  # Bottom position
                cv2.putText(frame, "ALERT! WAKE UP DUDE!", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            elif alert_triggered and time.time() - alert_time > 10:
                print("\n[INFO] Alert ended. Closing window...")
                time.sleep(2)  # Short delay before closing
                cap.release()
                cv2.destroyAllWindows()
                if engine:  # ✅ Check before stopping
                    engine.stop()
                    engine = None
                exit(0)

        # Display the frame with status
        cv2.imshow("Drowsiness Detector", frame)

        # Close the window manually when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Exiting...")
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Closing...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if engine:  # ✅ Only stop if it's not None
        engine.stop()
        engine = None
    time.sleep(1)
    os._exit(0)  # ✅ Force close without traceback