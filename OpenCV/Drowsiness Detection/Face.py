import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import os
import time

# Initialize pyttsx3 for audio alerts
engine = pyttsx3.init()

# Start camera capture
cap = cv2.VideoCapture(0)

# Face detector and landmarks predictor
face_detector = dlib.get_frontal_face_detector()

# Load the landmark model safely
try:
    dlib_facelandmark = dlib.shape_predictor(r"C:\\models\\shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    cap.release()
    cv2.destroyAllWindows()
    os._exit(1)

# Function to calculate Eye Aspect Ratio (EAR)
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_scale)

        for face in faces:
            face_landmarks = dlib_facelandmark(gray_scale, face)
            leftEye = []
            rightEye = []

            # Right eye points (42 to 47)
            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                next_point = n + 1 if n != 47 else 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

            # Left eye points (36 to 41)
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1 if n != 41 else 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        cv2.imshow("Drowsiness Detector", frame)

        # Close window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Closing...")

finally:
    # Release the camera and close window
    cap.release()
    cv2.destroyAllWindows()

    # Small delay before force exit (if needed)
    time.sleep(1)
    os._exit(0)  # Force kill all threads and processes