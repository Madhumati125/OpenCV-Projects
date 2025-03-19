import cv2
import time

# Load the face detection model
cascade = cv2.CascadeClassifier(r"C:\\file\\haarcascade_frontalface_default.xml")

# Start capturing video from the webcam (0 for laptop webcam)
video_capture = cv2.VideoCapture(0)

# Start the timer
start_time = time.time()

while True:
    # Capture the latest frame
    check, frame = video_capture.read()
    if not check:
        break

    # Convert to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(gray_image, scaleFactor=2.0, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Apply blur to the face
        frame[y:y + h, x:x + w] = cv2.medianBlur(frame[y:y + h, x:x + w], 35)

        # ✅ Print message in the console
        print("Face blurred")

    # Show the output frame
    cv2.imshow('Face Blurred', frame)

    # Close the window after 5 seconds or on 'q' key press
    if (time.time() - start_time > 5) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()