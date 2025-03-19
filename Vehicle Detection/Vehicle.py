import cv2
import time

# Load video and cascade
haar_cascade = 'haarcascade_car.xml'
video = 'output.avi'
cap = cv2.VideoCapture(video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

start_time = time.time()  # Record the start time

while True:
    ret, frames = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    print("Starting detection...")
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    print("Detection complete")

    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('video', frames)

    # Stop after 10 seconds
    if time.time() - start_time > 10:
        break

    # Stop manually by pressing 'q'
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()