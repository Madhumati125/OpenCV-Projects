import cv2
import webbrowser

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)  # Set frame rate to 10 FPS

detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    if frame is None:
        continue

    try:
        data, bbox, _ = detector.detectAndDecode(frame)
    except Exception as e:
        print(f"Decoding error: {e}")
        continue

    if data:
        print(f"QR Code detected: {data}")
        webbrowser.open(data)  # Open the URL in the browser
        break

    cv2.imshow("QR Code Scanner", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()