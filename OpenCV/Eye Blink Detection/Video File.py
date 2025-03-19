import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('assets/my_blink.mp4', fourcc, 05.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to file
    out.write(frame)

    # Display the frame
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()