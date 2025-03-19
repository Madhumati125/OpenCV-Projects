import cv2
import imutils
import time

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video file
cap = cv2.VideoCapture('img/vid.mp4')

# Check if video was opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Timer to stop after 5 seconds
start_time = time.time()

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Detecting regions in the image that have pedestrians
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

        # Displaying detected pedestrian regions
        if len(regions) > 0:
            print(f"🚶 Detected {len(regions)} pedestrian(s).")

            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            print("❌ No pedestrians detected.")

        # Showing the output image
        cv2.imshow("Pedestrian Detection", image)

        # Check if 5 seconds have passed
        if time.time() - start_time >= 5:
            print("✅ 5 seconds of detection have passed. Stopping the video.")
            break

        # Quit on pressing 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("✅ Video stopped by user.")
            break
    else:
        print("❌ Error: Failed to read frame.")
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
print("✅ Program completed.")