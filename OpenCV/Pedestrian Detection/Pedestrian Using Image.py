import cv2
import imutils

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Reading the Image
image = cv2.imread('img/img.jpg')

if image is None:
    print("❌ Error: Could not read the image. Please check the file path.")
else:
    print("✅ Image loaded successfully.")

# Resizing the Image
image = imutils.resize(image, width=min(400, image.shape[1]))

# Detecting all the regions in the image that have pedestrians
print("🔎 Detecting pedestrians...")
(regions, _) = hog.detectMultiScale(image,
                                    winStride=(4, 4),
                                    padding=(4, 4),
                                    scale=1.05)

# Drawing the regions in the Image
if len(regions) > 0:
    print(f"🚶 Detected {len(regions)} pedestrian(s).")
else:
    print("❌ No pedestrians detected.")

for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Showing the output Image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Release resources
cv2.destroyAllWindows()
print("✅ Detection complete. Window closed.")