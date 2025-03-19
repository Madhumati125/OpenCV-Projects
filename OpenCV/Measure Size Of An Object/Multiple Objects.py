import cv2
import numpy as np

# Read the image
img = cv2.imread('content/image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold to separate objects from background
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours of objects
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Scale factor for real-world size (1 pixel = 0.1 cm)
scale_factor = 0.1

for cnt in contours:
    # Calculate the area in pixels
    area = cv2.contourArea(cnt)

    # Convert pixel area to real-world size
    size = area * (scale_factor ** 2)

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(cnt)

    # Draw rectangle around the object
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display size on the image
    cv2.putText(img, f"{size:.2f} cm²", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)

    # ✅ Print size in terminal
    print(f"Object at ({x}, {y}) - Area: {area:.2f} pixels, Size: {size:.2f} cm²")

# Display the image with contours and size labels
cv2.imshow('Objects with Size', img)
cv2.waitKey(0)
cv2.destroyAllWindows()