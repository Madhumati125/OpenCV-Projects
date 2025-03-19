import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\madhu\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

# Load the image
image = cv2.imread('image/sample.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to preprocess the image
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Optional: Apply noise reduction using median blur
gray = cv2.medianBlur(gray, 3)

# Detect text using Tesseract OCR
text = pytesseract.image_to_string(gray)

# Print recognized text in the console
print("Extracted Text:")
print(text)

# Save the recognized text to a file
with open("recognized.txt", "w") as file:
    file.write(text)

# Draw bounding boxes around the detected text
height, width, _ = image.shape
boxes = pytesseract.image_to_boxes(gray)
for b in boxes.splitlines():
    b = b.split()
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    x, y = x, height - y  # Adjust coordinates since OpenCV uses top-left origin
    w, h = w, height - h
    cv2.rectangle(image, (x, h), (w, y), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Text Detection', image)

# Close window when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()