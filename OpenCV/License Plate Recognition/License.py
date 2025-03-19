import cv2
import pytesseract

# Step 1: Set the Tesseract path (update path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\madhu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Step 2: Load the image
img = cv2.imread('car_with_plate.jpg')

# Step 3: Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 4: Apply a bilateral filter to reduce noise and retain edges
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Step 5: Detect edges using Canny edge detection
edged = cv2.Canny(gray, 30, 200)

# Step 6: Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 7: Sort contours based on area and filter out small ones
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
license_plate = None

# Step 8: Loop through contours to find a possible license plate
for contour in contours:
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)

    # If the contour has 4 corners, it might be a license plate
    if len(approx) == 4:
        license_plate = approx
        break

# Step 9: If a license plate is found, extract it
if license_plate is not None:
    x, y, w, h = cv2.boundingRect(license_plate)
    plate_img = gray[y:y + h, x:x + w]

    # Step 10: Apply OCR to extract text from the plate
    text = pytesseract.image_to_string(plate_img, config='--psm 8')

    print("Detected License Plate:", text.strip())

    # Step 11: Draw a rectangle around the license plate
    cv2.drawContours(img, [license_plate], -1, (0, 255, 0), 2)
    cv2.putText(img, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Step 12: Display the final image
cv2.imshow("License Plate Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()