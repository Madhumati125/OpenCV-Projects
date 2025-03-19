import cv2
import os

file_path = r'C:\Users\madhu\PycharmProjects\OpenCV\Measure Size Of An Object\content\download.jpeg'

# Check if file exists
if not os.path.exists(file_path):
    print("File not found at:", file_path)
else:
    # Read the image
    img = cv2.imread(file_path)

    if img is None:
        print("Failed to load image. Check file format or path.")
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        ret, thresh = cv2.threshold(gray, 127, 255, 0)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the image
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        # Calculate the area of the largest contour (if it exists)
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            scale_factor = 0.1  # 1 pixel = 0.1 cm
            size = area * (scale_factor ** 2)
            print('Size:', size)

        # Display the image with contours
        cv2.imshow('Object with Contours', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the result
        cv2.imwrite('object_with_contours.jpg', img)