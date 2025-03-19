import cv2
import dlib
import numpy as np

# Load the input image
img = cv2.imread('img/1651254812872.jpg')
img = cv2.resize(img, (720, 640))
frame = img.copy()

# ------------ Model for Age detection --------#
age_weights = r"C:\\file\\age_deploy.prototxt"
age_config = r"C:\\file\\age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights)

# Model requirements for image
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Storing the image dimensions
fH = img.shape[0]
fW = img.shape[1]

Boxes = []  # To store the face coordinates
mssg = 'Face Detected'  # To display on image

# ------------- Model for face detection --------#
face_detector = dlib.get_frontal_face_detector()

# Converting to grayscale
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ------------- Detecting the faces -------------#
faces = face_detector(img_gray)

# If no faces are detected
if not faces:
    mssg = 'No face detected'
    cv2.putText(img, f'{mssg}', (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
    cv2.imshow("Age Detection", img)
    cv2.waitKey(5000)  # Close after 5 seconds
    cv2.destroyAllWindows()

else:
    # --------- Bounding Face ---------#
    for face in faces:
        x = face.left()
        y = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Rescaling those coordinates for our image
        box = [x, y, x2, y2]
        Boxes.append(box)
        cv2.rectangle(frame, (x, y), (x2, y2),
                      (0, 200, 200), 2)

    for box in Boxes:
        face = frame[box[1]:box[3], box[0]:box[2]]

        # ----- Image preprocessing --------#
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), model_mean, swapRB=False)

        # ------- Age Prediction ---------#
        age_Net.setInput(blob)
        age_preds = age_Net.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.putText(frame, f'{mssg}:{age}', (box[0],
                                             box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

        # ✅ Display the age in the console
        print(f"Detected Age: {age}")

    # ✅ Display the output and close after 5 seconds
    cv2.imshow("Age Detection", frame)
    cv2.waitKey(5000)  # Close after 5 seconds
    cv2.destroyAllWindows()