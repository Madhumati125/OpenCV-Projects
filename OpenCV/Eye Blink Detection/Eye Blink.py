import cv2  # for video rendering
import dlib  # for face and landmark detection
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Load the video file
cam = cv2.VideoCapture('assets/my_blink.mp4')

# ✅ Check if the video file opened successfully
if not cam.isOpened():
    print("Error: Could not open video file.")
    exit()


# defining a function to calculate the EAR
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR


# Variables
blink_thresh = 0.45
succ_frame = 2
count_frame = 0

# Eye landmarks 
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initializing the Models for Landmark and face Detection 
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
    'C:\\models\\shape_predictor_68_face_landmarks.dat')

# ✅ Start the timer
start_time = time.time()

while True:
    # If the video is finished, reset it to the start 
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    else:
        ret, frame = cam.read()
        if not ret:
            print("Error reading frame from video.")
            break

        frame = imutils.resize(frame, width=640)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting the faces 
        faces = detector(img_gray)
        for face in faces:
            shape = landmark_predict(img_gray, face)
            shape = face_utils.shape_to_np(shape)

            lefteye = shape[L_start:L_end]
            righteye = shape[R_start:R_end]

            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)

            avg = (left_EAR + right_EAR) / 2
            if avg < blink_thresh:
                count_frame += 1
            else:
                if count_frame >= succ_frame:
                    cv2.putText(frame, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 4)
                count_frame = 0

        cv2.imshow("Video", frame)

        # ✅ Close window after 5 seconds
        if time.time() - start_time > 5:
            print("Closing window after 5 seconds.")
            break

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()