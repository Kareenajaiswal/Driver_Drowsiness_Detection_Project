import cv2
import dlib
from scipy.spatial import distance
import pygame
pygame.mixer.init()
pygame.mixer.music.load('alert.wav')
pygame.mixer.music.play()

# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness detection
FRAME_CHECK = 20  # Number of consecutive frames indicating drowsiness
counter = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:  # Check if no faces are detected
        return None

    for face in faces:
        shape = landmark_detector(gray, face)

        left_eye = []
        right_eye = []

        for i in range(36, 42):
            left_eye.append((shape.part(i).x, shape.part(i).y))
        for i in range(42, 48):
            right_eye.append((shape.part(i).x, shape.part(i).y))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0
        return ear

def alert_sound():
    pygame("alert.wav")  # Play the alert sound

cap = cv2.VideoCapture(0)  # Open the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ear = detect_eyes(frame)

    if ear is not None and ear < EAR_THRESHOLD:
        counter += 1
        if counter >= FRAME_CHECK:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_sound()  # Play an alert sound when drowsiness is detected
    else:
        counter = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
