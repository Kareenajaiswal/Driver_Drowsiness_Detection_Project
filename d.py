import cv2
from scipy.spatial import distance
import pygame

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('alert.wav')

# Load the OpenCV DNN face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

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

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold for face detection
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_dnn(frame)

    if len(faces) == 0:
        return None

    for (startX, startY, endX, endY) in faces:
        face_roi = gray[startY:endY, startX:endX]  # Extract the face region of interest
        # You can add code to detect eyes and calculate EAR using landmarks in this region
        # For now, we'll just return a dummy EAR for testing purposes
        ear = 0.3  # Placeholder EAR value for testing
        return ear

def alert_sound():
    if not pygame.mixer.music.get_busy():  # Play the alert sound if not already playing
        pygame.mixer.music.play()

cap = cv2.VideoCapture(0)  # Open the webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
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
