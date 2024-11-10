# pip install opencv-python mediapipe scipy pygame numpy


import cv2
import mediapipe as mp
from scipy.spatial import distance
import pygame
import time
import logging

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('alert.wav')

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Define constants
EAR_THRESHOLD = 0.3  # Detect closed eyes when EAR falls below this threshold
CLOSED_EYE_MIN_DURATION = 2  # Minimum duration (seconds) eyes must be closed for alert
FATIGUE_THRESHOLD = 5  # Number of drowsiness events to indicate fatigue
ALERT_COOLDOWN = 5  # Cooldown in seconds between alerts
counter = 0
drowsy_event_count = 0
start_time = None  # To track time when eyes are closed
last_alert_time = 0  # Track the time of the last alert

# Setup logging
logging.basicConfig(filename='drowsiness_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# EAR calculation function using eye landmarks
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_eye_landmarks(face_landmarks, left_eye_indices, right_eye_indices):
    left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in left_eye_indices]
    right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in right_eye_indices]
    return left_eye, right_eye

def alert_sound():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def stop_sound():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def alert_message(frame):
    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def log_event():
    logging.info("Drowsiness event detected!")

# Left and right eye landmark indices (based on Mediapipe's face mesh model)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            left_eye, right_eye = get_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)

            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            print(f"EAR: {ear}")
            
            # Detect if eyes are closed
            if ear < EAR_THRESHOLD:
                if start_time is None:
                    # Record the time when eyes were first detected closed
                    start_time = time.time()
                else:
                    # Check how long the eyes have been closed
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= CLOSED_EYE_MIN_DURATION and (time.time() - last_alert_time) > ALERT_COOLDOWN:
                        # Trigger alert when eyes are closed for longer than the minimum duration
                        alert_message(frame)
                        alert_sound()
                        log_event()  # Log the drowsiness event
                        drowsy_event_count += 1
                        last_alert_time = time.time()  # Update the last alert time
                        if drowsy_event_count >= FATIGUE_THRESHOLD:
                            cv2.putText(frame, "FATIGUE DETECTED! TAKE A BREAK!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Reset the timer and stop sound when eyes are open
                start_time = None
                stop_sound()
                drowsy_event_count = max(0, drowsy_event_count - 1)  # Reduce event count over time

            # Visual feedback for eye state
            state_text = "Active" if ear >= EAR_THRESHOLD else "Drowsy"
            cv2.putText(frame, f"State: {state_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("No face landmarks detected, resetting counter.")
        start_time = None  # Reset if no face is detected
        stop_sound()

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
