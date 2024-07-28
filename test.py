# test.py
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

# Function to speak out text
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load data
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

# Debugging prints
print(f"Shape of FACES: {FACES.shape}")
print(f"Number of LABELS: {len(LABELS)}")

# Ensure consistency
assert FACES.shape[0] == len(LABELS), "Number of faces does not match number of labels"

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load background image
imgBackground = cv2.imread("background.png")

# Define column names for the attendance CSV
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        file_path = f"Attendance/Attendance_{date}.csv"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [str(output[0]), str(timestamp)]

    # Update background image with frame
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        # Save attendance to CSV
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)
    
    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
