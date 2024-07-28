import streamlit as st
import pandas as pd
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import csv

# Load data for face recognition
def load_data():
    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)
    with open('data/names.pkl', 'rb') as w:
        names = pickle.load(w)
    with open('data/rollnos.pkl', 'rb') as r:
        rollnos = pickle.load(r)
    with open('data/roles.pkl', 'rb') as l:
        roles = pickle.load(l)
    return faces_data, names, rollnos, roles

def capture_face_data(name, rollno, role):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0

    st.write("Capturing face data...")

    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, f'Samples: {len(faces_data)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels='RGB', caption='Capturing Face Data')

        if len(faces_data) >= 100:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(faces_data.shape[0], -1)

    if not os.path.exists('data'):
        os.makedirs('data')

    # Save names
    names_path = 'data/names.pkl'
    if os.path.isfile(names_path):
        with open(names_path, 'rb') as f:
            existing_names = pickle.load(f)
        existing_names += [name] * 100
    else:
        existing_names = [name] * 100
    with open(names_path, 'wb') as f:
        pickle.dump(existing_names, f)

    # Save rollnos
    rollnos_path = 'data/rollnos.pkl'
    if os.path.isfile(rollnos_path):
        with open(rollnos_path, 'rb') as f:
            existing_rollnos = pickle.load(f)
        existing_rollnos += [rollno] * 100
    else:
        existing_rollnos = [rollno] * 100
    with open(rollnos_path, 'wb') as f:
        pickle.dump(existing_rollnos, f)

    # Save roles
    roles_path = 'data/roles.pkl'
    if os.path.isfile(roles_path):
        with open(roles_path, 'rb') as f:
            existing_roles = pickle.load(f)
        existing_roles += [role] * 100
    else:
        existing_roles = [role] * 100
    with open(roles_path, 'wb') as f:
        pickle.dump(existing_roles, f)

    # Save faces data
    faces_data_path = 'data/faces_data.pkl'
    if os.path.isfile(faces_data_path):
        with open(faces_data_path, 'rb') as f:
            existing_faces = pickle.load(f)
        existing_faces = np.append(existing_faces, faces_data, axis=0)
    else:
        existing_faces = faces_data
    with open(faces_data_path, 'wb') as f:
        pickle.dump(existing_faces, f)

    st.write(f"Data for {name} saved successfully!")

def track_attendance():
    faces_data, names, rollnos, roles = load_data()
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_data, names)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    COL_NAMES = ['NAME', 'ROLLNO', 'ROLE', 'TIME']

    st.write("Press 'Start Tracking' to begin.")
    st.write("Press 'Stop Tracking' to stop.")

    tracking = False
    detected_faces = {}

    stframe = st.empty()

    if st.button('Start Tracking'):
        tracking = True
        st.write("Tracking started.")

    if st.button('Stop Tracking'):
        tracking = False
        st.write("Tracking stopped.")
        video.release()
        cv2.destroyAllWindows()

    while tracking:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture image from camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            person = output[0]
            person_index = names.index(person)

            current_time = time.time()

            if person not in detected_faces:
                detected_faces[person] = current_time
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                file_path = f"Attendance/Attendance_{date}.csv"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x, y), (50, 50, 255), -1)
                cv2.putText(frame, person, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                attendance = [person, rollnos[person_index], roles[person_index], str(timestamp)]
                file_exists = os.path.isfile(file_path)
                with open(file_path, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

                message = f"Attendance marked for {person} at {timestamp}"
                cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels='RGB', caption='Attendance Tracking')

                time.sleep(2)

    if not tracking:
        video.release()
        cv2.destroyAllWindows()

def main():
    st.title('Face Recognition Attendance System')

    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select Option', ['Home', 'Register', 'Track Attendance'])

    if options == 'Home':
        st.write('Welcome to the Face Recognition Attendance System.')
    elif options == 'Register':
        st.header('Register Face Data')
        name = st.text_input('Enter your Name:')
        rollno = st.text_input('Enter your Roll Number:')
        role = st.selectbox('Select your Role:', ['Student', 'Teacher'])
        if st.button('Register'):
            if name and rollno and role:
                capture_face_data(name, rollno, role)
            else:
                st.warning('Please enter all fields.')
    elif options == 'Track Attendance':
        st.header('Track Attendance')
        track_attendance()

if __name__ == '__main__':
    main()
