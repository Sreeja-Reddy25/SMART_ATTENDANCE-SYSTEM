import csv
import datetime
import face_recognition
import cv2
import dlib
import os
import subprocess
import time
import numpy as np
from imutils import face_utils

def mark_attendance(name):
    # Read existing attendance records
    existing_attendance = {}
    try:
        with open('attendance.csv', mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    existing_attendance[row[2]] = row[0]

    except FileNotFoundError:
        print("Attendance file not found. Creating a new one.")
        with open('attendance.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Name"])

    except PermissionError:
        print("The attendance file is being used by another process. Please close it and try again.")
        return

    # Get today's date
    today_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # Check if name is valid and attendance is not already marked for today
    if name in existing_attendance and existing_attendance[name] == today_date:
        print(f"Attendance already marked for {name} today.")
    elif name != "Unknown":
        # Append the attendance to the CSV file
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        with open('attendance.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([today_date, current_time, name])
        print("Attendance marked for", name)
    else:
        print("Invalid user.")

def open_attendance_sheet():
    try:
        subprocess.Popen(['start', 'attendance.csv'], shell=True)
    except Exception as e:
        print(f"Failed to open attendance sheet: {e}")

def detect_blinks(frames):
    # Initialize dlib's face detector (HOG-based) and create a facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    blink_count = 0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            left_eye = shape[left_eye_start:left_eye_end]
            right_eye = shape[right_eye_start:right_eye_end]
            
            # Compute the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio together for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Log the ear values for debugging
            print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Average EAR: {ear:.2f}")

            # Check if the eye aspect ratio is below the blink threshold
            if ear < 0.21:  # Adjusted threshold
                blink_count += 1

    print(f"Blink Count: {blink_count}")
    # A threshold to determine if the face is live (this is a simplistic approach)
    return blink_count > 1

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def recognize_faces():
    known_encodings = []
    known_names = []

    # Load known faces and encodings
    for filename in os.listdir('dataset'):
        if filename.endswith('.jpg'):
            image = face_recognition.load_image_file(os.path.join('dataset', filename))
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            name = os.path.splitext(filename)[0]
            known_names.append(name)

    # Initialize variables for face recognition
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Images")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.imshow("Capture Images", frame)
        start_time = time.time()  # Start time for capturing 10 seconds
        frames = []

        while time.time() - start_time < 10:
            ret, frame = cam.read()
            if not ret:
                break

            frames.append(frame)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face matches any known face
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                # Check if any match is found
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                    if detect_blinks(frames):
                        mark_attendance(name)
                        open_attendance_sheet()
                        cam.release()
                        cv2.destroyAllWindows()
                        return
                    else:
                        print("Liveness detection failed.")

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # If 'q' is pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()