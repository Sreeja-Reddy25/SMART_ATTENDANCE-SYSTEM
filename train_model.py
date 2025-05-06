import face_recognition
import os
import cv2
import sqlite3
import numpy as np

def encode_faces():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB,
            roll_no TEXT,
            class TEXT,
            picture BLOB
        )
    ''')

    for image_name in os.listdir('dataset'):
        if image_name.endswith(".jpg"):
            image_path = os.path.join('dataset', image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                encoding = encodings[0]
                # Split image name based on underscores
                name_parts = image_name.split('_')
                if len(name_parts) >= 2:  # Ensure at least two parts are available
                    name = name_parts[0]
                    roll_no = name_parts[1]
                    student_class = 'Unknown'  # Default value if class is not provided
                    if len(name_parts) >= 3:
                        student_class = name_parts[2]
                    with open(image_path, 'rb') as file:
                        picture = file.read()
                    cursor.execute("INSERT INTO faces (name, encoding, roll_no, class, picture) VALUES (?, ?, ?, ?, ?)",
                                   (name, encoding.tobytes(), roll_no, student_class, picture))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    encode_faces()