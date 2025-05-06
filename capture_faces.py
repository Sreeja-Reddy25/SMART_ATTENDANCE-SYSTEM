# capture_images.py
import cv2
import os
import csv

def capture_images():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    if not os.path.exists('captured_images.csv'):
        with open('captured_images.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Roll No", "Class", "Image Path"])

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Images")

    # Prompt user for details
    name = input("Enter the name of the person: ")
    roll_no = input("Enter the roll number: ")
    student_class = input("Enter the class (press Enter if unknown): ").strip() or "Unknown"

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Capture Images", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"dataset/{name}_{roll_no}_{student_class}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")

            # Append the details to the CSV file
            with open('captured_images.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, roll_no, student_class, img_name])
            break  # Exit loop after capturing one image

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
