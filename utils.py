import os
import numpy as np
import cv2
from PIL import Image
import face_recognition
from models import Employee, Embedding, Session

session = Session()

def preprocess_and_extract_embeddings(employee_name, employee_image_dir):
    embeddings = []
    labels = []

    for image_name in os.listdir(employee_image_dir):
        image_path = os.path.join(employee_image_dir, image_name)
        try:
            image = Image.open(image_path)
            image = np.array(image)

            # Convert image to RGB (if needed)
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

            # Detect face(s)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 0:
                # Extract face embeddings
                face_encodings = face_recognition.face_encodings(image, face_locations)

                # Assuming one face per image
                if len(face_encodings) > 0:
                    embeddings.append(face_encodings[0])
                    labels.append(employee_name)
            else:
                print(f"No face detected in image: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Save embeddings to database
    if embeddings:
        for embedding, label in zip(embeddings, labels):
            # Check if the employee already exists in the database
            employee = session.query(Employee).filter_by(name=label).first()
            if not employee:
                employee = Employee(name=label)
                session.add(employee)
                session.commit()

            # Create a new Embedding record
            embedding_record = Embedding(
                embedding=embedding.tobytes(),
                employee_id=employee.id
            )
            session.add(embedding_record)
        session.commit()
    else:
        print(f"No embeddings found for employee {employee_name}.")

def add_new_employee(employee_name):
    base_dir = 'E:\\FACE Detection\\employee_images'
    save_dir = os.path.join(base_dir, employee_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    directions = ["Straight", "Left", "Right", "Up", "Down"]
    frame_rate = 5  # Extract one frame every 5 frames
    frame_count = 0
    direction_index = 0

    while direction_index < len(directions):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the current direction on the frame
        cv2.putText(frame, f"Move your head: {directions[direction_index]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capture Frames', frame)
        
        # Extract and save the frame at regular intervals
        if frame_count % frame_rate == 0:
            img_path = os.path.join(save_dir, f"{employee_name}_{directions[direction_index].lower()}_{frame_count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured frame {frame_count} for {directions[direction_index]} direction.")
        
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the capture
            break
        elif frame_count >= frame_rate * 20:  # Move to next direction after capturing enough frames
            frame_count = 0
            direction_index += 1
            if direction_index < len(directions):
                print(f"Now move your head: {directions[direction_index]}")
    
    cap.release()
    cv2.destroyAllWindows()

    # After capturing images, preprocess and extract embeddings
    preprocess_and_extract_embeddings(employee_name, save_dir)
    print(f"Employee {employee_name} added to the database.")

def view_existing_employees():
    employees = session.query(Employee).all()
    return employees

def delete_embeddings_for_employee(employee_name):
    employee = session.query(Employee).filter_by(name=employee_name).first()
    if employee:
        session.delete(employee)
        session.commit()
        print(f"Embeddings deleted for employee: {employee_name}")
    else:
        print(f"Employee '{employee_name}' not found in the database.")

        
def authenticate_employee_real_time():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture device.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations using face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Initialize lists for storing names, locations, and confidence scores
        names = []
        locations = []
        confidences = []

        # Loop through each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encodings with stored embeddings
            employees = session.query(Employee).all()
            best_match = None
            min_distance = 1.0  # Initialize to a value greater than possible distance

            for employee in employees:
                for embedding_record in employee.embeddings:
                    stored_embedding = np.frombuffer(embedding_record.embedding, dtype=np.float64)
                    distance = np.linalg.norm(face_encoding - stored_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = employee.name

            confidence = (1 - min_distance) * 100  # Convert distance to confidence percentage
            if confidence >= 60:
                names.append(best_match)
            else:
                names.append("Unknown")
            
            confidences.append(confidence)
            locations.append((top, right, bottom, left))

        # Display the results
        for (top, right, bottom, left), name, confidence in zip(locations, names, confidences):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f}%)", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the video feed
        cv2.imshow('Video', frame)

        # Break the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

