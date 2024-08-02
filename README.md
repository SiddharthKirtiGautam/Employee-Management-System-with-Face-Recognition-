# Employee Management System with Face Recognition
Face Detection and embedding

Overview

This project is an Employee Management System that uses face recognition technology to handle employee data and authentication. It includes functionality for adding new employees, viewing existing employees, authenticating employees in real-time using a camera, and deleting employee embeddings. The system utilizes OpenCV, face_recognition, and SQLAlchemy to manage and process employee data.

Features

- Add New Employee: Capture multiple images of an employee from different directions, preprocess them, and save face embeddings to a database.
- View Existing Employees: List all employees stored in the database.
- Authenticate Employee: Real-time face recognition using a camera to verify employee identity.
- Delete Embeddings: Remove an employee's embeddings from the database.

Requirements

•	Python 3.x
•	`cv2` (OpenCV)
•	`numpy`
•	`Pillow`
•	`face_recognition`
•	`SQLAlchemy`
•	`sqlite3` (included with Python standard library)

You can install the required Python packages using pip:
pip install opencv-python-headless numpy Pillow face_recognition sqlalchemy


Setup

1. Clone the Repository:
2. Create and Activate a Virtual Environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the Required Packages:
   pip install -r requirements.txt
4. Run the Application:
   python app.py

Usage

1. Add a New Employee:
•	Follow the on-screen prompts to enter the employee's name.
•	The system will guide you to capture images from different directions using your camera.
•	Images will be saved in a directory named after the employee and processed to extract face embeddings.

2. View Existing Employees:
•	Select this option to list all employees stored in the database.

3. Authenticate an Employee:
•	The system will start real-time authentication using the camera.
•	It will display the recognized employee's name and confidence score.

4. Delete Embeddings for an Employee:
•	Enter the employee's name to remove their embeddings from the database.

5. Exit:
•	Choose this option to close the application.

Database

The application uses SQLite for storing employee data and face embeddings. The database file is `employees.db`, and it will be created in the same directory as the script.

Code Structure

- `app.py`: Main script to run the application.
- `requirements.txt`: List of Python dependencies.
- `employee_images/`: Directory to save captured employee images.

Contributing
Feel free to open an issue or submit a pull request if you find any bugs or want to add new features. Contributions are welcome!

Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision functionalities.
- [face_recognition](https://github.com/ageitgey/face_recognition) for face recognition.
- [SQLAlchemy](https://www.sqlalchemy.org/) for ORM functionalities.


Feel free to modify the repository URL, license information, or any other details to better fit your project.
