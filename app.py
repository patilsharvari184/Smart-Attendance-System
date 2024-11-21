import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

# Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")

def datetoday2():
    return date.today().strftime("%d-%B-%Y")

# Paths
attendance_folder = 'E:/OneDrive/Desktop/Final proj/Attendance'
faces_folder = 'E:/OneDrive/Desktop/Final proj/Static/faces'
model_path = 'E:/OneDrive/Desktop/Final proj/Static/face_recognition_model.pkl'
haarcascade_path = 'E:/OneDrive/Desktop/Final proj/haarcascade_frontalface_default.xml'

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Ensure directories exist
os.makedirs(attendance_folder, exist_ok=True)
os.makedirs(faces_folder, exist_ok=True)
attendance_file = f'{attendance_folder}/Attendance-{datetoday()}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Get a number of total registered users
def totalreg():
    return len(os.listdir(faces_folder))

# Extract faces from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load(model_path)
    return model.predict(facearray)

# Train the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir(faces_folder)
    for user in userlist:
        user_path = os.path.join(faces_folder, user)
        for imgname in os.listdir(user_path):
            img = cv2.imread(os.path.join(user_path, imgname))
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, model_path)

# Extract info from today's attendance file
def extract_attendance():
    df = pd.read_csv(f'{attendance_folder}/Attendance-{datetoday()}.csv')
    if 'Time' not in df.columns:
        df['Time'] = ''  # Add 'Time' column if missing
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'{attendance_folder}/Attendance-{datetoday()}.csv')
    if int(userid) not in df['Roll'].astype(int).values:
        with open(f'{attendance_folder}/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

################## ROUTING FUNCTIONS #######################
####### for Face Recognition based Attendance System #######

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# Our main Face Recognition functionality
@app.route('/start', methods=['GET'])
def start():
    if not os.path.exists(model_path):
        return render_template('home.html', totalreg=totalreg(), mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', totalreg=totalreg(), mess='Error accessing webcam.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
            identified_person = identify_face(face)[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# A function to add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = os.path.join(faces_folder, f'{newusername}_{newuserid}')
    os.makedirs(userimagefolder, exist_ok=True)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', totalreg=totalreg(), mess='Error accessing webcam.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 50*5:
            break

        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
