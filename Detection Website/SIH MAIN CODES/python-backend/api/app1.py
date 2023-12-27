from flask import Flask,Response, render_template, request, redirect, url_for
import cv2
from flask_socketio import SocketIO
import mediapipe as mp
import numpy as np
import winsound
import math
import time
import threading 
import csv
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3

# min_x, min_y, max_x, max_y = 0,0,0,0
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
# prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y = 0, 0, 0, 0, 0, 0, 0, 0
# start_time = time.time()
# pTime = 0
# cTime = 0



# app = Flask(__name__,template_folder = 'Templates',static_url_path='/static/', static_folder='static/')

# # SQLite database initialization
# def create_db():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)''')
#     conn.commit()
#     conn.close()

# create_db()



# def start_timer():
#     global start_time
#     start_time = time.time()

# def record_time(event,elapsed_time):

#     statement = (f"{event} detected. Time stamp: {elapsed_time:.2f} seconds")
#     print(statement)

#     csv_file = "timestamp.csv"

#     with open(csv_file, mode ='a', newline='') as file:
      
#       writer = csv.writer(file)

#       writer.writerow([statement])

# def detect_running_pose(pose_landmarks):
#     left_knee = pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
#     right_knee = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
#     left_ankle = pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
#     right_ankle = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

#     left_leg_angle = math.degrees(math.atan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
#     right_leg_angle = math.degrees(math.atan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

#     running_threshold = 100

#     return left_leg_angle > running_threshold and right_leg_angle > running_threshold
#     # Your existing function

# def detect_human_actions(frame):
#     global prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y
#     global start_time
#     mp_holistic = mp.solutions.holistic
#     holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     if results.pose_landmarks:
#         min_x = min([lm.x for lm in results.pose_landmarks.landmark])
#         min_y = min([lm.y for lm in results.pose_landmarks.landmark])
#         max_x = max([lm.x for lm in results.pose_landmarks.landmark])
#         max_y = max([lm.y for lm in results.pose_landmarks.landmark])

        

#     if results.pose_landmarks:
#         pose_landmarks = results.pose_landmarks.landmark
#         #nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
#         left_shoulder = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
#         left_ear = pose_landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
#         right_ear = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]

#         left_ear_x, left_ear_y = int(left_ear.x * image.shape[1]), int(left_ear.y * image.shape[0])
#         right_ear_x, right_ear_y = int(right_ear.x * image.shape[1]), int(right_ear.y * image.shape[0])
#         left_shoulder_x, left_shoulder_y = int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])
#         right_shoulder_x, right_shoulder_y = int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])

#         #left_ear_dx = abs(left_ear_x - prev_left_ear_x)
#         left_ear_dy = abs(left_ear_y - prev_left_ear_y)
#         #right_ear_dx = abs(right_ear_x - prev_right_ear_x)
#         right_ear_dy = abs(right_ear_y - prev_right_ear_y)
#         #left_shoulder_dx = abs(left_shoulder_x - prev_left_shoulder_x)
#         left_shoulder_dy = abs(left_shoulder_y - prev_left_shoulder_y)
#         #right_shoulder_dx = abs(right_shoulder_x - prev_right_shoulder_x)
#         right_shoulder_dy = abs(right_shoulder_y - prev_right_shoulder_y)

#         jump_threshold = 60

#         if left_ear_dy > jump_threshold and right_ear_dy> jump_threshold and left_shoulder_dy > jump_threshold and right_shoulder_dy > jump_threshold :
#             elapsed_time = time.time() - start_time
#             record_time("Jumping", elapsed_time)
#             duration = 1000  
#             freq = 440  
#             winsound.Beep(freq, duration)
           

#         prev_left_ear_x, prev_left_ear_y = left_ear_x, left_ear_y
#         prev_right_ear_x, prev_right_ear_y = right_ear_x, right_ear_y
#         prev_left_shoulder_x, prev_left_shoulder_y = left_shoulder_x, left_shoulder_y
#         prev_right_shoulder_x, prev_right_shoulder_y = right_shoulder_x, right_shoulder_y
        
#         if detect_running_pose(pose_landmarks):
#             elapsed_time = time.time() - start_time
#             record_time("Running", elapsed_time)
       
#             duration = 1000 
#             freq = 1000 
#             winsound.Beep(freq, duration)

#         if results.pose_landmarks is not None:
#             pose_landmarks = results.pose_landmarks
#             landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])

#             mean_position = np.mean(landmarks_array[:, :2], axis=0)
#             distances = np.linalg.norm(landmarks_array[:, :2] - mean_position, axis=1)
#             all_on_same_line = np.all(distances < 0.1)

#             if all_on_same_line:
#                 elapsed_time = time.time() - start_time
#                 record_time("Crawling", elapsed_time)
#                 duration = 1000  
                
#                 freq = 600
#                 winsound.Beep(freq, duration)
#         # Your existing code for pose landmarks

# def process_frame(frame):
#     cv2.rectangle(frame, (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),
#                   (int(max_x * frame.shape[1]), int(max_y * frame.shape[0])), (0, 255, 0), 2)
#     detect_human_actions(frame)
#     ret, buffer = cv2.imencode('.jpg', frame)
#     return buffer.tobytes()

# def generate_frames():
#     cap = cv2.VideoCapture('SIH MAIN CODES/python-backend/api/static/Media/30vid.mp4')
#     executor = ThreadPoolExecutor(max_workers=2)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Submit the frames to the executor for concurrent processing
#         future1 = executor.submit(process_frame, frame.copy())
#         future2 = executor.submit(process_frame, frame.copy())

#         # Wait for the results
#         frame1_bytes = future1.result()
#         frame2_bytes = future2.result()

#         # Yield both frames
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame1_bytes + b'\r\n'
#                b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame2_bytes + b'\r\n')

min_x, min_y, max_x, max_y = 0,0,0,0
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y = 0, 0, 0, 0, 0, 0, 0, 0
start_time = time.time()
pTime = 0
cTime = 0
executor = ThreadPoolExecutor(max_workers=2)


app = Flask(__name__,template_folder = 'Templates',static_url_path='/static/', static_folder='static/')


def calculate_distance(focal_length, known_width, pixel_width):
    return (known_width * focal_length) / pixel_width / 39.37  


def start_timer():
    global start_time
    start_time = time.time()

def record_time(event,elapsed_time):

    statement = (f"{event} detected. Time stamp: {elapsed_time:.2f} seconds")
    print(statement)

    csv_file = "timestamp.csv"

    with open(csv_file, mode ='a', newline='') as file:
      
      writer = csv.writer(file)

      writer.writerow([statement])

def detect_running_pose(pose_landmarks):
    left_knee = pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    left_leg_angle = math.degrees(math.atan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
    right_leg_angle = math.degrees(math.atan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

    running_threshold = 100

    return left_leg_angle > running_threshold and right_leg_angle > running_threshold
    # Your existing function

def detect_human_actions(frame):
    global prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y
    global start_time
    #global min_x, min_y, max_x, max_y
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        min_x = min([lm.x for lm in results.pose_landmarks.landmark])
        min_y = min([lm.y for lm in results.pose_landmarks.landmark])
        max_x = max([lm.x for lm in results.pose_landmarks.landmark])
        max_y = max([lm.y for lm in results.pose_landmarks.landmark])

        cv2.rectangle(frame, (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),(int(max_x * frame.shape[1]), int(max_y * frame.shape[0])), (0, 255, 0), 2)

        

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        #nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_ear = pose_landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]

        left_ear_x, left_ear_y = int(left_ear.x * image.shape[1]), int(left_ear.y * image.shape[0])
        right_ear_x, right_ear_y = int(right_ear.x * image.shape[1]), int(right_ear.y * image.shape[0])
        left_shoulder_x, left_shoulder_y = int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])
        right_shoulder_x, right_shoulder_y = int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])

        #left_ear_dx = abs(left_ear_x - prev_left_ear_x)
        left_ear_dy = abs(left_ear_y - prev_left_ear_y)
        #right_ear_dx = abs(right_ear_x - prev_right_ear_x)
        right_ear_dy = abs(right_ear_y - prev_right_ear_y)
        #left_shoulder_dx = abs(left_shoulder_x - prev_left_shoulder_x)
        left_shoulder_dy = abs(left_shoulder_y - prev_left_shoulder_y)
        #right_shoulder_dx = abs(right_shoulder_x - prev_right_shoulder_x)
        right_shoulder_dy = abs(right_shoulder_y - prev_right_shoulder_y)

        jump_threshold = 60

        if left_ear_dy > jump_threshold and right_ear_dy> jump_threshold and left_shoulder_dy > jump_threshold and right_shoulder_dy > jump_threshold :
            elapsed_time = time.time() - start_time
            record_time("Jumping", elapsed_time)
            duration = 1000  
            freq = 440  
            winsound.Beep(freq, duration)
           

        prev_left_ear_x, prev_left_ear_y = left_ear_x, left_ear_y
        prev_right_ear_x, prev_right_ear_y = right_ear_x, right_ear_y
        prev_left_shoulder_x, prev_left_shoulder_y = left_shoulder_x, left_shoulder_y
        prev_right_shoulder_x, prev_right_shoulder_y = right_shoulder_x, right_shoulder_y
        
        if detect_running_pose(pose_landmarks):
            elapsed_time = time.time() - start_time
            record_time("Running", elapsed_time)
       
            duration = 1000 
            freq = 1000 
            winsound.Beep(freq, duration)

        if results.pose_landmarks is not None:
            pose_landmarks = results.pose_landmarks
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])

            mean_position = np.mean(landmarks_array[:, :2], axis=0)
            distances = np.linalg.norm(landmarks_array[:, :2] - mean_position, axis=1)
            all_on_same_line = np.all(distances < 0.1)

            if all_on_same_line:
                elapsed_time = time.time() - start_time
                record_time("Crawling", elapsed_time)
                duration = 1000  
                
                freq = 600
                winsound.Beep(freq, duration)
        # Your existing code for pose landmarks
def measure_distance(frame, focal_length, known_width):
    
    known_width = 6.0  

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate the pixel width of the detected face
        pixel_width = w

        # Calculate the distance using the known parameters
        distance = calculate_distance(focal_length, known_width, pixel_width)

        # Display the distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame
def generate_frames():
    
    focal_length = 500  

    known_width = 6.0  
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        future1 = executor.submit(process_frame, frame.copy(), focal_length, known_width)
        future2 = executor.submit(process_frame, frame.copy(), focal_length, known_width)

        frame1_bytes = future1.result()
        frame2_bytes = future2.result()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1_bytes + b'\r\n'
               b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2_bytes + b'\r\n')

# Modify process_frame to accept additional arguments

def process_frame(frame, focal_length, known_width):
    detect_human_actions(frame)
    frame_with_distance = measure_distance(frame.copy(), focal_length, known_width)
    ret, buffer = cv2.imencode('.jpg', frame_with_distance)
    return buffer.tobytes()

# app = Flask(__name__,template_folder = 'Templates')
def create_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

create_db()


@app.route('/')
def index6():
    return render_template('index.html')

@app.route('/signup.html', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        return redirect ('/login.html')  # Redirect to the 'login' route
    return render_template('signup.html')
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            return render_template('index2.html')
        else:
             return render_template('signup.html')

    return render_template('/login.html')
@app.route('/index2.html')
def index():
    return render_template('index2.html')

@app.route('/detection2.html')
def detection():
    return render_template('detection2.html')

@app.route('/video2.html')
def video1():
    return render_template('video2.html')

@app.route('/upload2.html')
def upload():
    return render_template('upload2.html')

@app.route('/information1.html')
def information1():
    return render_template('information1.html')

@app.route('/ppt2.html')
def ppt():
    return render_template('ppt2.html')

@app.route('/information1.html')
def information1_page():
    return render_template('information1.html')

@app.route('/information2.html')
def information2_page():
    return render_template('information2.html')

@app.route('/contact2.html', methods=['GET', 'POST'])
def contact2():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO details (name, email, message) VALUES (?, ?, ?)", (name, email, message))
        conn.commit()
        conn.close()

        return redirect('/response page2.html')  # Redirect to the 'index' route after saving details

    return render_template('contact2.html')

@app.route('/response page2.html')
def response():
    return render_template('response page2.html')
@app.route('/index2.html')
def index4():
    return render_template('index2.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":

    app.run(debug=True,threaded=True,port=5000)

#cap.release()
    # cv2.destroyAllWindows()