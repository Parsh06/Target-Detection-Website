import cv2
import mediapipe as mp
import numpy as np
import winsound
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y = 0, 0, 0, 0, 0, 0, 0, 0
start_time = None
pTime = 0
cTime = 0
def start_timer():
    global start_time
    start_time = time.time()

def record_time(event,elapsed_time):
    print(f"{event} detected. Time stamp: {elapsed_time:.2f} seconds")

def detect_running_pose(pose_landmarks):
    left_knee = pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    left_leg_angle = math.degrees(math.atan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
    right_leg_angle = math.degrees(math.atan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

    running_threshold = 100

    return left_leg_angle > running_threshold and right_leg_angle > running_threshold

def detect_human_actions(frame):
    global prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is None:
        print("Object too small to be detected")
        return image
     
    if results.pose_landmarks:
            
            min_x = min([lm.x for lm in results.pose_landmarks.landmark])
            min_y = min([lm.y for lm in results.pose_landmarks.landmark])
            max_x = max([lm.x for lm in results.pose_landmarks.landmark])
            max_y = max([lm.y for lm in results.pose_landmarks.landmark])

            cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),
                          (int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 255, 0), 2)
    

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark

        nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_HIP]
        left_knee = pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
        right_knee = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
        left_ankle = pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
        right_ankle = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]
        left_ear = pose_landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]

        left_ear_x, left_ear_y = int(left_ear.x * image.shape[1]), int(left_ear.y * image.shape[0])
        right_ear_x, right_ear_y = int(right_ear.x * image.shape[1]), int(right_ear.y * image.shape[0])
        left_shoulder_x, left_shoulder_y = int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])
        right_shoulder_x, right_shoulder_y = int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])

        left_ear_dx = abs(left_ear_x - prev_left_ear_x)
        left_ear_dy = abs(left_ear_y - prev_left_ear_y)
        right_ear_dx = abs(right_ear_x - prev_right_ear_x)
        right_ear_dy = abs(right_ear_y - prev_right_ear_y)
        left_shoulder_dx = abs(left_shoulder_x - prev_left_shoulder_x)
        left_shoulder_dy = abs(left_shoulder_y - prev_left_shoulder_y)
        right_shoulder_dx = abs(right_shoulder_x - prev_right_shoulder_x)
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
          

    return image

if __name__ == "__main__":

    #for internal web cam use (0)
    #for external web cam // virtual machine//  WSL use (1)
    #for user input of video add the relative path of your video inside quotes (' ' ) instead of zero. 
    #example videos are provided in media folder
    cap = cv2.VideoCapture(0)
    
    start_timer()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_human_actions(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('K'):
            break
    cap.release()
    cv2.destroyAllWindows()
 