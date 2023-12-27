import cv2
import mediapipe as mp
import winsound

mp_drawing= mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#for internal web cam use (0)
#for external web cam // virtual machine//  WSL use (1)
#for user input of video add the relative path of your video inside quotes (' ' ) instead of zero. 
#example videos are provided in media folder
cap = cv2.VideoCapture(0)  


prev_left_ear_x, prev_left_ear_y,prev_right_ear_x,prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y= 0, 0, 0, 0, 0, 0, 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        
        landmarks = results.pose_landmarks.landmark

        
        left_ear = landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear= landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        
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


        jump_threshold = 30

        
        if left_ear_dy > jump_threshold and right_ear_dy> jump_threshold and left_shoulder_dy > jump_threshold and right_shoulder_dy > jump_threshold :
            cv2.putText( image,"Jump detected",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            print("Jump detected")
            duration = 1000  
            freq = 440 
            winsound.Beep(freq, duration)
        
        prev_left_ear_x, prev_left_ear_y = left_ear_x, left_ear_y
        prev_right_ear_x, prev_right_ear_y= right_ear_x,right_ear_y
        prev_left_shoulder_x, prev_left_shoulder_y = left_shoulder_x, left_shoulder_y
        prev_right_shoulder_x, prev_right_shoulder_y = right_shoulder_x, right_shoulder_y

    cv2.imshow("Jump Detection", image)
    if cv2.waitKey(10) & 0xFF == ord("K"):
        break
cap.release()
cv2.destroyAllWindows()

