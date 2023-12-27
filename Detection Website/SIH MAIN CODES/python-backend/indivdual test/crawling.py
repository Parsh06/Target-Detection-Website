import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# cap = cv2.VideoCapture(0)



#for internal web cam use (0)
#for external web cam // virtual machine//  WSL use (1)
#for user input of video add the relative path of your video inside quotes (' ' ) instead of zero. 
#example videos are provided in media folder
cap = cv2.VideoCapture('Media\crawling_vid.mp4') 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = holistic.process(image)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.pose_landmarks is not None:
            pose_landmarks = results.pose_landmarks
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])
            mean_position = np.mean(landmarks_array[:, :2], axis=0)
            distances = np.linalg.norm(landmarks_array[:, :2] - mean_position, axis=1)
            all_on_same_line = np.all(distances < 0.4)
            if all_on_same_line:
                cv2.putText(image, "Crawling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Crawling Detected")
            else:
                print("No Crawling Detected")
        else:
            print("no human detected")
        cv2.imshow('crawling', image)

        if cv2.waitKey(10) & 0xFF == ord('K'):
            break

cap.release()
cv2.destroyAllWindows()
