import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    enable_segmentation=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# MediaPipe Pose keypoint names (33 punkter)
POSE_LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Konvertera till RGB för MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    h, w = frame.shape[:2]
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Rita alla 33 punkter
        for i, landmark in enumerate(landmarks):
            # Konvertera normaliserade koordinater till pixlar
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Kontrollera synlighet (MediaPipe ger visibility score)
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            
            if visibility > 0.5:  # Rita bara synliga punkter
                # Olika färger för olika kroppsdelar
                if i < 11:  # Ansikte
                    color = (255, 0, 0)  # Blå
                elif i < 23:  # Armar/händer
                    color = (0, 255, 0)  # Grön
                elif i < 29:  # Ben/höfter
                    color = (0, 0, 255)  # Röd
                else:  # Fötter
                    color = (255, 255, 0)  # Cyan
                
                # Rita punkt
                cv2.circle(frame, (x, y), 4, color, -1)
                
                # Rita punkt-nummer (valfritt, kan bli rörigt)
                cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Rita skelett-linjer (MediaPipe har inbyggd funktion för detta)
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        # Visa fotvinkel för vänster fot (exempel)
        left_heel = landmarks[29]
        left_foot_index = landmarks[31]
        
        if left_heel.visibility > 0.5 and left_foot_index.visibility > 0.5:
            heel_x, heel_y = int(left_heel.x * w), int(left_heel.y * h)
            foot_x, foot_y = int(left_foot_index.x * w), int(left_foot_index.y * h)
            
            # Beräkna fotvinkel
            import math
            vx = foot_x - heel_x
            vy = foot_y - heel_y
            angle_deg = math.degrees(math.atan2(vy, vx))
            
            cv2.putText(frame, f"L foot angle: {angle_deg:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Rita fotlinje
            cv2.line(frame, (heel_x, heel_y), (foot_x, foot_y), (255, 255, 0), 2)
    
    # Visa punkt-legend (valfritt)
    cv2.putText(frame, "Blue=Face, Green=Arms, Red=Legs, Cyan=Feet", (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("MediaPipe Pose - All 33 Keypoints", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()