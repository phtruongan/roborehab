import time
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

model = YOLO('yolo11n-pose.pt')  # load YOLOv11 pose estimation model 
cap = cv2.VideoCapture(0)

#Draw the joints on the frame. In this case only left side is working
def draw_joints(frame, joints):
    global angle

    for joint in joints:                
        if not (joint[0] == 0 and joint[1] == 0):
            cv2.circle(frame, joint, 3, (0, 0, 0), 3)

    # Draw line between hip and knee
        shoulder = joints[0]
        hip = joints[1]
        knee = joints[2]
        ankle = joints[3]

    if not (hip[0] == 0 and hip[1] == 0) and not (knee[0] == 0 and knee[1] == 0):
        cv2.line(frame, hip, knee, (255, 0, 0), 2)  # Blue line

    if not (knee[0] == 0 and knee[1] == 0) and not (ankle[0] == 0 and ankle[1] == 0):
        cv2.line(frame, knee, ankle, (255, 0, 0), 2)  # Blue line

    if not (hip[0] == 0 and hip[1] == 0) and not (shoulder[0] == 0 and shoulder[1] == 0):
        cv2.line(frame, hip, shoulder, (255, 0, 0), 2)  # Blue line

    if not (knee[0] == 0 and knee[1] == 0) and not (ankle[0] == 0 and ankle[1] == 0):
        cv2.line(frame, knee, ankle, (255, 0, 0), 2)  # Blue line

    return frame

def get_left_joints(frame):
    # Predict with the model
    results = model(frame)  # predict on an image

    for r in results:
        keypoints = r.keypoints.xy.int().numpy()

        # Check if any keypoints were detected BEFORE accessing keypoints[0]
        if keypoints.shape[0] == 0:
            return None

        joints = keypoints[0][5:17]

        left_joints = [
            joints[0],  #left shoulder
            joints[6],  #left hip
            joints[8],  #left knee
            joints[10]  #left ankle (vrist)
        ]

        return left_joints
    return None

def calculate_thigh_angle(hip, knee):
    hip = np.array(hip)
    knee = np.array(knee)

    dx = abs(knee[0] - hip[0])  
    dy = knee[1] - hip[1]  
    
    if dy >= 0:
        angle_rad = np.arctan2(dx, dy)
        angle_deg = np.degrees(angle_rad)
    else:
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = 180.0 - np.degrees(angle_rad)
    
    return angle_deg

def squat_counter(thigh_angle, angle_buffer, squat_counts, in_squat):
    angle_buffer.append(thigh_angle)
    
    # Only check if we have enough frames
    if len(angle_buffer) == 1:
        mean_angle = np.mean(angle_buffer)
        
        # Detect squat position (thigh horizontal or beyond)
        if mean_angle >= 90 and not in_squat:
            in_squat = True
        
        # Reset when standing up (thigh almost vertical)
        elif mean_angle < 20 and in_squat:
            squat_counts += 1
            in_squat = False
    
    return squat_counts, in_squat


#MAIN loop
thigh_angles = []  # For saving the joint angle time series
angle_buffer = deque(maxlen=1)  # Buffer for last 5 frames
squat_counts = 0 #Total number of squats
in_squat = False  # Track if currently in squat position
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    joints = get_left_joints(frame)
    
    if joints is None:
        cv2.putText(frame, "No person detected", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_NEAREST)
        cv2.imshow('YoloV11-based Joints', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        continue
    
    draw_joints(frame, joints)
    
    thigh_angle = calculate_thigh_angle(joints[1], joints[2])
    thigh_angles.append(thigh_angle)
    
    squat_counts, in_squat = squat_counter(thigh_angle, angle_buffer, squat_counts, in_squat)
    
    # Display squat count on frame
    cv2.putText(frame, f"Squats: {squat_counts}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Thigh Angle: {thigh_angle:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_NEAREST)
    cv2.imshow('YoloV11-based Joints', frame)
        
    key = cv2.waitKey(1)
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()