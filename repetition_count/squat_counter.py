import time
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque


#Threshold value for the knee angle (angle between hip-knee and knee-ankle line. 0 degreees means straigt legg)
angle_threshold = 90

#Threshold value for the slope angle (the slope angle of the hip-knee line. 0 degrees meaans vertical. > 90 degrees means that the hip joint is lower than the knee joint)
slope_threshold = 90

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

def calculate_knee_angle(hip, knee, ankle):
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)
    
    # Vector from knee to hip
    vector1 = hip - knee
    # Vector from knee to ankle
    vector2 = ankle - knee
    
    # Calculate angle between vectors using dot product
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Convert so that straight leg = 0 degrees
    knee_angle = 180.0 - angle_deg
    
    return knee_angle

def check_hip_depth(hip, knee):
    """
    Check if hip is lower than or equal to knee (proper squat depth)
    Returns True if hip y-coordinate >= knee y-coordinate (lower in image coordinates)
    """
    hip = np.array(hip)
    knee = np.array(knee)
    
    # In image coordinates, y increases downward
    # So hip_y >= knee_y means hip is lower than knee
    return hip[1] >= knee[1]

def calculate_slope_angle(hip, knee):
    """
    Calculate the slope angle of the line between hip and knee
    Vertical line = 0 degrees, Horizontal line = 90 degrees
    """
    hip = np.array(hip)
    knee = np.array(knee)
    
    # Calculate differences
    dx = knee[0] - hip[0]
    dy = knee[1] - hip[1]
    
    # Avoid division by zero
    if dx == 0 and dy == 0:
        return 0
    
    # Calculate angle from vertical (0 degrees = vertical, 90 degrees = horizontal)
    # atan2 gives angle from horizontal, so we convert
    angle_from_horizontal = np.degrees(np.arctan2(abs(dy), abs(dx)))
    
    # Convert to angle from vertical
    slope_angle = 90.0 - angle_from_horizontal
    
    return abs(slope_angle)

def squat_counter(knee_angle, slope_angle, hip_below_knee, angle_buffer, squat_counts, in_squat):
    angle_buffer.append(knee_angle)
    
    # Only check if we have enough frames
    if len(angle_buffer) == 1:
        mean_angle = np.mean(angle_buffer)
        
        # Detect proper squat position: knee angle >= 90 AND (hip below knee OR slope angle >= 90)
        proper_depth = mean_angle >= angle_threshold and (hip_below_knee or slope_angle >= slope_threshold)
        
        if proper_depth and not in_squat:
            in_squat = True
        
        # Reset when standing up (leg almost straight)
        elif mean_angle < 20 and in_squat:
            squat_counts += 1
            in_squat = False
    
    return squat_counts, in_squat


#MAIN loop
knee_angles = []  # For saving the joint angle time series
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
    
    knee_angle = calculate_knee_angle(joints[1], joints[2], joints[3])
    knee_angles.append(knee_angle)
    
    # Calculate slope angle and check hip depth
    slope_angle = calculate_slope_angle(joints[1], joints[2])
    hip_below_knee = check_hip_depth(joints[1], joints[2])
    
    squat_counts, in_squat = squat_counter(knee_angle, slope_angle, hip_below_knee, angle_buffer, squat_counts, in_squat)
    
    # Determine if current position is proper depth
    proper_depth = knee_angle >= angle_threshold and (hip_below_knee or slope_angle >= slope_threshold)
    depth_color = (0, 255, 0) if proper_depth else (0, 0, 255)
    depth_text = "PROPER DEPTH" if proper_depth else "NOT DEEP ENOUGH"
    
    # Display squat count on frame
    cv2.putText(frame, f"Squats: {squat_counts}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Knee Angle: {knee_angle:.1f} | Slope: {slope_angle:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Hip Below Knee: {hip_below_knee}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, depth_text, (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, depth_color, 2)

    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_NEAREST)
    cv2.imshow('YoloV11-based Joints', frame)
        
    key = cv2.waitKey(1)
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()