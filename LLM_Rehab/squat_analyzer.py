import time
from ultralytics import YOLO
import cv2
import math
import datetime
import numpy as np
import os
from collections import deque
import joint_feature_analysis
import openai
import json
import yaml
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    enable_segmentation=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)


model = YOLO('yolo11n-pose.pt')  # load YOLOv11 pose estimation model 

cap = cv2.VideoCapture(0)

left = False
fNum = 0


# MediaPipe Pose keypoint names (33 punkter)
kp = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]



#Draw the joints on the frame. In this case only left side is working
def draw_joints(frame, joints):
    global angle

    for joint in joints:                
        if not (joint[0] == 0 and joint[1] == 0):
            cv2.circle(frame, joint, 3, (0, 0, 0), 3)

    return frame


def draw_keypoints_mp(frame, keypoints):
    for kpinfo in keypoints.values():
        if kpinfo.get("visible"):
            cv2.circle(frame, kpinfo["coordinates"], 3, (0, 0, 0), 3)



#Request feedback on joint angle time series from the LLM
def request_feedback(feature_sequence):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    squat_prompt = prompts["personas"]["squat_expert2"]["prompt"]
    system_role = prompts["personas"]["squat_expert2"]["system_role"]

    # Skapa en prompt med ledsekvensen
    prompt = f"""
            {squat_prompt} 
            {json.dumps(feature_sequence, indent=2)}
            </time_series>
            """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ]
        )
        
        feedback = response.choices[0].message.content
        return feedback
        
    except Exception as e:
        return f"API call error: {str(e)}"    


def get_all_keypoints_mp(frame):
    h, w = frame.shape[:2]
    
    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    keypoints = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Rita alla 33 punkter
        if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for i, lm in enumerate(landmarks):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    x = max(0, min(w - 1, x))
                    y = max(0, min(h - 1, y))

                    name = kp[i] if 0 <= i < len(kp) else f"kp_{i}"
                    visible = getattr(lm, "visibility", 1.0) > 0.7

                    keypoints[name] = {
                        "coordinates": (x, y),
                        "x": x,
                        "y": y,
                        "visible": visible
                    }
            
    return keypoints  


#Run keypoint detection and extracts left shoulder, left hip, left knee and left ankle joints
def get_left_joints(frame):
    # Predict with the model
    results = model(frame)  # predict on an image
    
    for r in results:
        keypoints = r.keypoints.xy.int().numpy()
        joints = keypoints[0][5:17]  
        
        left_joints = [
            joints[0],  #left shoulder
            joints[6],  #left hip
            joints[8],  #left knee 
            joints[10]  #left ankle (vrist)
        ]
        
        return left_joints



#MAIN loop
joint_sequence = [] #For saving the joint angle time series
frame_count = 0
start_time = time.time()

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    
    if not ret:
        break
    
    #joints = get_left_joints(frame)
    #draw_joints(frame, joints)
    
    keypoints = get_all_keypoints_mp(frame)
    draw_keypoints_mp(frame, keypoints)
   
    #if(frame_count == 1):
    #    joint_feature_analysis.add_joint_features(joints, joint_sequence)
    #    print(joint_sequence)

    #if(frame_count == 2):
    #    frame_count = 0

    if(frame_count == 1):
        joint_feature_analysis.add_joint_features_mp(keypoints, joint_sequence)
        print(joint_sequence)
    
    if(frame_count == 2):
        frame_count = 0

    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_NEAREST)
    cv2.imshow('YoloV11-based Joints', frame)
        
    key = cv2.waitKey(1)
    
    if key == 27:
        break

    #Run the software for 15 seconds
    if time.time() - start_time >= 15:
        print("15 seconds have gone - closing motion capture and proceeding to Movement analysys. Please wait a few seconds...")
        break

cap.release()
cv2.destroyAllWindows() 

feedback = request_feedback(joint_sequence)
print(feedback)