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

model = YOLO('yolo11n-pose.pt')  # load YOLOv11 pose estimation model 

cap = cv2.VideoCapture(0)

left = False
fNum = 0


#Draw the joints on the frame. In this case only left side is working
def draw_joints(frame, joints):
    global angle

    for joint in joints:                
        cv2.circle(frame, joint, 3, (0, 0, 0), 3)

    return frame


#Request feedback on joint angle time series from the LLM
def request_feedback(feature_sequence):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    squat_prompt = prompts["personas"]["squat_expert"]["prompt"]
    system_role = prompts["personas"]["squat_expert"]["system_role"]

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
    
    joints = get_left_joints(frame)
    draw_joints(frame, joints)
    
    if(frame_count == 1):
        joint_feature_analysis.add_joint_features(joints, joint_sequence)
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