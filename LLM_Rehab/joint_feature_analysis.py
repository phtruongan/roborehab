import numpy as np
import time
from datetime import datetime


def calculate_knee_flexion(hip, knee, ankle):

    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    vector_knee_to_hip = hip - knee      
    vector_knee_to_ankle = ankle - knee  
    
    dot_product = np.dot(vector_knee_to_hip, vector_knee_to_ankle)
    magnitude_thigh = np.linalg.norm(vector_knee_to_hip)
    magnitude_shank = np.linalg.norm(vector_knee_to_ankle)
    
    if magnitude_thigh == 0 or magnitude_shank == 0:
        return 0.0
    
    cos_angle = dot_product / (magnitude_thigh * magnitude_shank)
    
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    knee_flexion = 180.0 - angle_deg
    
    return knee_flexion



#Calculate and return torso lean angle based on shoulder and hip joint
def calculate_torso_lean_simple(shoulder, hip):
    
    shoulder = np.array(shoulder)
    hip = np.array(hip)
    
    dx = shoulder[0] - hip[0] 
    dy = shoulder[1] - hip[1]  

    angle_rad = np.arctan2(dx, -dy)
    angle_deg = np.degrees(angle_rad)
    
    return -angle_deg


#Based on joint coordinates calculate knee flexion angle and torso lean angle and add those
#features together with a time stamp to the time series
def add_joint_features(joint_coords, joint_sequence):
    shoulder_coords = joint_coords[0]
    hip_coords = joint_coords[1]
    knee_coords = joint_coords[2]
    ankle_coords = joint_coords[3]
    
    knee_angle = calculate_knee_flexion(hip_coords, knee_coords, ankle_coords)
    torso_angle = calculate_torso_lean_simple(shoulder_coords, hip_coords)
    timestamp = datetime.now().isoformat()
    
    joint_angles = {
        "timestamp": timestamp,
        "knee_angle": round(knee_angle, 2),
        "torso_angle": round(torso_angle, 2)
        #"coordinates": {
        #    "knee": knee_coords.tolist(),
        #    "ankle": ankle_coords.tolist(),
        #    "hip": hip_coords.tolist(),
        #    "shoulder": shoulder_coords.tolist()
        #}
    }
    
    joint_sequence.append(joint_angles)
    
    return joint_angles