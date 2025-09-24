import cv2
from ultralytics import YOLO
import numpy as np

# Ladda YOLOv11 pose model
model = YOLO('yolo11n-pose.pt')

# YOLOv11 Pose keypoint names (17 punkter - COCO format)
YOLO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# YOLO pose connections (skelett-linjer)
YOLO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Ansikte
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Armar
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Ben
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # YOLO prediction
    results = model(frame)
    
    h, w = frame.shape[:2]
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # Hämta koordinater och konfidenser
            keypoints = r.keypoints.xy[0].cpu().numpy()  # (17, 2) array
            confidences = r.keypoints.conf[0].cpu().numpy()  # (17,) array
            
            # Rita alla 17 punkter
            for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                x, y = int(kp[0]), int(kp[1])
                
                if conf > 0.5:  # Rita bara punkter med hög konfidans
                    # Olika färger för olika kroppsdelar
                    if i < 5:  # Ansikte
                        color = (255, 0, 0)  # Blå
                    elif i < 11:  # Armar/händer
                        color = (0, 255, 0)  # Grön
                    else:  # Ben/höfter
                        color = (0, 0, 255)  # Röd
                    
                    # Rita punkt
                    cv2.circle(frame, (x, y), 4, color, -1)
                    
                    # Rita punkt-nummer och konfidans
                    cv2.putText(frame, f"{i}({conf:.2f})", (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Rita skelett-linjer
            for connection in YOLO_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                
                # Kontrollera att båda punkterna har tillräcklig konfidans
                if (confidences[pt1_idx] > 0.5 and confidences[pt2_idx] > 0.5):
                    pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                    pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                    
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Exempel: Beräkna knävinkel (vänster ben)
            # Punkter: höft(11), knä(13), ankel(15)
            if (confidences[11] > 0.5 and confidences[13] > 0.5 and confidences[15] > 0.5):
                hip = keypoints[11]
                knee = keypoints[13] 
                ankle = keypoints[15]
                
                # Vektorer
                v1 = hip - knee  # knä -> höft
                v2 = ankle - knee  # knä -> ankel
                
                # Beräkna vinkel
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Säkerhet
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                cv2.putText(frame, f"L knee angle: {angle_deg:.1f}°", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Rita vinkellinjerna
                knee_pt = (int(knee[0]), int(knee[1]))
                hip_pt = (int(hip[0]), int(hip[1]))
                ankle_pt = (int(ankle[0]), int(ankle[1]))
                cv2.line(frame, knee_pt, hip_pt, (255, 255, 0), 2)
                cv2.line(frame, knee_pt, ankle_pt, (255, 255, 0), 2)
    
    # Visa punkt-legend
    cv2.putText(frame, "Blue=Face, Green=Arms, Red=Legs", (10, h-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "YOLOv11 Pose - 17 Keypoints", (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("YOLOv11 Pose - All 17 Keypoints", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()