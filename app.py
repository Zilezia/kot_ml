import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time
import pandas as pd

def app():
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model = torch.hub.load(
        "ultralytics/yolov5", 
        "custom", 
        path='yolov5/runs/train/exp14/weights/last.pt',
        force_reload=True
    )

    # IMAGES_PATH = os.path.join('data', 'images')
    # labels = ['neutral', 'happy']
    # number_imgs = 20

    video = cv2.VideoCapture(0)
    while video.isOpened():
        ok, frame = video.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        results = model(frame)

        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, confidence, class_id = det[:6]
            
            # print(f"Detection: Class ID: {int(class_id)}, Confidence: {confidence.item()}")
            # class id start: 15
            
            if confidence > 0.7:
                if int(class_id) == 15:
                    print('kot, confidence:', confidence.item())
                
        cv2.imshow('kot_ml', np.squeeze(results.render()))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    app()