# For veterinary privacy concerns, we cannot share the images in FRAME_PATH and YOLO_SAVE_BASE.
# You can run this code using your own video frames. Ensure that the frames are extracted beforehand.
# For assistance with extracting frames, refer to EPIA_code/extract_frames.py.

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import tensorflow as tf

# Paths and constants
detection_count_file = 'detection_counts.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = 'equine_asthma/feature_extraction_method/sam_vit_h_4b8939.pth'
YOLO_MODEL_PATH = '/cfs/home/u021554/tese/equine_asthma/best.pt'
FRAME_PATH = 'EPIA_code/feature_extraction_method/extracted_frames'
MASK_SAVE_BASE = 'EPIA_code/feature_extraction_method/saved_masks'
YOLO_SAVE_BASE = 'EPIA_code/feature_extraction_method/saved_yolo_output'

# Load SAM and YOLO models
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
model = YOLO(YOLO_MODEL_PATH)
model.to(DEVICE)

mask_predictor = SamPredictor(sam)

# Create directories for saving masks and YOLO outputs
os.makedirs(MASK_SAVE_BASE, exist_ok=True)
os.makedirs(YOLO_SAVE_BASE, exist_ok=True)

total_frames = 0
total_no_detection = 0

# Process each category in the frames directory
for category in os.listdir(FRAME_PATH): 
    if category == 'asthmatic':
        continue
    
    print('Processing category:', category)
    category_path = os.path.join(FRAME_PATH, category)
    category_mask_path = os.path.join(MASK_SAVE_BASE, category)
    category_YOLO_path = os.path.join(YOLO_SAVE_BASE, category)
    os.makedirs(category_mask_path, exist_ok=True)
    os.makedirs(category_YOLO_path, exist_ok=True)

    # Process each video in the category
    for video in os.listdir(category_path):
        video_path = os.path.join(category_path, video)
        video_mask_path = os.path.join(category_mask_path, video)
        video_YOLO_path = os.path.join(category_YOLO_path, video)
        os.makedirs(video_mask_path, exist_ok=True)
        os.makedirs(video_YOLO_path, exist_ok=True)
        
        no_detection_count = 0
        frame_count = 0

        # Process each frame in the video
        for frame in os.listdir(video_path):
            if not frame.endswith('.jpg'):
                continue
            
            frame_path = os.path.join(video_path, frame)
            image_bgr = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mask_predictor.set_image(image_rgb)

            # Perform inference on the image
            results = model(image_rgb)
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()

            if len(boxes) == 0:
                no_detection_count += 1
            frame_count += 1

            # Save YOLO results and masks
            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                output_path = os.path.join(video_YOLO_path, frame)
                cv2.imwrite(output_path, image_bgr)

                bbox = np.array(box, dtype=np.int32)
                masks, scores, logits = mask_predictor.predict(box=bbox, multimask_output=False)
                
                for mask in masks:
                    mask_image = (mask * 255).astype(np.uint8)
                    mask_save_path = os.path.join(video_mask_path, frame)
                    cv2.imwrite(mask_save_path, mask_image)

        percentage_no_detections = (no_detection_count / frame_count) * 100 
        total_frames += frame_count
        total_no_detection += no_detection_count
        overall_percentage = (total_no_detection / total_frames) * 100 if total_frames else 0

        # Write detection count to file
        with open(detection_count_file, 'a') as f:
            f.write(f'{video}: {no_detection_count} frames with no detections ({percentage_no_detections}% of frames)\n')

        print(f'Processed {video}: {no_detection_count} frames had no detections')

# Write overall detection percentage to file
with open(detection_count_file, 'a') as f:
    f.write(f'Total percentage of no detections: {overall_percentage}% for all videos.\n')
