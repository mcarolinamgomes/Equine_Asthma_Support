# python3 inference.py /cfs/home/u021554/tese/SAM/concat/full_dataset/abdomen/asthmatic/horse22.mp4 /cfs/home/u021554/tese/SAM/concat/full_dataset/nostrils/asthmatic/horse22.mp4
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['CUDA_HOME'] = '/cfs/home/u021554/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn' 
os.environ['LD_LIBRARY_PATH'] = '~/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 0 is visible to this program
import argparse
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import csv

# Set the destination directory for extracted frames
destination_directory = 'extracted_frames'
csv_dir = 'new_extracted_features'

def extract_frames(video_path,body_part):
    video_file = os.path.basename(video_path)
    video_name = os.path.splitext(video_file)[0]
    video_frames_dir = os.path.join(destination_directory, video_name, body_part)
    
    if os.path.exists(video_frames_dir):
        return video_frames_dir
    
    os.makedirs(video_frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Error opening video file {video_file}')
        return None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(video_frames_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break

    cap.release()
    print(f"Frame extraction for {video_name} is complete.")
    return video_frames_dir


def process_frame(frame_path, mask_generator, mask_dir, device):
    try:
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
        binary_mask = (largest_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_dir, os.path.basename(frame_path)), binary_mask)
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}")
    finally:
        # Clear GPU cache
        torch.cuda.empty_cache()

def abdomen_segmentation_mask(video_frames_dir):
    sam_checkpoint = "/equine_asthma/feature_extraction_method/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    mask_dir = os.path.join(video_frames_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    frame_paths = [os.path.join(video_frames_dir, frame) for frame in os.listdir(video_frames_dir)]
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_frame, frame_path, mask_generator, mask_dir, device) for frame_path in frame_paths]
        for future in as_completed(futures):
            future.result()  # wait for all futures to complete

    return mask_dir

def extract_number(base):
    parts = base.split('_')
    if len(parts) > 1:
        numbers = ''.join(filter(str.isdigit, parts[1]))
        return int(numbers) if numbers.isdigit() else 0
    return 0

def calculate_area(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Failed to load image at {image_path}")
        return 0
    return np.sum(image > 0)

def rms_change(areas):
    changes = np.diff(areas)
    squared_changes = np.square(changes)
    return np.sqrt(np.mean(squared_changes))

def process_mask(csv_file_path, mask_dir):
    print('mask_dir',mask_dir)
    with open(csv_file_path, mode='w', newline='') as file:
        print('opened csv')
        writer = csv.writer(file)
        writer.writerow(['Mean Relative Area', 'Area StdDev', 'Coefficient of Variation', 'Range', 'Interquartile Range', 'Variance', 'RMS Change'])

        mask_files = [f for f in os.listdir(mask_dir)]
        mask_files.sort(key=extract_number)
        for i in range(0, len(mask_files), 180):
            batch_files = mask_files[i:i+180]
            areas = [calculate_area(os.path.join(mask_dir, f)) for f in batch_files]
            if areas:
                baseline_area = areas[0]
                relative_areas = [area / baseline_area for area in areas]
                mean_relative_area = np.mean(relative_areas)
                std_dev = np.std(relative_areas)
                cv = std_dev / mean_relative_area if mean_relative_area else 0
                data_range = np.ptp(relative_areas)
                iqr_value = iqr(relative_areas)
                variance = np.var(relative_areas)
                rms_change_value = rms_change(relative_areas)
                writer.writerow([mean_relative_area, std_dev, cv, data_range, iqr_value, variance, rms_change_value])

    print(f'Data saved to {csv_file_path}')

def classify(csv_path, svm_model_path, scaler_path, selector_path):
    svm_model = joblib.load(svm_model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    new_data = pd.read_csv(csv_path)
    feature_columns = ['Mean Relative Area', 'Area StdDev', 'Coefficient of Variation', 'Range', 'Interquartile Range', 'Variance', 'RMS Change']
    new_data_scaled = scaler.transform(new_data[feature_columns].values)
    new_data_selected = selector.transform(new_data_scaled)

    new_predictions = svm_model.predict(new_data_selected)
    print("Individual Predictions:", new_predictions)

    average_prediction = np.mean(new_predictions)
    print("Average Prediction:", average_prediction)

    final_prediction = 'asthmatic' if average_prediction > 0.5 else 'healthy'
    print("Final Prediction:", final_prediction)

def nostrils_segmentation_mask(video_frames_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_type = "vit_h"
    checkpoint_path = '/cfs/home/u021554/tese/SAM/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    model = YOLO("/cfs/home/u021554/tese/equine_asthma/best.pt")
    mask_predictor = SamPredictor(sam)

    for frame in os.listdir(video_frames_dir):
        frame_path = os.path.join(video_frames_dir, frame)
        if not frame.endswith('.png'):
            continue
        
        image_bgr = cv2.imread(frame_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_predictor.set_image(image_rgb)
        results = model(image_rgb)
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        video_YOLO_path = os.path.join(video_frames_dir, 'yolo')
        os.makedirs(video_YOLO_path, exist_ok=True)

        for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            output_path = os.path.join(video_YOLO_path, frame)
            cv2.imwrite(output_path, image_bgr)

            bbox = np.array(box, dtype=np.int32)
            masks, scores, logits = mask_predictor.predict(box=bbox, multimask_output=False)

            mask_dir = os.path.join(video_frames_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            
            for mask_idx, mask in enumerate(masks):
                mask_image = (mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, frame), mask_image)
    
    return mask_dir

def concat_abdomen(video_frames_dir, csv_path):
    mask_dir = os.path.join(video_frames_dir, 'mask')
    print('mask_dir',mask_dir)
    if not os.path.exists(mask_dir):
        print('lets do segmentation')
        mask_dir = abdomen_segmentation_mask(video_frames_dir)
    print('segmentation done, lets process mask')
    process_mask(csv_path, mask_dir)
    print('mask has been processed')
    classify(csv_path, '/cfs/home/u021554/tese/SAM/final/best_model_iteration_abdomen.pkl', '/cfs/home/u021554/tese/SAM/final/best_scaler_iteration_abdomen.pkl', '/cfs/home/u021554/tese/SAM/final/best_selector_iteration_abdomen.pkl')

def concat_nostrils(video_frames_dir, csv_path):
    mask_dir = os.path.join(video_frames_dir, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = nostrils_segmentation_mask(video_frames_dir)
    process_mask(csv_path, mask_dir)
    classify(csv_path, '/cfs/home/u021554/tese/SAM/final/best_model_iteration_nostrils.pkl', '/cfs/home/u021554/tese/SAM/final/best_scaler_iteration_nostrils.pkl', '/cfs/home/u021554/tese/SAM/final/best_selector_iteration_nostrils.pkl')

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Process video files.')
    parser.add_argument('video_path_abdomen', type=str, help='Full path to the video file of the abdomen.')
    parser.add_argument('video_path_nostrils', type=str, help='Full path to the video file of the nostrils.')
    parser.add_argument('--body_part', type=str, choices=['abdomen', 'nostrils'], help='Body part indication if only one video path is provided.')

    args = parser.parse_args()

    if args.body_part:
        video_path = args.video_path_abdomen if args.body_part == 'abdomen' else args.video_path_nostrils
        video_frames_dir = extract_frames(video_path,args.body_part)
        csv_path = os.path.join(video_frames_dir, 'processed_mask.csv')
        if args.body_part == 'abdomen':
            concat_abdomen(video_frames_dir, csv_path)
        else:
            concat_nostrils(video_frames_dir, csv_path)
    else:
        video_frames_dir_abdomen = extract_frames(args.video_path_abdomen,'abdomen')
        video_frames_dir_nostrils = extract_frames(args.video_path_nostrils,'nostrils')
        csv_path_abdomen = os.path.join(csv_dir, 'abdomen_processed_mask.csv')
        csv_path_nostrils = os.path.join(csv_dir, 'nostrils_processed_mask.csv')
        concat_abdomen(video_frames_dir_abdomen, csv_path_abdomen)
        concat_nostrils(video_frames_dir_nostrils, csv_path_nostrils)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total inference time: {elapsed_time:.2f} seconds")

