import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
destination_directory = 'frames'
csv_dir = 'features'

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

def abdomen_segmentation_mask(video_frames_dir):
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
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

    for frame in os.listdir(video_frames_dir):
        frame_path = os.path.join(video_frames_dir, frame)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
        binary_mask = (largest_mask > 0).astype(np.uint8) * 255

        mask_dir = os.path.join(video_frames_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(mask_dir, frame), binary_mask)

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
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    
    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
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

def classify_with_random_forest(csv_path, rf_model_path, scaler_path, selector_path):
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    #new_data_selected = preprocess_data(csv_path, scaler_path, selector_path)

    new_data = pd.read_csv(csv_path)
    feature_columns = ['Mean Relative Area','Area StdDev','Coefficient of Variation','Range','Interquartile Range','Variance','RMS Change','Mean Relative Area_from_data2','Area StdDev_from_data2','Coefficient of Variation_from_data2','Range_from_data2','Interquartile Range_from_data2','Variance_from_data2','RMS Change_from_data2']
    new_data_scaled = scaler.transform(new_data[feature_columns].values)
    new_data_selected = selector.transform(new_data_scaled)

    new_predictions = rf_model.predict(new_data_selected)
    print("Individual Predictions (Random Forest):", new_predictions)

    average_prediction = np.mean(new_predictions)
    print("Average Prediction (Random Forest):", average_prediction)

    final_prediction = 'asthmatic' if average_prediction > 0.5 else 'healthy'
    print("Final Prediction (Random Forest):", final_prediction)

def classify_with_decision_tree(csv_path, dt_model_path, scaler_path, selector_path):
    dt_model = joblib.load(dt_model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    new_data = pd.read_csv(csv_path)
    feature_columns = ['Mean Relative Area', 'Area StdDev', 'Coefficient of Variation', 'Range', 'Interquartile Range', 'Variance', 'RMS Change']
    new_data_scaled = scaler.transform(new_data[feature_columns].values)
    new_data_selected = selector.transform(new_data_scaled)

    
    new_predictions = dt_model.predict(new_data_selected)
    print("Individual Predictions (Decision Tree):", new_predictions)

    average_prediction = np.mean(new_predictions)
    print("Average Prediction (Decision Tree):", average_prediction)

    final_prediction = 'asthmatic' if average_prediction > 0.5 else 'healthy'
    print("Final Prediction (Decision Tree):", final_prediction)

def classify_with_svm(csv_path, svm_model_path, scaler_path, selector_path):
    svm_model = joblib.load(svm_model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    
    new_data = pd.read_csv(csv_path)
    feature_columns = ['Mean Relative Area', 'Area StdDev', 'Coefficient of Variation', 'Range', 'Interquartile Range', 'Variance', 'RMS Change']
    new_data_scaled = scaler.transform(new_data[feature_columns].values)
    new_data_selected = selector.transform(new_data_scaled)

    
    new_predictions = svm_model.predict(new_data_selected)
    print("Individual Predictions (SVM):", new_predictions)

    average_prediction = np.mean(new_predictions)
    print("Average Prediction (SVM):", average_prediction)

    final_prediction = 'asthmatic' if average_prediction > 0.5 else 'healthy'
    print("Final Prediction (SVM):", final_prediction)


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
    checkpoint_path = '../sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    model = YOLO('../../nostrils_detector.pt')
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
    if not os.path.exists(mask_dir):
        mask_dir = abdomen_segmentation_mask(video_frames_dir)
    process_mask(csv_path, mask_dir)
    classify_with_decision_tree(csv_path, 'best_model_iteration_abdomen.pkl', 'best_scaler_iteration_abdomen.pkl', 'best_selector_iteration_abdomen.pkl')

def concat_nostrils(video_frames_dir, csv_path):
    mask_dir = os.path.join(video_frames_dir, 'mask')
    if not os.path.exists(mask_dir):
        mask_dir = nostrils_segmentation_mask(video_frames_dir)
    process_mask(csv_path, mask_dir)
    classify_with_svm(csv_path, 'best_model_iteration_nostrils.pkl', 'best_scaler_iteration_nostrils.pkl', 'best_selector_iteration_nostrils.pkl')

def concat_both(video_frames_dir_abdomen, csv_path_abdomen, video_frames_dir_nostrils, csv_path_nostrils, csv_path_both):
    mask_dir_abdomen = os.path.join(video_frames_dir_abdomen, 'mask')
    if not os.path.exists(mask_dir_abdomen):
        abdomen_segmentation_mask(video_frames_dir_abdomen)
    process_mask(csv_path_abdomen, mask_dir_abdomen)

    mask_dir_nostrils = os.path.join(video_frames_dir_nostrils, 'mask')
    if not os.path.exists(mask_dir_nostrils):
        nostrils_segmentation_mask(video_frames_dir_nostrils)
    process_mask(csv_path_nostrils, mask_dir_nostrils)

    merge_data(csv_path_abdomen, csv_path_nostrils, csv_path_both)

    classify_with_random_forest(csv_path_both, 'best_model_iteration_both.pkl', 'best_scaler_iteration_both.pkl', 'best_selector_iteration_both.pkl')

def merge_data(csv_file_path1, csv_file_path2, output_csv_file_path):
    """Merge two CSV files and save the result."""

    if os.path.exists(output_csv_file_path):
        os.remove(output_csv_file_path)
    
    data1 = pd.read_csv(csv_file_path1)
    data2 = pd.read_csv(csv_file_path2)

    # Initialize a list to hold the merged rows
    merged_data = []
    
    # Get the minimum number of rows from both dataframes
    min_length = min(len(data1), len(data2))
    
    # Rename columns in data2 to ensure they are unique
    data2.columns = [f"{col}_from_data2" if col in data1.columns else col for col in data2.columns]
    
    # Merge line by line until one dataframe is exhausted
    for i in range(min_length):
        merged_row = pd.concat([data1.iloc[i], data2.iloc[i]], axis=0)
        merged_data.append(merged_row)
    
    # Convert the list of Series to a DataFrame
    merged_data = pd.DataFrame(merged_data)
    
    # Save the merged data to a new CSV file
    merged_data.to_csv(output_csv_file_path, index=False)
    print(f'Merged data saved to {output_csv_file_path}')


if __name__ == '__main__':

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Process video files.')
    parser.add_argument('video_paths', type=str, nargs='+', help='Paths to the video files. Provide one or two paths.')
    parser.add_argument('--body_part', type=str, choices=['abdomen', 'nostrils'], help='Body part indication if only one video path is provided.')

    args = parser.parse_args()

    if len(args.video_paths) == 1:
        print('len = 1')
        if not args.body_part:
            print("Error: --body_part must be specified when providing only one video path.")
        else:
            video_path = args.video_paths[0]
            video_frames_dir = extract_frames(video_path,args.body_part)
            if args.body_part == 'abdomen':
                print("Starting processing for abdomen video...")
                csv_path_abdomen = os.path.join(csv_dir, 'abdomen_processed_mask.csv')
                concat_abdomen(video_frames_dir, csv_path_abdomen)
            elif args.body_part == 'nostrils':
                print("Starting processing for nostrils video...")
                csv_path_nostrils = os.path.join(csv_dir, 'nostrils_processed_mask.csv')
                concat_nostrils(video_frames_dir, csv_path_nostrils)
    elif len(args.video_paths) == 2:
        print('len = 2')
        video_path_abdomen = args.video_paths[0]
        video_path_nostrils = args.video_paths[1]
        print("Starting processing both videos...")
        video_frames_dir_abdomen = extract_frames(video_path_abdomen,'abdomen')
        video_frames_dir_nostrils = extract_frames(video_path_nostrils,'nostrils')
        csv_path_abdomen = os.path.join(csv_dir, 'abdomen_processed_mask.csv')
        csv_path_nostrils = os.path.join(csv_dir, 'nostrils_processed_mask.csv')
        csv_path_both = os.path.join(csv_dir, 'both_processed_mask.csv')
        concat_both(video_frames_dir_abdomen, csv_path_abdomen, video_frames_dir_nostrils, csv_path_nostrils, csv_path_both)
    else:
        print("Error: Provide one or two video paths.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total inference time: {elapsed_time:.2f} seconds")
