import os
os.environ['CUDA_HOME'] = '/cfs/home/u021554/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn' 
os.environ['LD_LIBRARY_PATH'] = '~/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Only GPU 0 is visible to this program
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Available GPUs:", gpus)
else:
    print("GPU is not available.")

import time
import argparse
import cv2
import re
from ultralytics import YOLO
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from inf_video import predict_frames_in_directory, predict_from_two_directories
import gc

# Set the destination directory for extracted frames
FRAMES_DIR = 'extracted_frames'
ROI_DIR = 'roi_frames'
SUBTRACTED_DIR = 'subtracted_frames'

# Ensure TensorFlow uses only one GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth for each GPU
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f'Set memory growth for GPUs: {physical_devices}')
    except Exception as e:
        print(f'Error setting memory growth: {e}')
else:
    print('No GPU devices found.')

def blurred_frame_differencing(frame1, frame2):
    blurred_frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    blurred_frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    return cv2.absdiff(blurred_frame1, blurred_frame2)

def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()

def process_video(video_path, extract_nostrils):
    print(f"Processing video at {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if extract_nostrils:
        output_dir = os.path.join(FRAMES_DIR, video_name, 'nostrils')
    else:
        output_dir = os.path.join(FRAMES_DIR, video_name, 'abdomen')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    count = 0
    frame_skip = 5  # Process every 5th frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                raise ValueError("Captured frame is None")
            if count % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_path = os.path.join(output_dir, f'frame_{count}.jpg')
                if not cv2.imwrite(frame_path, gray_frame):
                    raise IOError(f"Failed to write frame to {frame_path}")
            count += 1
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        raise
    finally:
        cap.release()
        print(f"Extracted {count} frames from the video.")

        if extract_nostrils:
            try:
                roi_dir = extract_nostrils_from_video(video_path, output_dir)
                subtracted_dir = subtract(video_path, roi_dir, 'nostrils')
                model_dir = 'best_nostrils_model.keras'
                result_nostrils = predict_frames_in_directory(subtracted_dir, model_dir)
                print(f"RESULT NOSTRILS: HORSE IS {result_nostrils}.")
                return result_nostrils
            except Exception as e:
                print(f"Error in nostrils processing for {video_path}: {str(e)}")
                raise
        else:
            try:
                subtracted_dir = subtract(video_path, output_dir, 'abdomen')
                model_dir = 'best_abdomen_model.keras'
                result_abdomen = predict_frames_in_directory(subtracted_dir, model_dir)
                print(f"RESULT ABDOMEN: HORSE IS {result_abdomen}.")
                return result_abdomen
            except Exception as e:
                print(f"Error in abdomen processing for {video_path}: {str(e)}")
                raise

def process_both(video_path_abdomen, video_path_nostrils):
    print(f"Processing videos {video_path_abdomen} and {video_path_nostrils}")
    video_name_abdomen = os.path.splitext(os.path.basename(video_path_abdomen))[0]
    video_name_nostrils = os.path.splitext(os.path.basename(video_path_nostrils))[0]
    output_dir_abdomen = os.path.join(FRAMES_DIR, video_name_abdomen, 'abdomen')
    output_dir_nostrils = os.path.join(FRAMES_DIR, video_name_abdomen, 'nostrils')

    if not os.path.exists(output_dir_abdomen):
        os.makedirs(output_dir_abdomen)

    if not os.path.exists(output_dir_nostrils):
        os.makedirs(output_dir_nostrils)

    # Extract frames for abdomen
    cap = cv2.VideoCapture(video_path_abdomen)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path_abdomen}")

    count = 0
    frame_skip = 5  # Process every 5th frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                raise ValueError("Captured frame is None")
            if count % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_path = os.path.join(output_dir_abdomen, f'frame_{count}.jpg')
                if not cv2.imwrite(frame_path, gray_frame):
                    raise IOError(f"Failed to write frame to {frame_path}")
            count += 1
    except Exception as e:
        print(f"Error processing video {video_path_abdomen}: {str(e)}")
        raise
    finally:
        cap.release()
        print(f"Extracted {count} frames from the video.")

    # Extract frames for nostrils
    cap = cv2.VideoCapture(video_path_nostrils)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path_nostrils}")

    count = 0
    frame_skip = 5  # Process every 5th frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                raise ValueError("Captured frame is None")
            if count % frame_skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_path = os.path.join(output_dir_nostrils, f'frame_{count}.jpg')
                if not cv2.imwrite(frame_path, gray_frame):
                    raise IOError(f"Failed to write frame to {frame_path}")
            count += 1
    except Exception as e:
        print(f"Error processing video {video_path_nostrils}: {str(e)}")
        raise
    finally:
        cap.release()
        print(f"Extracted {count} frames from the video.")

    try:
        roi_dir = extract_nostrils_from_video(video_path_nostrils, output_dir_nostrils)
        subtracted_dir_nostrils = subtract(video_path_nostrils, roi_dir, 'nostrils')
        subtracted_dir_abdomen = subtract(video_path_abdomen, output_dir_abdomen, 'abdomen')
        model_dir = 'best_model_both.keras'
        result_both = predict_from_two_directories(subtracted_dir_nostrils, subtracted_dir_abdomen, model_dir)
        print(f"RESULT BOTH: HORSE IS {result_both}.")
        return result_both
    except Exception as e:
        print(f"Error in processing for {video_path_abdomen} and {video_path_nostrils}: {str(e)}")
        raise

def extract_frame_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else None

def extract_nostrils_from_video(video_path, input_dir):
    model = YOLO("../../nostrils_detector.pt")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    roi_dir = os.path.join(ROI_DIR, video_name)

    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    with ThreadPoolExecutor() as executor:
        for frame_file in os.listdir(input_dir):
            frame_path = os.path.join(input_dir, frame_file)
            executor.submit(process_frame, model, frame_path, roi_dir)
                        
    return roi_dir

def process_frame(model, frame_path, roi_dir):
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Failed to load image: {frame_path}")
        return

    results = model(img)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_img = img[y1:y2, x1:x2]
        output = os.path.join(roi_dir, frame_path.split('/')[-1])
        cv2.imwrite(output, cropped_img)
        print(f"Cropped image saved at: {output}")

def subtract(video_path, input_dir, body_part):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(SUBTRACTED_DIR, video_name, body_part)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')], key=extract_frame_number)
    
    with ThreadPoolExecutor() as executor:
        for i in range(len(frame_files) - 1):
            frame1_path = os.path.join(input_dir, frame_files[i])
            frame2_path = os.path.join(input_dir, frame_files[i + 1])
            executor.submit(subtract_frames, frame1_path, frame2_path, output_dir, i)
    
    return output_dir

def subtract_frames(frame1_path, frame2_path, output_dir, idx):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print(f"Warning: Trouble opening files '{frame1_path}' or '{frame2_path}'. Skipping subtraction.")
        return

    if frame1.shape != frame2.shape:
        if frame1.size < frame2.size:
            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        else:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    subtracted_frame = blurred_frame_differencing(frame1, frame2)
    output_frame_path = os.path.join(output_dir, f'frame_{idx}.jpg')
    cv2.imwrite(output_frame_path, subtracted_frame)
    print(f"Subtracted frame saved at: {output_frame_path}")

if __name__ == '__main__':

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Process video files.')
    parser.add_argument('video_paths', type=str, nargs='+', help='Paths to the video files. Provide one or two paths.')
    parser.add_argument('--body_part', type=str, choices=['abdomen', 'nostrils'], help='Body part indication if only one video path is provided.')

    args = parser.parse_args()

    if len(args.video_paths) == 1:
        if not args.body_part:
            print("Error: --body_part must be specified when providing only one video path.")
        else:
            video_path = args.video_paths[0]
            if args.body_part == 'abdomen':
                print("Starting processing for abdomen video...")
                process_video(video_path, extract_nostrils=False)
            elif args.body_part == 'nostrils':
                print("Starting processing for nostrils video...")
                process_video(video_path, extract_nostrils=True)
    elif len(args.video_paths) == 2:
        video_path_abdomen = args.video_paths[0]
        video_path_nostrils = args.video_paths[1]
        print("Starting processing both videos...")
        process_both(video_path_abdomen, video_path_nostrils)
    else:
        print("Error: Provide one or two video paths.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total inference time: {elapsed_time:.2f} seconds")
