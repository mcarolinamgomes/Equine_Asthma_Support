import cv2
import os
import numpy as np
import re

base_path = 'path/to/the/abdomen_frames'  # or 'path/to/the/roi_nostrils_frames'
body_part = 'abdomen' # or 'nostrils'
subtraction_methods = [basic_frame_differencing, thresholding, blurred_frame_differencing, adaptive_thresholding, edge_based_differencing ]  # Add all your methods

# SUBTRACTION METHODS 

def basic_frame_differencing(frame1, frame2):
    return cv2.absdiff(frame1, frame2)

def thresholding(frame1, frame2):
    # Compute the absolute difference between the frames
    difference = cv2.absdiff(frame1, frame2)
    # Apply thresholding to the difference
    _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    return thresholded

def blurred_frame_differencing(frame1, frame2):
    blurred_frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    blurred_frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    return cv2.absdiff(blurred_frame1, blurred_frame2)

def adaptive_thresholding(frame1, frame2):
    # Compute the absolute difference between the frames
    difference = cv2.absdiff(frame1, frame2)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresholded = cv2.adaptiveThreshold(gray_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return adaptive_thresholded

def edge_based_differencing(frame1,frame2):
    edge_frame1 = cv2.Canny(frame1, 100, 200)
    edge_frame2 = cv2.Canny(frame2, 100, 200)
    difference = cv2.absdiff(edge_frame1, edge_frame2)
    return difference

def extract_frame_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else None

def process_all_videos(base_path, subtraction_methods):
    for subtraction_method in subtraction_methods:
        for condition in os.listdir(base_path):
            condition_path = os.path.join(base_path, condition)
            if not os.path.isdir(condition_path):
                continue
            for horse in os.listdir(condition_path):
                print(horse)
                horse_path = os.path.join(condition_path, horse)
                # Now, use this function to sort the frame files
                if not os.path.isdir(horse_path):
                    continue
                output_horse_path = os.path.join(subtraction_method.__name__, body_part, condition, horse)
                os.makedirs(output_horse_path, exist_ok=True)
                frame_files = sorted([f for f in os.listdir(horse_path) if f.endswith('.jpg')], key=extract_frame_number)
                for i in range(len(frame_files) - 1):
                    frame1_path = os.path.join(horse_path, frame_files[i])
                    frame2_path = os.path.join(horse_path, frame_files[i + 1])
                    frame1 = cv2.imread(frame1_path)
                    frame2 = cv2.imread(frame2_path)

                    # Check if the images were loaded successfully
                    if frame1 is None or frame2 is None:
                        print(f"Warning: Trouble opening files '{frame1_path}' or '{frame2_path}'. Skipping subtraction.")
                        continue

                    if frame1.shape != frame2.shape:
                        # Determine which frame is smaller
                        if frame1.size < frame2.size:
                            # Resize frame1 to match frame2's dimensions
                            frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
                        else:
                            # Resize frame2 to match frame1's dimensions
                            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                    
                    subtracted_frame = subtraction_method(frame1, frame2)
                    output_frame_path = os.path.join(output_horse_path, f'frame_{i}.jpg')
                    print('output_frame_path',output_frame_path)
                    cv2.imwrite(output_frame_path, subtracted_frame)

if __name__ == "__main__":
    process_all_videos(base_path,subtraction_methods)

    
    
