import os
import cv2
import numpy as np
import csv
from scipy.stats import iqr
import pandas as pd

def extract_number(filename):
    """Extract number from the filename assuming format 'prefix_number_suffix.ext'."""
    base = os.path.basename(filename)
    try:
        num_part = base.split('_')[1]
        return int(num_part)
    except (IndexError, ValueError):
        return 0

def calculate_area(image_path):
    """Calculate the area of the white region in a binary mask."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return 0

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image from: {image_path}")
        return 0

    return np.sum(image > 0)

def rms_change(areas):
    """Calculate Root Mean Square of the changes between consecutive areas."""
    changes = np.diff(areas)
    squared_changes = np.square(changes)
    return np.sqrt(np.mean(squared_changes))

def process_videos(base_dir, csv_file_path):
    """Process videos to extract and save area metrics to a CSV file."""
    # Clear the CSV file if it already exists
    open(csv_file_path, 'w').close()
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Video', 'Mean Relative Area', 'Area StdDev', 'Coefficient of Variation', 'Range', 'Interquartile Range', 'Variance', 'RMS Change'])
        for category in os.listdir(base_dir):
            print('Processing category:', category)
            category_path = os.path.join(base_dir, category)
            if os.path.isdir(category_path):
                for video in os.listdir(category_path):
                    print('Processing video:', video)
                    video_path = os.path.join(category_path, video)
                    if os.path.isdir(video_path):
                        mask_files = [f for f in os.listdir(video_path)]
                        mask_files.sort(key=extract_number)
                        for i in range(0, len(mask_files), 180):
                            batch_files = mask_files[i:i+180]
                            areas = [calculate_area(os.path.join(video_path, f)) for f in batch_files]
                            if areas:
                                baseline_area = areas[0]
                                relative_areas = [area / baseline_area for area in areas if baseline_area > 0]
                                if not relative_areas:
                                    continue
                                mean_relative_area = np.mean(relative_areas)
                                std_dev = np.std(relative_areas)
                                cv = std_dev / mean_relative_area if mean_relative_area else 0
                                data_range = np.ptp(relative_areas)
                                iqr_value = iqr(relative_areas)
                                variance = np.var(relative_areas)
                                rms_change_value = rms_change(relative_areas)
                                writer.writerow([category, video, mean_relative_area, std_dev, cv, data_range, iqr_value, variance, rms_change_value])

    # Remove rows with NaN values
    df = pd.read_csv(csv_file_path)
    df.dropna(inplace=True)
    df.to_csv(csv_file_path, index=False)

    print(f'Data saved to {csv_file_path}')

def merge_data(csv_file_path1, csv_file_path2, output_csv_file_path):
    """Merge two CSV files and save the result."""
    data1 = pd.read_csv(csv_file_path1)
    data2 = pd.read_csv(csv_file_path2)

    merged_data = pd.merge(data1, data2, on=['Category', 'Video'], how='left', suffixes=('', '_from_data2'))
    merged_data = merged_data.dropna()

    columns_to_add = ['Mean Relative Area_from_data2', 'Area StdDev_from_data2', 
                      'Coefficient of Variation_from_data2', 'Range_from_data2', 
                      'Interquartile Range_from_data2', 'Variance_from_data2', 
                      'RMS Change_from_data2']
    final_columns = list(data1.columns) + columns_to_add
    final_data = merged_data[final_columns]

    final_data.to_csv(output_csv_file_path, index=False)
    print(f'Merged data saved to {output_csv_file_path}')

if __name__ == "__main__":
    base_dir_nostrils = '/cfs/home/u021554/tese/equine_asthma/feature_extraction_method/training/image_processing/saved_masks/nostrils_masks'
    csv_file_path_nostrils = '/cfs/home/u021554/tese/equine_asthma/feature_extraction_method/training/image_processing/extracted_features/nostrils_mask.csv'
    process_videos(base_dir_nostrils, csv_file_path_nostrils)

    base_dir_abdomen = '/cfs/home/u021554/tese/equine_asthma/feature_extraction_method/training/image_processing/saved_masks/abdomen_masks'
    csv_file_path_abdomen = '/cfs/home/u021554/tese/equine_asthma/feature_extraction_method/training/image_processing/extracted_features/abdomen_mask.csv'
    process_videos(base_dir_abdomen, csv_file_path_abdomen)

    output_csv_file_path = '/cfs/home/u021554/tese/equine_asthma/feature_extraction_method/training/image_processing/extracted_features/both_mask.csv'
    merge_data(csv_file_path_nostrils, csv_file_path_abdomen, output_csv_file_path)
