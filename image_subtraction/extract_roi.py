# This code is used to generate the cropped nostrils frames and save them into a directory.
import os
import cv2
from ultralytics import YOLO

# Load your model
model = YOLO('/cfs/home/u021554/tese/equine_asthma/nostrils_detector.pt')
#base_path = '/path/to/your/nostrils/frames'
base_path = '/cfs/home/u021554/tese/image_subtraction/final/extracted_frames'

def process_all_videos(base_path):
    """
    Process all videos in the base path, skipping certain directories and conditions.
    Extracts bounding boxes and saves cropped images.

    Args:
    - base_path (str): The base directory containing video frames.

    Returns:
    None
    """
    for body_part in os.listdir(base_path):
        body_part_path = os.path.join(base_path, body_part)
        if body_part == 'abdomen':
            continue

        if not os.path.isdir(body_part_path):
            continue
        
        for condition in os.listdir(body_part_path):
            condition_path = os.path.join(body_part_path, condition)
            if not os.path.isdir(condition_path):
                continue
            
            for horse in os.listdir(condition_path):

                horse_path = os.path.join(condition_path, horse)
                if not os.path.isdir(horse_path):
                    continue

                output_horse_path = os.path.join('roi', body_part, condition, horse)
                os.makedirs(output_horse_path, exist_ok=True)
                print('Output horse path:', output_horse_path)

                for frame_file in os.listdir(horse_path):
                    frame_path = os.path.join(horse_path, frame_file)
                    img = cv2.imread(frame_path)
                    if img is None:
                        print(f"Failed to load image: {frame_path}")
                        continue

                    results = model(img)
                    boxes = results[0].boxes.xyxy.tolist()
                    classes = results[0].boxes.cls.tolist()
                    confidences = results[0].boxes.conf.tolist()

                    for box in boxes:
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        cropped_img = img[y1:y2, x1:x2]
                        output = os.path.join(output_horse_path, frame_file)
                        cv2.imwrite(output, cropped_img)
                        print("Cropped image saved at:", output)

# Call the function with the base path
if __name__ == "__main__":
    process_all_videos(base_path)
