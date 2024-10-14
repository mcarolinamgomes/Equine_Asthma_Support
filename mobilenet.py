import os
os.environ['CUDA_HOME'] = '/cfs/home/u021554/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn' 
os.environ['LD_LIBRARY_PATH'] = '~/miniconda3/envs/SAM/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Only GPU 0 is visible to this program
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random
from tensorflow.keras.models import save_model

base_dir = '/cfs/home/u021554/tese/image_subtraction/final'
chosen_body_part = 'abdomen' # or 'nostrils'

# Constants
img_height, img_width = 180, 180
batch_size = 32
techniques = [
    'adaptive_thresholding',
    'basic_frame_differencing',
    'blurred_frame_differencing',
    'edge_based_differencing',
    'thresholding'
]
epochs = 10
k_folds = 5
class_labels = ['asthmatic', 'healthy']

# Helper function to extract the base video name
def base_video_name(video_name):
    # If video name contains '_', split and take the first part
    if '_' in video_name:
        return video_name.split('_')[0]
    return video_name

def debug_count_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

# Step 1: Verify Data Availability
def count_batches(dataset):
    num_batches = tf.data.experimental.cardinality(dataset)
    return num_batches

def get_video_frame_paths(technique_dir):
    video_frames = {}
    for body_part in os.listdir(technique_dir):
        if body_part != chosen_body_part:
            continue
        body_part_path = os.path.join(technique_dir, body_part)
        for condition in os.listdir(body_part_path):
            condition_path = os.path.join(body_part_path, condition)
            for horse in os.listdir(condition_path):
                horse_path = os.path.join(condition_path, horse)
                base_name = base_video_name(horse)  # Use the base name for grouping
                if base_name not in video_frames:
                    video_frames[base_name] = []
                video_frames[base_name].extend([
                    os.path.join(horse_path, frame) for frame in os.listdir(horse_path)
                ])
    return video_frames


def resize_and_pad(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]), Image.Resampling.LANCZOS)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill='black')

def process_and_load_images(frame_paths, img_height, img_width, class_labels):
    def generator():
        for frame_path in frame_paths:
            try:
                with Image.open(frame_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if img.size != (img_height, img_width):
                        img = resize_and_pad(img, (img_height, img_width))
                    img_array = np.array(img)
                    
                    parts = frame_path.split(os.sep)
                    class_label = parts[-3]
                    
                    if class_label in class_labels:
                        yield (img_array, class_labels.index(class_label))
                    else:
                        print(f"Unrecognized label {class_label} found in {frame_path}")
            except Exception as e:
                print(f"Error processing image {frame_path}: {e}")

    output_signature = (
        tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.prefetch(tf.data.AUTOTUNE)

def prepare_and_batch_dataset(frame_paths, img_height, img_width, class_labels, batch_size):
    dataset = process_and_load_images(frame_paths, img_height, img_width, class_labels)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def cross_validate(technique, base_dir, k_folds):
    technique_path = os.path.join(base_dir, technique)
    video_frame_paths = get_video_frame_paths(technique_path)
    videos = list(video_frame_paths.keys())
    print('videos',videos)
    
    # Assuming each video base name corresponds to a single label in the paths
    video_labels = [video_frame_paths[video][0].split(os.sep)[-3] for video in videos]
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    video_labels_numeric = [label_to_index[label] for label in video_labels]

    min_class_count = min(np.bincount(video_labels_numeric))
    k_folds = min(k_folds, min_class_count)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    with open(f"{base_dir}/image_classification/mobilenet/metrics/{technique}_{chosen_body_part}_metrics.txt", "w") as file:
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'confusion_matrix': [], 'test_videos': [] }

        for fold, (train_idx, test_idx) in enumerate(skf.split(videos, video_labels_numeric)):
            train_videos = [videos[idx] for idx in train_idx]
            test_videos = [videos[idx] for idx in test_idx]

            # Shuffle the video lists to ensure randomness
            random.shuffle(train_videos)
            random.shuffle(test_videos)

            train_frames = [frame for video in train_videos for frame in video_frame_paths[video]]
            test_frames = [frame for video in test_videos for frame in video_frame_paths[video]]

            total_samples1 = len(train_frames)  # Replace `frame_paths` with your actual data collection
            print(f"Total samples in dataset: {total_samples1}")
            total_samples2 = len(test_frames)  # Replace `frame_paths` with your actual data collection
            print(f"Total samples in dataset: {total_samples2}")

            img_height, img_width = 180, 180
            # When creating your training and validation datasets:
            train_ds = prepare_and_batch_dataset(train_frames, img_height, img_width, class_labels, batch_size)
            test_ds = prepare_and_batch_dataset(test_frames, img_height, img_width, class_labels, batch_size)

            # Now iterate over the dataset
            for images, labels in train_ds.take(1):  # Taking one batch to test
                print(f'Images batch size: {len(images)}')  # Should print: Images batch size: 32
                print(f'Labels batch size: {len(labels)}')  # Should print: Labels batch size: 32

            #train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            # To debug and ensure all batches are full
            num_batches = tf.data.experimental.cardinality(train_ds).numpy()
            print(f"Total full batches formed: {num_batches}")   

            #test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            # To debug and ensure all batches are full
            num_batches = tf.data.experimental.cardinality(test_ds).numpy()
            print(f"Total full batches formed: {num_batches}")        

            print("Number of batches in training dataset:", count_batches(train_ds))
            print("Number of batches in validation dataset:", count_batches(train_ds))

            # Debugging the dataset size
            print("Debug: Number of batches in training dataset:", debug_count_dataset(train_ds))
            print("Debug: Number of batches in validation dataset:", debug_count_dataset(test_ds))

            # Model setup
            base_model = MobileNet(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg', weights='imagenet')
            base_model.trainable = False
            model = Sequential([
                base_model,
                Dense(128, activation='relu'),
                Dense(len(class_labels), activation='softmax')
            ])
            # Learning rate schedule
            initial_learning_rate = 1e-5
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True)

            model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            #model.fit(train_ds, validation_data=test_ds, epochs=epochs)
            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(video_labels_numeric), y=video_labels_numeric)
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}


            # Use class weights in model training
            model.fit(train_ds, validation_data=test_ds, epochs=epochs, class_weight=class_weight_dict)

            # Evaluate the model on all test batches
            all_labels = []
            all_predictions = []
            for test_images, test_labels in test_ds:
                predictions = model.predict(test_images)
                predicted_classes = np.argmax(predictions, axis=1)
                all_labels.extend(test_labels.numpy())
                all_predictions.extend(predicted_classes)


            # Calculate metrics
            acc = accuracy_score(all_labels, all_predictions)
            prec = precision_score(all_labels, all_predictions, average='macro')
            rec = recall_score(all_labels, all_predictions, average='macro')
            f1 = f1_score(all_labels, all_predictions, average='macro')
            cm = confusion_matrix(all_labels, all_predictions)

            # Save metrics
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            fold_metrics['f1-score'].append(f1)
            fold_metrics['confusion_matrix'].append(cm)
            fold_metrics['test_videos'].append(test_videos)

            # Write to file
            file.write(f"Fold {fold+1}\n")
            file.write(f"Accuracy: {acc}\n")
            file.write(f"Precision: {prec}\n")
            file.write(f"Recall: {rec}\n")
            file.write(f"F1-Score: {f1}\n")
            file.write(f"Confusion Matrix:\n{cm}\n")
            file.write(f"Test videos: {test_videos}\n\n")


            model_save_path = f"{base_dir}/image_classification/mobilenet/models/{technique}_{chosen_body_part}_fold_{fold+1}_model_new.keras"
            model.save(model_save_path)

            # Write to file
            file.write(f"Model saved at: {model_save_path}\n")

    return fold_metrics


# Run cross-validation for each technique and body part
for technique in techniques:
    technique_path = os.path.join(base_dir,technique)
    cross_validate(technique, base_dir, k_folds)


