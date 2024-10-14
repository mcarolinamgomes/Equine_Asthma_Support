import os
import re
import statistics

techniques = [
    'adaptive_thresholding',
    'basic_frame_differencing',
    'blurred_frame_differencing',
    'edge_based_differencing',
    'thresholding'
]

def extract_metrics_from_file(filename):
    # Read the content of the file
    with open(filename, 'r') as file:
        content = file.read()
    
    # Define regex patterns to find metrics
    accuracy_pattern = r"Accuracy: ([0-9.]+)"
    precision_pattern = r"Precision: ([0-9.]+)"
    recall_pattern = r"Recall: ([0-9.]+)"
    f1_score_pattern = r"F1-Score: ([0-9.]+)"
    
    # Find all matches in the file content
    accuracies = list(map(float, re.findall(accuracy_pattern, content)))
    precisions = list(map(float, re.findall(precision_pattern, content)))
    recalls = list(map(float, re.findall(recall_pattern, content)))
    f1_scores = list(map(float, re.findall(f1_score_pattern, content)))
    
    # Compute averages and standard deviations
    metrics = {
        'Average Accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
        'Std Dev Accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
        'Average Precision': sum(precisions) / len(precisions) if precisions else 0,
        'Std Dev Precision': statistics.stdev(precisions) if len(precisions) > 1 else 0,
        'Average Recall': sum(recalls) / len(recalls) if recalls else 0,
        'Std Dev Recall': statistics.stdev(recalls) if len(recalls) > 1 else 0,
        'Average F1-Score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'Std Dev F1-Score': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0
    }
    return metrics

# Path directories with model names
metrics_dirs = [
    '/cfs/home/u021554/tese/equine_asthma/image_subtraction_method/training/classification/mobilenet/metrics',
    '/cfs/home/u021554/tese/equine_asthma/image_subtraction_method/training/classification/resnet/metrics',
    '/cfs/home/u021554/tese/equine_asthma/image_subtraction_method/training/classification/vgg16/metrics',
    '/cfs/home/u021554/tese/equine_asthma/image_subtraction_method/training/classification/efficientnet/metrics'
]

# Output file path
output_file_path = 'average_metrics_summary.txt'

# Initialize dictionary for all metrics organized by body part, model, and subtraction techniques
all_metrics = {'abdomen': {}, 'both': {}, 'nostrils': {}}
for body_part in all_metrics.keys():
    for metrics_dir in metrics_dirs:
        model_name = metrics_dir.split('/')[-2]
        all_metrics[body_part][model_name] = []

# Process each directory
for metrics_dir in metrics_dirs:
    model_name = metrics_dir.split('/')[-2]  # Get the model name from directory path
    for metrics_file in os.listdir(metrics_dir):
        if metrics_file.endswith(".txt") and any(tech in metrics_file for tech in techniques):
            metric_file_path = os.path.join(metrics_dir, metrics_file)
            body_part = 'abdomen' if 'abdomen_metrics' in metrics_file else 'nostrils' if 'nostrils_metrics' in metrics_file else 'both'
            metrics = extract_metrics_from_file(metric_file_path)
            all_metrics[body_part][model_name].append((metrics_file, metrics))

# Write to a single output file
with open(output_file_path, 'w') as output_file:
    for body_part, models in all_metrics.items():
        output_file.write(f"\n=== {body_part.upper()} ===\n")
        for model, files_metrics in models.items():
            output_file.write(f"\nModel: {model.upper()}\n")
            for file_name, metrics in files_metrics:
                output_file.write(f"Subtraction Technique: {file_name}\n")
                for metric, value in metrics.items():
                    output_file.write(f"{metric}: {value:.4f}\n")
                output_file.write("\n")
