import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def extract_keyword_from_path(data_path):
    """Extract keyword (nostrils, abdomen, both) from the given data path."""
    keywords = ['nostrils', 'abdomen', 'both']
    return next((keyword for keyword in keywords if keyword in data_path), None)

def base_video_name(video_name):
    """Extract base video name from the given video name."""
    return video_name.split('_')[0] if '_' in video_name else video_name

def custom_stratified_group_kfold(data, n_splits=5):
    """Perform custom stratified group K-Fold split."""
    video_labels = data.groupby('Base_Video')['Category'].first()
    fold_assignments = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_number, (train_idx, test_idx) in enumerate(skf.split(np.array(video_labels.index.tolist()), video_labels)):
        for idx in test_idx:
            fold_assignments[video_labels.index[idx]] = fold_number
    data['fold'] = data['Base_Video'].map(fold_assignments)
    return data

def train_and_predict(train_data, test_data, feature_cols):
    """Train SVM model and predict using the test data."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_cols].values)
    X_test = scaler.transform(test_data[feature_cols].values)

    clf = SVC(kernel="linear", class_weight='balanced')
    clf.fit(X_train, train_data['Category'].values)
    y_pred = clf.predict(X_test)
    return y_pred, clf, scaler

def main():
    data_paths = ['../image_processing/extracted_features/abdomen_mask.csv',
              '../image_processing/extracted_features/nostrils_mask.csv',
              '../image_processing/extracted_features/both_mask.csv']

    for data_path in data_paths:
        extracted_keyword = extract_keyword_from_path(data_path)
        os.makedirs(extracted_keyword, exist_ok=True)
        os.makedirs(f'{extracted_keyword}/SVM', exist_ok=True)

        data = pd.read_csv(data_path)
        data['Category'] = data['Category'].map({'asthmatic': 1, 'healthy': 0})
        data['Base_Video'] = data['Video'].apply(base_video_name)
        data = custom_stratified_group_kfold(data)

        majority_metrics_data = []
        feature_cols = [col for col in data.columns if col not in ['Category', 'Video', 'Base_Video', 'fold']]
        
        for fold in range(5):
            train_data = data[data['fold'] != fold]
            test_data = data[data['fold'] == fold]
            y_pred, clf, scaler = train_and_predict(train_data, test_data, feature_cols)
            metrics = calculate_metrics(test_data['Category'].values, y_pred)
            majority_metrics_data.append(metrics)
            save_fold_results(fold, extracted_keyword, test_data, y_pred, clf, scaler)

        save_summary(extracted_keyword, majority_metrics_data)

def calculate_metrics(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': classification_report(y_true, y_pred, output_dict=True, zero_division=0)['weighted avg']['precision'],
        'Recall': classification_report(y_true, y_pred, output_dict=True, zero_division=0)['weighted avg']['recall'],
        'F1-Score': classification_report(y_true, y_pred, output_dict=True, zero_division=0)['weighted avg']['f1-score']
    }

def save_fold_results(fold, keyword, test_data, predictions, clf, scaler):
    """Save confusion matrix plot, model, and scaler for the current fold."""
    filename = f'{keyword}/SVM/confusion_matrix_iteration_{fold+1}.png'
    cm = confusion_matrix(test_data['Category'].values, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Iteration {fold+1}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

    joblib.dump(clf, f'{keyword}/SVM/best_svc_model_iteration_{fold+1}.joblib')
    joblib.dump(scaler, f'{keyword}/SVM/best_scaler_iteration_{fold+1}.joblib')

def save_summary(keyword, data):
    """Save summary of majority metrics to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(f'{keyword}/SVM/majority_metrics_summary.csv', index=False)

if __name__ == '__main__':
    main()
