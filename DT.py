import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import json

def extract_keyword_from_path(data_path):
    keywords = ['nostrils', 'abdomen', 'both']
    return next((keyword for keyword in keywords if keyword in data_path), None)

def base_video_name(video_name):
    if '_' in video_name:
        return video_name.split('_')[0]
    return video_name

def custom_stratified_group_kfold(data, n_splits=5):
    video_labels = data.groupby('Base_Video')['Category'].first()
    fold_assignments = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_number, (train_idx, test_idx) in enumerate(skf.split(np.array(video_labels.index.tolist()), video_labels)):
        for idx in test_idx:
            fold_assignments[video_labels.index[idx]] = fold_number
    data['fold'] = data['Base_Video'].map(fold_assignments)
    return data

data_paths = ['../image_processing/extracted_features/abdomen_mask.csv',
              '../image_processing/extracted_features/nostrils_mask.csv',
              '../image_processing/extracted_features/both_mask.csv']

for data_path in data_paths:
    extracted_keyword = extract_keyword_from_path(data_path)
    os.makedirs(extracted_keyword, exist_ok=True)
    os.makedirs(f'{extracted_keyword}/DT', exist_ok=True)
    data = pd.read_csv(data_path)
    data['Category'] = data['Category'].map({'asthmatic': 1, 'healthy': 0})
    data['Base_Video'] = data['Video'].apply(base_video_name)
    data = custom_stratified_group_kfold(data)

    best_iteration = 0  # Initialize best_iteration before the loop
    best_model = None
    best_scaler = None
    best_accuracy = 0
    metrics_data = []
    wrongly_classified_videos = []
    i = 0
    # Initialize an empty dictionary to store majority predictions across iterations
    all_majority_predictions = {}
    video_predictions = {}
    results_per_iteration = []
    fold_video_info = []
    fold_info_filename = f'{extracted_keyword}/DT/fold_video_info.txt'
    majority_metrics_data = []

    
    for fold in range(5):
        train_data = data[data['fold'] != fold]
        test_data = data[data['fold'] == fold]
        feature_cols = [col for col in data.columns if col not in ['Category', 'Video', 'Base_Video', 'fold']]
        X_train = train_data[feature_cols].values
        y_train = train_data['Category'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['Category'].values

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Store predictions for majority voting
        for video, prediction in zip(test_data['Video'], y_pred):
            if video not in video_predictions:
                video_predictions[video] = []
            video_predictions[video].append(prediction)


        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
            best_iteration = fold + 1
            joblib.dump(clf, f'{extracted_keyword}/DT/best_dt_model_iteration_{fold+1}.joblib')

        metrics_data.append({
            'Iteration': fold + 1,
            'Accuracy': accuracy,
            'Precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
            'Recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall'],
            'F1-Score': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
        })

        # Processing train data
        train_counts = train_data['Video'].value_counts()
        for video, count in train_counts.items():
            video_class = 'asthmatic' if train_data[train_data['Video'] == video]['Category'].iloc[0] == 1 else 'healthy'
            fold_video_info.append(f"{fold + 1}\ttrain\t{video}\t{count}\t{video_class}\t{video_class}\n")
        
        # Map predictions to videos using DataFrame indices as keys
        video_to_pred = dict(zip(test_data.index, y_pred))
        for video, count in test_data['Video'].value_counts().items():
            actual_class = 'asthmatic' if test_data[test_data['Video'] == video]['Category'].iloc[0] == 1 else 'healthy'
            predicted_class = 'asthmatic' if video_to_pred[test_data[test_data['Video'] == video].index[0]] == 1 else 'healthy'
            fold_video_info.append(f"{fold + 1}\ttest\t{video}\t{count}\t{actual_class}\t{predicted_class}\n")


        #file.write(f'Iteration {fold + 1}:\n')
        #file.write('Test Videos:\n')
        #file.write('\n'.join(test_data['Video'].tolist()) + test_data['Category'].tolist() + '\n')
        #file.write('Train Videos:\n')
        #file.write('\n'.join(train_data['Video'].tolist()) + test_data['Category'].tolist() + '\n\n')

        for video, true, pred in zip(test_data['Video'], y_test, y_pred):
            if true != pred:
                wrongly_classified_videos.append({
                    'Iteration': fold + 1,
                    'Video': video,
                    'True_Label': 'asthmatic' if true == 1 else 'healthy',
                    'Predicted_Label': 'asthmatic' if pred == 1 else 'healthy'
                })

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Iteration {fold+1}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{extracted_keyword}/DT/confusion_matrix_iteration_{fold+1}.png')
        plt.close()

        i += 1

         # Calculate majority votes
        majority_predictions = {}
        actual_labels = data[['Video', 'Category']].drop_duplicates().set_index('Video')['Category']
        video_majority_predictions = {}  # Stores final predictions for each video
        for video, predictions in video_predictions.items():
            majority_vote = max(set(predictions), key=predictions.count)
            majority_predictions[video] = majority_vote
            video_majority_predictions[video] = {'predicted': majority_vote, 'actual': actual_labels[video]}

        # Convert majority predictions to format suitable for classification report
        y_true_majority = [info['actual'] for info in video_majority_predictions.values()]
        y_pred_majority = [info['predicted'] for info in video_majority_predictions.values()]

        # Calculate metrics for majority voting
        accuracy_majority = accuracy_score(y_true_majority, y_pred_majority)
        report_majority = classification_report(y_true_majority, y_pred_majority, output_dict=True)

        majority_metrics_data.append({
            'Iteration': fold + 1,
            'Accuracy': accuracy_majority,
            'Precision': report_majority['weighted avg']['precision'],
            'Recall': report_majority['weighted avg']['recall'],
            'F1-Score': report_majority['weighted avg']['f1-score']
        })

    metrics_df = pd.DataFrame(metrics_data)
    std_deviation = metrics_df.std()  # Calculating standard deviation

        # Convert standard deviation Series to DataFrame and adjust to match the structure of metrics_df
    std_deviation_df = pd.DataFrame([std_deviation], index=['Std Deviation']).reset_index(drop=True)
    std_deviation_df['Iteration'] = 'Standard Deviation'

    # Concatenate the standard deviation row to the original metrics DataFrame
    complete_metrics_df = pd.concat([metrics_df, std_deviation_df], ignore_index=True)

    complete_metrics_df.to_csv(f'{extracted_keyword}/DT/metrics_summary.csv', index=False)

    best_model_details = {
        'Best_Iteration': best_iteration,
        'Best_Accuracy': best_accuracy
    }
    with open(f'{extracted_keyword}/DT/best_model_details.txt', 'w') as f:
        f.write(str(best_model_details))

    wrongly_classified_df = pd.DataFrame(wrongly_classified_videos)
    wrongly_classified_df.to_csv(f'{extracted_keyword}/DT/wrongly_classified_videos.csv', index=False)

    # Save the majority vote metrics
    majority_metrics_df = pd.DataFrame(majority_metrics_data)
    std_deviation_majority = majority_metrics_df.std()

    std_deviation_majority_df = pd.DataFrame([std_deviation_majority], index=['Std Deviation']).reset_index(drop=True)
    std_deviation_majority_df['Iteration'] = 'Standard Deviation'

    # Concatenate the standard deviation row to the original metrics DataFrame
    majority_complete_metrics_df = pd.concat([majority_metrics_df, std_deviation_majority_df], ignore_index=True)

    majority_complete_metrics_df.to_csv(f'{extracted_keyword}/DT/majority_metrics_summary.csv', index=False)


    # Writing detailed fold information to a text file
    with open(fold_info_filename, 'w') as file:
        file.write("Fold\tVideo_Split\tVideo\tOccurrences\tActual Class\tPredicted Class\n")  # Writing header
        for info in fold_video_info:
            file.write(info)  # Write each line of information to the file

    print("All data has been saved to files.")
