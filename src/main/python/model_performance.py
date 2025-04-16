import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from llama_cpp import Llama
import time
import random
import json

# Path to the model
MODEL_PATH = r"C:\Users\user\Documents\EEG_Project\model\unsloth.Q4_K_M1.gguf"
# Path to the dataset
DATASET_PATH = r"C:\Users\user\Documents\EEG_Project\src\EEG_dataset.json"

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    print(f"Loading dataset from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)  # Load the JSON data
        
        print(f"Dataset loaded. {len(data)} entries found.")
        
        eeg_data = []
        labels = []
        
        for entry in data:
            eeg_data.append(entry['input'])
            labels.append(entry['output'])
        
        eeg_data = np.array(eeg_data)
        labels = np.array(labels)
        
        label_mapping = {0: "Class 0", 1: "Class 1"}
        
        print(f"\nLabel distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for encoded, count in zip(unique, counts):
            print(f"{encoded} ({label_mapping[encoded]}): {count} samples")
        
        return eeg_data, labels, label_mapping
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def preprocess_eeg_data(eeg_data):
    print("\nPreprocessing EEG data...")
    
    if np.isnan(eeg_data).any():
        print("Filling missing values with column means...")
        col_means = np.nanmean(eeg_data, axis=0)
        inds = np.where(np.isnan(eeg_data))
        eeg_data[inds] = np.take(col_means, inds[1])
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(eeg_data)
    print("Data preprocessing complete.")
    
    return normalized_data

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print("\nLoading model...")
    
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=512,  # Keep context size small for CPU performance
            n_batch=128, # Smaller batch size for faster CPU inference
            verbose=False  # Reduce logging verbosity
        )
        
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def classify_eeg(model, eeg_features, label_mapping, class_weights=None):
    max_features = min(50, len(eeg_features))
    
    eeg_data_str = " ".join([f"{val:.4f}" for val in eeg_features[:max_features]])
    
    class0 = label_mapping.get(0, "Class 0")
    class1 = label_mapping.get(1, "Class 1")
    
    prompt = f"""
### EEG Classification Task
Analyze this EEG data to determine whether it represents {class0} or {class1}.

### Important Instructions:
- Consider both classes equally likely
- Be objective in your analysis
- Provide ONLY a number: 0 or 1
- Any ambiguity should be resolved by careful feature analysis

### Classes:
- 0: {class0}
- 1: {class1}

### EEG Data Features:
{eeg_data_str}

### Classification (ONLY respond with 0 or 1):
"""
    
    try:
        temperature = 0.1
        if class_weights is not None:
            temp_multiplier = max(class_weights.values()) / min(class_weights.values())
            temperature = min(0.3, 0.1 * temp_multiplier)
        
        output = model(
            prompt,
            max_tokens=3,  # Only need a single digit
            temperature=temperature,
            stop=[".", "\n", " "],
            echo=False
        )
        
        response_text = output["choices"][0]["text"].strip()
        
        for i in range(len(response_text)):
            if response_text[i:i+1] in ["0", "1"]:
                predicted_class = int(response_text[i:i+1])
                return predicted_class
        
        print(f"Warning: Could not extract a valid class from response: '{response_text}'")
        if class_weights:
            options = [0, 1]
            probabilities = [class_weights[0], class_weights[1]]
            total = sum(probabilities)
            probabilities = [p/total for p in probabilities]
            return np.random.choice(options, p=probabilities)
        else:
            return random.randint(0, 1)
    
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return random.randint(0, 1)

def evaluate_with_cv(model, eeg_data, labels, label_mapping, n_folds=5):
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_confusion_matrices = []
    
    all_y_true = []
    all_y_pred = []
    
    class_predictions = {0: 0, 1: 0}
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(eeg_data, labels)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        
        X_test = eeg_data[test_idx]
        y_test = labels[test_idx]
        
        class_weights = None
        if fold_idx > 0 and (class_predictions[0] + class_predictions[1]) > 0:
            # If there's a bias toward one class, adjust weights
            total = class_predictions[0] + class_predictions[1]
            if class_predictions[0] > class_predictions[1]:
                # Model predicts class 0 more often, boost class 1
                ratio = class_predictions[0] / class_predictions[1] if class_predictions[1] > 0 else 2.0
                class_weights = {0: 1.0, 1: min(3.0, ratio)}
            else:
                # Model predicts class 1 more often, boost class 0
                ratio = class_predictions[1] / class_predictions[0] if class_predictions[0] > 0 else 2.0
                class_weights = {0: min(3.0, ratio), 1: 1.0}
            
            print(f"Applying class weights to counteract bias: {class_weights}")
        
        fold_predictions = []
        
        batch_size = 10
        num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size > 0 else 0)
        
        start_time = time.time()
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(X_test))
            
            if batch_idx % 2 == 0:
                print(f"  Processing batch {batch_idx + 1}/{num_batches} (samples {batch_start}-{batch_end-1})")
            
            for i in range(batch_start, batch_end):
                prediction = classify_eeg(model, X_test[i], label_mapping, class_weights)
                fold_predictions.append(prediction)
                
                if prediction in class_predictions:
                    class_predictions[prediction] += 1
        
        elapsed_time = time.time() - start_time
        print(f"  Processed {len(X_test)} samples in {elapsed_time:.2f} seconds")
        print(f"  Class distribution in predictions: Class 0: {fold_predictions.count(0)}, Class 1: {fold_predictions.count(1)}")
        
        accuracy = accuracy_score(y_test, fold_predictions)
        precision = precision_score(y_test, fold_predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, fold_predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, fold_predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, fold_predictions)
        
        print(f"  Fold {fold_idx + 1} Metrics:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_confusion_matrices.append(cm)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(fold_predictions)
        
        plt.figure(figsize=(8, 6))
        display_labels = [f"{i}: {label_mapping[i]}" for i in sorted(label_mapping.keys())]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_fold_{fold_idx + 1}.png")
        plt.show()
    
    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    
    print("\nCross-Validation Results:")
    print(f"  Average Accuracy: {avg_accuracy:.4f} (±{np.std(all_accuracies):.4f})")
    print(f"  Average Precision: {avg_precision:.4f} (±{np.std(all_precisions):.4f})")
    print(f"  Average Recall: {avg_recall:.4f} (±{np.std(all_recalls):.4f})")
    print(f"  Average F1-Score: {avg_f1:.4f} (±{np.std(all_f1s):.4f})")
    
    overall_cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 8))
    display_labels = [f"{i}: {label_mapping[i]}" for i in sorted(label_mapping.keys())]
    disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm, display_labels=display_labels)
    disp.plot(cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.tight_layout()
    plt.savefig("overall_confusion_matrix.png")
    plt.show()
    
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    
    class_metrics = {}
    for cls in np.unique(all_y_true):
        cls_indices = np.array(all_y_true) == cls
        if np.sum(cls_indices) > 0:
            cls_accuracy = accuracy_score(
                np.array(all_y_true)[cls_indices], 
                np.array(all_y_pred)[cls_indices]
            )
            class_metrics[f"Class {cls} ({label_mapping[cls]})"] = cls_accuracy
    
    metrics = {
        "Overall Accuracy": overall_accuracy,
        "Overall Precision": overall_precision,
        "Overall Recall": overall_recall,
        "Overall F1-Score": overall_f1,
        "Cross-Val Accuracy": avg_accuracy,
        "Cross-Val Precision": avg_precision,
        "Cross-Val Recall": avg_recall,
        "Cross-Val F1-Score": avg_f1,
    }
    
    metrics.update(class_metrics)
    
    return metrics, all_y_true, all_y_pred

def plot_metrics(metrics):
    main_metrics = {k: v for k, v in metrics.items() if "Overall" in k or "Cross-Val" in k}
    class_metrics = {k: v for k, v in metrics.items() if "Class" in k}
    
    plt.figure(figsize=(12, 8))
    metric_names = list(main_metrics.keys())
    metric_values = list(main_metrics.values())
    
    plt.barh(metric_names, metric_values, color='skyblue')
    plt.xlabel("Value")
    plt.ylabel("Metric")
    plt.title("Overall Performance Metrics")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(metric_values):
        plt.text(max(v + 0.01, 0.01), i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig("overall_metrics.png")
    plt.show()
    
    if class_metrics:
        plt.figure(figsize=(12, 6))
        class_names = list(class_metrics.keys())
        class_values = list(class_metrics.values())
        
        colors = ['lightgreen' if 'Class 0' in name else 'salmon' for name in class_names]
        plt.barh(class_names, class_values, color=colors)
        plt.xlabel("Accuracy")
        plt.ylabel("Class")
        plt.title("Per-Class Accuracy")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(class_values):
            plt.text(max(v + 0.01, 0.01), i, f"{v:.4f}", va='center')
        
        plt.tight_layout()
        plt.savefig("class_metrics.png")
        plt.show()

def main():
    try:
        eeg_data, labels, label_mapping = load_dataset(DATASET_PATH)
        
        preprocessed_data = preprocess_eeg_data(eeg_data)
        
        model = load_model(MODEL_PATH)
        
        metrics, y_true, y_pred = evaluate_with_cv(
            model, preprocessed_data, labels, label_mapping, n_folds=3
        )
        
        plot_metrics(metrics)
        
        print("\nClassification completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
