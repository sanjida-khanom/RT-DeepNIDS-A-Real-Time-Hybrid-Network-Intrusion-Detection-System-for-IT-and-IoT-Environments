import matplotlib
matplotlib.use('Agg')  # MUST BE FIRST. Completely disables GUI popups.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import shap
import lime
import lime.lime_tabular

from preprocess import load_and_preprocess
from models import build_random_forest, build_decision_tree, build_xgboost, build_cnn_gru, build_cnn_transformer

# --------- CONFIGURATION ---------
BASE_PATH = r"D:\Capstone project\Cross Validation pipeline\plots"
os.makedirs(BASE_PATH, exist_ok=True)

PATH_2017 = r"D:\Capstone project\CIC-IDS-2017\MachineLearningCVE"
PATH_2018 = r"D:\Capstone project\CIC-IDS-2018\CIC-IDS-2018"
PATH_TON = r"D:\Capstone project\TON-IOT-V3\NF-ToN-IoT-v3.csv"

# --------- VISUALIZATION HELPERS ---------
def save_plot(title):
    file_path = os.path.join(BASE_PATH, f"{title}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.clf()
    plt.close('all')

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {title}\n(Tested on Unseen ToN-IoT)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(f"CM_{title}")

def plot_combined_roc(roc_data):
    plt.figure(figsize=(10, 8))
    for name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], lw=2, label=f"{name} (AUC = {data['auc']:.3f})")
    plt.plot([1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc="lower right")
    save_plot("Combined_ROC_Curve")

def plot_model_comparison(results):
    models = [res['Model'] for res in results]
    accuracies = [res['Accuracy'] * 100 for res in results]
    aucs = [res['AUC'] for res in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='skyblue')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance Comparison (Accuracy vs AUC)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, aucs, width, label='AUC Score', color='lightgreen')
    ax2.set_ylabel('AUC Score')

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    save_plot("Model_Comparison_Bar_Chart")

def plot_individual_metrics(results):
    models = [res['Model'] for res in results]
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=[res['Accuracy'] * 100 for res in results], hue=models, palette='viridis', legend=False)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    save_plot("Accuracy_Comparison")

    # 2. AUC Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=[res['AUC'] for res in results], hue=models, palette='magma', legend=False)
    plt.title('AUC Comparison')
    plt.ylabel('AUC Score')
    save_plot("AUC_Comparison")

    # 3. Training Time Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=[res['TrainTime(s)'] for res in results], hue=models, palette='crest', legend=False)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (Seconds)')
    save_plot("Training_Time")

    # 4. Testing Time Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=[res['TestTime(s)'] for res in results], hue=models, palette='flare', legend=False)
    plt.title('Testing Time Comparison')
    plt.ylabel('Time (Seconds)')
    save_plot("Testing_Time")

# --------- MAIN PIPELINE ---------
def main():
    results = []
    roc_data = {}
    global trained_xgb_model
    trained_xgb_model = None

    # Load data for Transfer Learning
    X_train_source, y_train_source, X_ton_train, X_ton_test, y_ton_train, y_ton_test, feature_names = load_and_preprocess(PATH_2017, PATH_2018, PATH_TON)

    # Base Training Validation Split
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_source, y_train_source, test_size=0.15, random_state=42, stratify=y_train_source)

    n_features = X_train_source.shape[1]
    timesteps = 1
    
    # Strictly convert to Numpy Arrays to fix "unknown rank" errors in Deep Learning models
    X_train_seq = np.array(X_train_dl).reshape((-1, timesteps, n_features)).astype('float32')
    X_val_seq = np.array(X_val_dl).reshape((-1, timesteps, n_features)).astype('float32')
    X_ton_train_seq = np.array(X_ton_train).reshape((-1, timesteps, n_features)).astype('float32')
    X_ton_test_seq = np.array(X_ton_test).reshape((-1, timesteps, n_features)).astype('float32')

    y_train_dl = np.array(y_train_dl).astype('float32')
    y_val_dl = np.array(y_val_dl).astype('float32')
    y_ton_train = np.array(y_ton_train).astype('float32')
    y_ton_test = np.array(y_ton_test).astype('float32')

    # For ML Models (Combined Training for Domain Adaptation)
    X_combined_ml = np.vstack((X_train_source, X_ton_train))
    y_combined_ml = np.concatenate((y_train_source, y_ton_train))

    weights = compute_class_weight('balanced', classes=np.unique(y_combined_ml), y=y_combined_ml)
    class_weights = dict(enumerate(weights))

    def evaluate_model(model, name, is_dl=False):
        global trained_xgb_model
        print(f"\n>> Training {name}...")
        start_train = time.time()
        
        if is_dl:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            print(f"   -> Phase 1: Base Training on Source Domain (CIC-IDS)")
            model.fit(X_train_seq, y_train_dl, validation_data=(X_val_seq, y_val_dl), epochs=15, batch_size=256, callbacks=[early_stop], verbose=1)
            
            print(f"   -> Phase 2: Fine-Tuning on Target Domain (ToN-IoT)")
            model.optimizer.learning_rate.assign(0.0001)
            model.fit(X_ton_train_seq, y_ton_train, epochs=5, batch_size=128, class_weight=class_weights, verbose=1)
        else:
            print(f"   -> Training on Combined Source + Target Train Data")
            model.fit(X_combined_ml, y_combined_ml)
            if name == 'XGB': trained_xgb_model = model
            
        train_time = time.time() - start_train
        
        print(f">> Final Testing {name} on Unseen NF-ToN-IoT-v3...")
        start_test = time.time()
        
        if is_dl:
            y_prob = model.predict(X_ton_test_seq).ravel()
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_ton_test)
            y_prob = model.predict_proba(X_ton_test)[:, 1]
            
        test_time = time.time() - start_test
        
        acc = accuracy_score(y_ton_test, y_pred)
        auc = roc_auc_score(y_ton_test, y_prob)
        fpr, tpr, _ = roc_curve(y_ton_test, y_prob)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        
        report = classification_report(y_ton_test, y_pred, zero_division=0)
        with open(os.path.join(BASE_PATH, f"Classification_Report_{name}.txt"), "w") as f:
            f.write(f"Classification Report for {name}\n")
            f.write("="*50 + "\n")
            f.write(report)
            
        plot_confusion(y_ton_test, y_pred, name)
        print(f"[{name}] Completed. Final Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
        return {'Model': name, 'Accuracy': acc, 'AUC': auc, 'TrainTime(s)': train_time, 'TestTime(s)': test_time}

    for m_func, m_name in zip([build_random_forest, build_decision_tree, build_xgboost], ['RF', 'DT', 'XGB']):
        results.append(evaluate_model(m_func(), m_name))

    results.append(evaluate_model(build_cnn_gru(timesteps, n_features), "CNN+GRU", is_dl=True))
    results.append(evaluate_model(build_cnn_transformer(timesteps, n_features), "CNN+Transformer", is_dl=True))

    print("\n--- Generating Explainable AI ---")
    if trained_xgb_model:
        # Save SHAP
        explainer = shap.TreeExplainer(trained_xgb_model)
        shap_values = explainer.shap_values(X_ton_test[:500])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_ton_test[:500], feature_names=feature_names, show=False)
        save_plot("SHAP_Feature_Importance")

        
       

    # Final Summary Visualizations
    plot_combined_roc(roc_data)
    plot_model_comparison(results)
    plot_individual_metrics(results)

    print("\n" + "="*95)
    print(f"{'Model':<18} | {'Final Test Accuracy':<23} | {'AUC':<10} | {'Train(s)':<12} | {'Test(s)':<10}")
    print("-" * 95)
    for res in results:
        print(f"{res['Model']:<18} | {res['Accuracy']*100:>15.2f}%         | {res['AUC']:>10.4f} | {res['TrainTime(s)']:>12.2f} | {res['TestTime(s)']:>10.2f}")
    print("="*95)
    print(f"\nALL PLOTS AND REPORTS SUCCESSFULLY SAVED TO: {BASE_PATH}")

if __name__ == "__main__":
    main()