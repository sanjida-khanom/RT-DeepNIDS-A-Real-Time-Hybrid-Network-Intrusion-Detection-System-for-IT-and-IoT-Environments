# train.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold
import shap
import joblib

from preprocess import load_and_preprocess
from models import build_random_forest, build_decision_tree, build_xgboost, build_cnn_gru

# --------- Directory Setup ---------
DATA_DIR = r"C:\Capstone Project\CIC-IDS-2017\MachineLearningCVE"
PLOT_DIR = r"C:\Capstone Project\CIC-IDS-2017\plot"

# New Export Directories
EXPORT_DIR = r"C:\Capstone Project\Real_Time_Export"
DATASET_NAME = "CIC_IDS_2017"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --------- Helper Functions ---------
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def plot_roc(y_test_bin, y_prob, title, filename, n_classes):
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i}')
    plt.plot([0, 1], [0, 1], 'k--') # Fixed strictly for standard ROC
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def save_class_distribution(y, title, filename):
    plt.figure(figsize=(10, 5))
    counts = pd.Series(y).value_counts().sort_index()
    counts.plot(kind='bar', color=plt.cm.tab20(np.linspace(0, 1, len(counts))))
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def save_supporting_files(export_path, scaler, target_names, X_test_raw):
    os.makedirs(export_path, exist_ok=True)
    model_types = {'DT': 'ML', 'RF': 'ML', 'XGB': 'ML', 'CNN_GRU': 'DL'}
    joblib.dump(scaler, os.path.join(export_path, "scaler.pkl"))
    joblib.dump(target_names, os.path.join(export_path, "target_names.pkl"))
    joblib.dump(model_types, os.path.join(export_path, "model_types.pkl"))
    X_test_raw.head(100).to_csv(os.path.join(export_path, "live_traffic_sample.csv"), index=False)

# =============================================================
# ===================== MAIN PIPELINE =========================
# =============================================================

def run_pipeline(X_train, y_train, X_test, y_test, method_name, export_path):
    print(f"\n========== RUNNING PIPELINE WITH {method_name.upper()} ==========")
    model_names = ['RF', 'DT', 'XGB', 'CNN+GRU']
    accuracies, train_times, test_times, auc_scores = [], [], [], []

    classes = np.unique(y_train)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    # For CNN+GRU sequence shaping
    timesteps = 1
    n_features = X_train.shape[1]
    X_train_seq = X_train.reshape((-1, timesteps, n_features))
    X_test_seq = X_test.reshape((-1, timesteps, n_features))

    models_dict = {
        'RF': build_random_forest(),
        'DT': build_decision_tree(),
        'XGB': build_xgboost(),
        'CNN+GRU': build_cnn_gru(timesteps, n_features, n_classes)
    }

    best_model = None
    best_acc = 0

    for name, model in models_dict.items():
        print(f"\n--- Training {name} ({method_name}) ---")
        start_train = time.time()

        if name == 'CNN+GRU':
            model.fit(X_train_seq, y_train, epochs=10, batch_size=256, verbose=1)
        else:
            model.fit(X_train, y_train)

        train_duration = time.time() - start_train

        print(f"--- Testing {name} ---")
        start_test = time.time()

        if name == 'CNN+GRU':
            y_prob = model.predict(X_test_seq)
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

        test_duration = time.time() - start_test

        acc = accuracy_score(y_test, y_pred)

        # Calculate AUC safely
        try:
            auc_s = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
        except:
            auc_s = 0

        print(f"{name} Train Time: {train_duration:.2f}s | Test Time: {test_duration:.2f}s | Acc: {acc:.4f}")

        if name == 'XGB' and acc > best_acc:
            best_model = model
            best_acc = acc

        accuracies.append(acc)
        train_times.append(train_duration)
        test_times.append(test_duration)
        auc_scores.append(auc_s)

        # Generate Classification Report
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(PLOT_DIR, f"{name}_{method_name}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report for {name} ({method_name})\n\n{report}")

        # Name mapping for exact Dashboard matching
        name_map = {
            'Decision Tree': 'DT',
            'Random Forest': 'RF',
            'XGBoost': 'XGB',
            'CNN+GRU': 'CNN_GRU',
            'Hybrid CNN-GRU': 'CNN_GRU',
            'DT': 'DT',
            'RF': 'RF',
            'XGB': 'XGB'
        }
        short_name = name_map.get(name, name)

        # Visualizations (Fixed parameters)
        plot_confusion(y_test, y_pred, f"{name} ({method_name})", f"{name}_{method_name}_confusion.png")
        if short_name != 'CNN_GRU':
            plot_roc(y_test_bin, y_prob, f"{name} ({method_name})", f"{name}_{method_name}_roc.png", n_classes)

        # Export Model
        if short_name == 'CNN_GRU':
            model.save(os.path.join(export_path, "CNN_GRU_model.h5"))
        else:
            joblib.dump(model, os.path.join(export_path, f"{short_name}_model.pkl"))

    return model_names, accuracies, train_times, test_times, best_model

def plot_final_comparison(results_smote, results_tomek, metric, ylabel, filename):
    models = results_smote['models']
    smote_vals = results_smote[metric]
    tomek_vals = results_tomek[metric]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, smote_vals, width, label='SMOTE')
    ax.bar(x + width/2, tomek_vals, width, label='Tomek+IHT')

    ax.set_xlabel('Models')
    ax.set_ylabel(ylabel)
    ax.set_title(f'SMOTE vs Tomek+IHT: {ylabel}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

if __name__ == "__main__":
    # 1. Load Data
    X_train_raw, X_test_scaled, y_train_raw, y_test, feature_names, scaler, target_names, X_test_raw = load_and_preprocess(
        DATA_DIR, PLOT_DIR)

    results_dict = []

    # ================= SMOTE PIPELINE =================
    print("\nApplying SMOTE...")
    smote_path = os.path.join(EXPORT_DIR, "SMOTE", DATASET_NAME)
    save_supporting_files(smote_path, scaler, target_names, X_test_raw)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
    save_class_distribution(y_train_smote, "Class Distribution After SMOTE", "countplot_after_smote.png")

    models_s, acc_s, train_t_s, test_t_s, best_model_smote = run_pipeline(
        X_train_smote, y_train_smote, X_test_scaled, y_test, "SMOTE", smote_path
    )
    for i in range(len(models_s)):
        results_dict.append({"Method": "SMOTE", "Model": models_s[i], "Accuracy": acc_s[i],
                             "Train_Time_s": train_t_s[i], "Test_Time_s": test_t_s[i]})

    # ================= SHAP EXPLAINER =================
    if best_model_smote is not None:
        print("\nApplying SHAP on XGBoost (SMOTE Pipeline)...")
        X_sample = pd.DataFrame(X_test_scaled[:500], columns=feature_names)
        explainer = shap.TreeExplainer(best_model_smote)
        shap_values = explainer.shap_values(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Summary Plot")
        plt.savefig(os.path.join(PLOT_DIR, "shap_summary_smote.png"), bbox_inches='tight')
        plt.close()

    # ================= TOMEK + IHT PIPELINE =================
    print("\nApplying Tomek Links + IHT on RAW data...")
    tomek_path = os.path.join(EXPORT_DIR, "Tomek_IHT", DATASET_NAME)
    save_supporting_files(tomek_path, scaler, target_names, X_test_raw)

    tomek = TomekLinks()
    X_train_tomek, y_train_tomek = tomek.fit_resample(X_train_raw, y_train_raw)

    iht = InstanceHardnessThreshold(random_state=42)
    X_train_tomek_iht, y_train_tomek_iht = iht.fit_resample(X_train_tomek, y_train_tomek)

    save_class_distribution(y_train_tomek_iht, "Class Distribution After Tomek+IHT", "countplot_after_tomek_iht.png")

    models_t, acc_t, train_t_t, test_t_t, _ = run_pipeline(
        X_train_tomek_iht, y_train_tomek_iht, X_test_scaled, y_test, "Tomek_IHT", tomek_path
    )

    for i in range(len(models_t)):
        results_dict.append({"Method": "Tomek+IHT", "Model": models_t[i], "Accuracy": acc_t[i],
                             "Train_Time_s": train_t_t[i], "Test_Time_s": test_t_t[i]})

    # ================= FINAL COMPARISONS =================
    print("\nGenerating Final Comparison Plots...")
    smote_res = {'models': models_s, 'accuracy': acc_s, 'train_time': train_t_s}
    tomek_res = {'models': models_t, 'accuracy': acc_t, 'train_time': train_t_t}

    plot_final_comparison(smote_res, tomek_res, 'accuracy', 'Accuracy Score', 'final_accuracy_comparison.png')
    plot_final_comparison(smote_res, tomek_res, 'train_time', 'Training Time (s)', 'final_traintime_comparison.png')

    # Save final CSV
    final_df = pd.DataFrame(results_dict)
    csv_path = os.path.join(PLOT_DIR, "final_models_comparison.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"\nExecution Complete! All results, reports, and plots are saved to: {PLOT_DIR}")
    print(f"Exported Live Testing models to: {EXPORT_DIR}")