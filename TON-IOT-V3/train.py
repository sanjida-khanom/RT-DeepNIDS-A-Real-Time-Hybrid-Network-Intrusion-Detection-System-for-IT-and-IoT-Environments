import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold

from preprocess import load_and_preprocess
from models import build_random_forest, build_decision_tree, build_xgboost, build_cnn_gru

# Define Directories
DATASET_PATH = r"D:\Capstone project\TON-IOT-V3\NF-ToN-IoT-v3.csv"
PLOT_DIR = r"D:\Capstone project\TON-IOT-V3\plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --------- Helper Functions ---------
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {title}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def save_roc_curve(y_test, y_prob, n_classes, title, filename):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        if y_prob.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([4], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def save_class_distribution(y, title, filename):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y, palette="Set1")
    plt.title(title)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# ===================== MAIN PIPELINE =====================
def run_pipeline():
    print("=== Starting Preprocessing Phase ===")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_and_preprocess(DATASET_PATH, "Attack", PLOT_DIR)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    
    all_results = [] # To store dicts for final CSV
    
    def evaluate_model(model, name, X_tr, y_tr, method_name, is_dl=False):
        print(f"\n--- Training {name} ({method_name}) ---")
        start_train = time.time()
        
        if is_dl:
            # Reshape for CNN+GRU
            X_tr_seq = X_tr.reshape((-1, 1, n_features))
            X_val_seq = X_val.reshape((-1, 1, n_features))
            X_test_seq = X_test.reshape((-1, 1, n_features))
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_tr_seq, y_tr, validation_data=(X_val_seq, y_val), epochs=50, batch_size=1024, callbacks=[early_stop], verbose=1)
        else:
            model.fit(X_tr, y_tr)
            
        train_duration = time.time() - start_train
        
        print(f"--- Testing {name} ({method_name}) ---")
        start_test = time.time()
        
        if is_dl:
            y_prob = model.predict(X_test_seq)
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
        test_duration = time.time() - start_test
        acc = accuracy_score(y_test, y_pred)
        
        y_test_bin = label_binarize(y_test, classes=classes)
        try:
            auc_score = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
        except:
            auc_score = 0.0

        print(f"Accuracy for {name}: {acc*100:.2f}%")
        
        # Save Plots and Reports
        safe_name = f"{name}_{method_name}".replace("+", "_")
        save_confusion_matrix(y_test, y_pred, f"{name} ({method_name})", f"{safe_name}_confusion_matrix.png")
        save_roc_curve(y_test, y_prob, n_classes, f"{name} ({method_name})", f"{safe_name}_roc_curve.png")
        
        # Save Classification Report
        report = classification_report(y_test, y_pred, digits=4)
        with open(os.path.join(PLOT_DIR, f"{safe_name}_classification_report.txt"), "w") as f:
            f.write(report)
            
        # Return model for SHAP (if needed)
        return {"Model": name, "Method": method_name, "Accuracy": acc, "Train_Time(s)": train_duration, "Test_Time(s)": test_duration, "AUC": auc_score}, model

    # ---------------------------------------------------------
    # PHASE 1: SMOTE
    # ---------------------------------------------------------
    print("\n=== PHASE 1: Applying SMOTE ===")
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    save_class_distribution(y_train_sm, "Class Distribution (After SMOTE)", "05_class_distribution_smote.png")
    
    models_to_run = {
        'RF': build_random_forest(),
        'DT': build_decision_tree(),
        'XGB': build_xgboost()
    }
    
    xgb_model_smote = None
    
    for m_name, m_func in models_to_run.items():
        res, trained_model = evaluate_model(m_func, m_name, X_train_sm, y_train_sm, "SMOTE")
        all_results.append(res)
        if m_name == 'XGB':
            xgb_model_smote = trained_model
            
    # Deep Learning Model
    cnn_model_sm = build_cnn_gru(1, n_features, n_classes)
    res_dl, _ = evaluate_model(cnn_model_sm, "CNN+GRU", X_train_sm, y_train_sm, "SMOTE", is_dl=True)
    all_results.append(res_dl)

    # Apply SHAP on XGBoost (SMOTE) - using sample to prevent extreme delays
    if xgb_model_smote:
        print("\n--- Applying SHAP for XGBoost (SMOTE) ---")
        X_test_sample = shap.sample(X_test, 500) # Small sample for speed
        explainer = shap.TreeExplainer(xgb_model_smote)
        shap_values = explainer.shap_values(X_test_sample)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(PLOT_DIR, "SHAP_Summary_XGB_SMOTE.png"), bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # PHASE 2: Tomek Links + IHT on RAW Train Data
    # ---------------------------------------------------------
    print("\n=== PHASE 2: Applying TomekLinks + IHT on Raw Data ===")
    # Note: IHT + Tomek Links can take significant time on large datasets
    tomek = TomekLinks()
    X_train_tmk, y_train_tmk = tomek.fit_resample(X_train, y_train)
    
    # IHT expects a classifier to estimate hardness (RandomForest is default and good)
    iht = InstanceHardnessThreshold(random_state=42, n_jobs=-1) 
    X_train_iht, y_train_iht = iht.fit_resample(X_train_tmk, y_train_tmk)
    save_class_distribution(y_train_iht, "Class Distribution (After Tomek+IHT)", "06_class_distribution_tomek_iht.png")

    models_to_run_2 = {
        'RF': build_random_forest(),
        'DT': build_decision_tree(),
        'XGB': build_xgboost()
    }
    
    for m_name, m_func in models_to_run_2.items():
        res, _ = evaluate_model(m_func, m_name, X_train_iht, y_train_iht, "Tomek_IHT")
        all_results.append(res)
        
    cnn_model_iht = build_cnn_gru(1, n_features, n_classes)
    res_dl, _ = evaluate_model(cnn_model_iht, "CNN+GRU", X_train_iht, y_train_iht, "Tomek_IHT", is_dl=True)
    all_results.append(res_dl)

    # ---------------------------------------------------------
    # PHASE 3: Final Comparison & CSV Export
    # ---------------------------------------------------------
    print("\n=== Generating Final Comparisons ===")
    results_df = pd.DataFrame(all_results)
    
    # Save CSV
    csv_path = os.path.join(PLOT_DIR, "final_results_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results successfully saved to {csv_path}")
    
    # Final Accuracy Comparison Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Method', palette='muted')
    plt.title('Final Comparison: SMOTE vs Tomek+IHT (Accuracy)')
    plt.ylim(0.8, 1.05) # Scaling y-axis to see differences for 99%+ accuracy clearly
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Final_Accuracy_Comparison.png"))
    plt.close()
    
    print("\nAll pipeline executions completed successfully. Check the 'plots' directory.")

if __name__ == "__main__":
    run_pipeline()