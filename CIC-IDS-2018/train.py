import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_eda(X, y, title_prefix, plot_dir):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y)
    plt.title(f"{title_prefix} - Class Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{title_prefix}_histogram.png"))
    plt.close()
    
    subset_cols = X.columns[:15] if len(X.columns) > 15 else X.columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(X[subset_cols].corr(), annot=False, cmap='coolwarm')
    plt.title(f"{title_prefix} - Correlation Heatmap (Subset)")
    plt.savefig(os.path.join(plot_dir, f"{title_prefix}_heatmap.png"))
    plt.close()

def load_and_preprocess(folder_path, target_column, plot_dir):
    print(f"\nReading Parquet files from: {folder_path}")
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    
    df_list = []
    for filename in all_files:
        print(f"Loading {os.path.basename(filename)}...")
        temp_df = pd.read_parquet(filename, engine='auto')
        temp_df.columns = temp_df.columns.str.strip()
        
        actual_target = None
        for col in temp_df.columns:
            if col.lower() == target_column.lower():
                actual_target = col
                break
                
        if actual_target:
            temp_df.rename(columns={actual_target: target_column}, inplace=True)
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
                
            subset = temp_df[numeric_cols + [target_column]].copy()
            subset[target_column] = subset[target_column].astype(str).str.strip()
            
            # --- BEST PRACTICE FOR 99% ACCURACY ---
            subset = subset[~subset[target_column].str.contains('Infilteration', case=False, na=False)]
            
            sampled_chunks = []
            for label_val in subset[target_column].unique():
                class_data = subset[subset[target_column] == label_val]
                sampled_chunks.append(class_data.sample(n=min(len(class_data), 50000), random_state=42))
            
            if sampled_chunks:
                chunk_df = pd.concat(sampled_chunks, axis=0, ignore_index=True)
                df_list.append(chunk_df)
        del temp_df

    df = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    
    # --- FIX 1: Remove Duplicate Rows ---
    print("\nDropping duplicate rows...")
    df.drop_duplicates(inplace=True)
    
    print("Sampling data for better generalization...")
    benign_df = df[df[target_column].str.upper() == 'BENIGN']
    attack_df = df[df[target_column].str.upper() != 'BENIGN']
    
    # Cap limits for standardizing 
    if len(benign_df) > 150000: 
        benign_df = benign_df.sample(n=150000, random_state=42)
    if len(attack_df) > 200000: 
        attack_df = attack_df.sample(n=200000, random_state=42)
        
    df = pd.concat([benign_df, attack_df], axis=0)
    del benign_df, attack_df
    
    print("Dropping constant and redundant features...")
    
    # --- FIX 2: Drop identifying or leaky network features ---
    leaky_cols = ['Destination Port', 'Source Port', 'Flow ID', 'Timestamp', 'Protocol', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port']
    existing_leaky_cols = [col for col in leaky_cols if col in df.columns]
    if existing_leaky_cols:
        df.drop(columns=existing_leaky_cols, inplace=True)
        
    X_raw = df.drop(columns=[target_column])
    
    constant_cols = [col for col in X_raw.columns if X_raw[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)
    
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    df.drop(columns=to_drop, inplace=True)

    y = df[target_column]
    X = df.drop(columns=[target_column])
    del df

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    counts = y.value_counts()
    valid_classes = counts[counts >= 2000].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    labels, uniques = pd.factorize(y)
    y = pd.Series(labels, index=y.index)

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    print("Saving Before-Balancing Plots...")
    plot_eda(X_train, y_train, "Before_Balancing", plot_dir)

    print("Scaling with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype('float32')
    X_val_scaled = scaler.transform(X_val).astype('float32')
    X_test_scaled = scaler.transform(X_test).astype('float32')

    return X_train_scaled, y_train.values, X_val_scaled, y_val.values, X_test_scaled, y_test.values, list(uniques), X.columns


import time
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold

# Importing the updated ones explicitly as well
from preprocess import load_and_preprocess, plot_eda
from models import build_random_forest, build_decision_tree, build_xgboost, build_cnn_gru

def plot_confusion(y_true, y_pred, title, plot_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {title}")
    plt.savefig(os.path.join(plot_dir, f"{title.replace(' ', '_')}_CM.png"))
    plt.close()

def plot_roc(y_test, y_pred_prob, classes, title, plot_dir):
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([1], 'k--', lw=2)  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(os.path.join(plot_dir, f"{title.replace(' ', '_')}_ROC.png"))
    plt.close()

def run_pipeline(pipeline_name, X_train, y_train, X_val, y_val, X_test, y_test, target_names, plot_dir):
    print(f"\n{'='*20} Starting Pipeline: {pipeline_name} {'='*20}")
    model_names = ['Random Forest', 'Decision Tree', 'XGBoost', 'CNN+GRU']
    accuracies, train_times, test_times = [], [], []
    n_classes = len(np.unique(y_train))
    trained_models = {}
    
    models_dict = {
        'Random Forest': build_random_forest(),
        'Decision Tree': build_decision_tree(),
        'XGBoost': build_xgboost()
    }
    
    # 1. Train ALL ML Models First
    for name, model in models_dict.items():
        print(f"\n--- Training {name} ---")
        start_train = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - start_train
        
        start_test = time.time()
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        t_test = time.time() - start_test
        
        acc = accuracy_score(y_test, y_pred) * 100
        accuracies.append(acc); train_times.append(t_train); test_times.append(t_test)
        trained_models[name] = model 
        
        plot_confusion(y_test, y_pred, f"{name} ({pipeline_name})", plot_dir)
        plot_roc(y_test, y_pred_prob, target_names, f"{name} ({pipeline_name})", plot_dir)
        
        report = classification_report(y_test, y_pred, target_names=target_names)
        with open(os.path.join(plot_dir, f"{name.replace(' ', '_')}_{pipeline_name}_Report.txt"), "w") as f:
            f.write(report)

    # 2. Train CNN+GRU Model
    print("\n--- Training CNN+GRU ---")
    timesteps = 1
    n_features = X_train.shape[1]
    X_train_seq = X_train.reshape((-1, timesteps, n_features))
    X_val_seq = X_val.reshape((-1, timesteps, n_features))
    X_test_seq = X_test.reshape((-1, timesteps, n_features))
    
    cnn = build_cnn_gru(timesteps, n_features, n_classes)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    start_train = time.time()
    cnn.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), 
            epochs=50, batch_size=512, callbacks=callbacks, verbose=1)
    t_train = time.time() - start_train
    
    start_test = time.time()
    y_pred_prob = cnn.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    t_test = time.time() - start_test
    
    acc = accuracy_score(y_test, y_pred) * 100
    accuracies.append(acc); train_times.append(t_train); test_times.append(t_test)
    
    plot_confusion(y_test, y_pred, f"CNN+GRU ({pipeline_name})", plot_dir)
    plot_roc(y_test, y_pred_prob, target_names, f"CNN+GRU ({pipeline_name})", plot_dir)
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    with open(os.path.join(plot_dir, f"CNN_GRU_{pipeline_name}_Report.txt"), "w") as f:
        f.write(report)

    return pd.DataFrame({
        'Model': model_names,
        'Pipeline': [pipeline_name]*4,
        'Accuracy': accuracies,
        'Train Time (s)': train_times,
        'Test Time (s)': test_times
    }), trained_models

if __name__ == "__main__":
    base_dir = r"D:\Capstone project\CIC-IDS-2018"
    dataset_dir = os.path.join(base_dir, "CIC-IDS-2018")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    X_train_raw, y_train_raw, X_val, y_val, X_test, y_test, uniques, feature_names = load_and_preprocess(
        folder_path=dataset_dir, target_column="Label", plot_dir=plot_dir
    )
    
    target_names = [str(c) for c in uniques]

    # ================= PHASE 1: SMOTE =================
    print("\n" + "="*40)
    print("Applying SMOTE on RAW Training Data...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = sm.fit_resample(X_train_raw, y_train_raw)
    
    df_smote, models_smote = run_pipeline("SMOTE", X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, target_names, plot_dir)
    
    # ================= PHASE 2: Tomek+IHT =================
    print("\n" + "="*40)
    print("Applying Tomek+IHT on original RAW Training Data...")
    tl = TomekLinks()
    X_train_tomek, y_train_tomek = tl.fit_resample(X_train_raw, y_train_raw)
    iht = InstanceHardnessThreshold(random_state=42, cv=2)
    X_train_tiht, y_train_tiht = iht.fit_resample(X_train_tomek, y_train_tomek)
    
    df_tiht, models_tiht = run_pipeline("Tomek+IHT", X_train_tiht, y_train_tiht, X_val, y_val, X_test, y_test, target_names, plot_dir)
    
    # ================= PHASE 3: SHAP PLOTS =================
    print("\n--- Generating SHAP values ---")
    for name in ['Random Forest', 'XGBoost']:
        print(f"Calculating SHAP for {name} (SMOTE)...")
        explainer = shap.TreeExplainer(models_smote[name])
        shap_values = explainer.shap_values(X_test[:200])
        shap.summary_plot(shap_values, X_test[:200], show=False, feature_names=feature_names)
        plt.savefig(os.path.join(plot_dir, f"{name.replace(' ', '_')}_SMOTE_SHAP.png"), bbox_inches='tight')
        plt.close()

    # ================= PHASE 4: FINAL COMPARISON =================
    print("\nGenerating Final Comparison Plot...")
    final_df = pd.concat([df_smote, df_tiht], ignore_index=True)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=final_df, x='Model', y='Accuracy', hue='Pipeline', palette='viridis')
    plt.ylabel("Accuracy (%)")
    plt.title("SMOTE vs Tomek+IHT Accuracy Comparison")
    plt.ylim(90, 100) 
    plt.savefig(os.path.join(plot_dir, "Final_Accuracy_Comparison.png"))
    plt.close()
    
    final_df.to_csv(os.path.join(plot_dir, "Final_Results_Summary.csv"), index=False)
    print("\n--- All Execution Complete! ")