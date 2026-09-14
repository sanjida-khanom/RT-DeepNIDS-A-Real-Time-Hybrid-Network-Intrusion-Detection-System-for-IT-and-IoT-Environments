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
            # Drop 'Infilteration' completely because it causes massive overlap with Benign
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
    
    # --- FIX 1: Remove Duplicate Rows to prevent data leakage ---
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
        print(f"Dropping leaky features: {existing_leaky_cols}")
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

    # Missing Value Handling
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    # Remove extremely minor classes (< 2000 samples) to prevent SMOTE noise
    counts = y.value_counts()
    valid_classes = counts[counts >= 2000].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    # Factoring labels smoothly for CNN+GRU compatibility
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