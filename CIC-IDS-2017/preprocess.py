# preprocess.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents plots from popping up in Windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def create_plot_dir(plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

def load_and_preprocess(folder_path, plot_dir, target_column="Label"):
    create_plot_dir(plot_dir)
    print(f"Reading CSV files from: {folder_path}")
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    df_list = []
    for filename in all_files:
        print(f"Loading {os.path.basename(filename)}...")
        temp_df = pd.read_csv(filename, encoding='latin-1', low_memory=True)
        temp_df.columns = temp_df.columns.str.strip()
        df_list.append(temp_df)
    
    print("Concatenating all files...")
    df = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    
    print(f"Original Dataset Shape: {df.shape}")
    
    # -------- Handling Missing Values & Data Types --------
    # Optimize memory by converting float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    # Replace infinity values and fill NaN with median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("Filling missing values with median...")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Apply Stratified Downsampling to prevent memory crashes
    print("Applying Stratified Downsampling to prevent memory crashes...")
    df = df.groupby(target_column, group_keys=False).apply(lambda x: x.sample(min(len(x), 100000), random_state=42))
    print(f"Downsampled Dataset Shape: {df.shape}")

    # Convert string labels into integers correctly using pd.factorize()
    df[target_column] = pd.factorize(df[target_column])[0]
    
    # -------- Initial Exploratory Data Analysis (EDA) Plots --------
    print("Generating Exploratory Data Analysis Plots...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 2. Boxplot
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[numeric_cols[:10]])
    plt.title("Boxplot of First 10 Features")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(plot_dir, "boxplot_before.png"))
    plt.close()

    # 3. Histogram
    plt.figure(figsize=(10, 5))
    df[numeric_cols].hist(bins=50) # Sample histogram for the first numeric column
    plt.title(f"Histogram of {numeric_cols}")
    plt.savefig(os.path.join(plot_dir, "histogram_before.png"))
    plt.close()

    # 4. Count Plot (Class Distribution Before Balancing)
    plt.figure(figsize=(10, 5))
    sns.countplot(x=target_column, data=df)
    plt.title("Class Distribution Before Balancing")
    plt.savefig(os.path.join(plot_dir, "countplot_before_balancing.png"))
    plt.close()

    # -------- Train / Test Split --------
    X = df.drop(columns=[target_column])
    y = df[target_column]
    del df
    
    print("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # -------- Scaling --------
    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Preprocessing completed successfully.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns