import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Prevents plots from popping up in Windows
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

        # --- BULLETPROOF COLUMN RENAMING ---
        temp_df.columns = temp_df.columns.str.strip()

        # Find label column
        actual_target = None
        for col in temp_df.columns:
            if 'label' in col.lower():
                actual_target = col
                break

        if not actual_target:
            actual_target = temp_df.columns[-1]

        temp_df.rename(columns={actual_target: "Label"}, inplace=True)

        df_list.append(temp_df)

    print("Concatenating all files...")
    df = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list

    target_column = "Label"
    print(f"Original Dataset Shape: {df.shape}")

    # -------- Handling Missing Values & Data Types --------
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    print("Filling missing values with median...")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # --- Safe Stratified Downsampling ---
    print("Applying Stratified Downsampling to prevent memory crashes...")
    sampled_chunks = []
    for label_val in df[target_column].unique():
        class_data = df[df[target_column] == label_val]
        sampled_chunks.append(class_data.sample(n=min(len(class_data), 100000), random_state=42))

    df = pd.concat(sampled_chunks, axis=0, ignore_index=True)
    print(f"Downsampled Dataset Shape: {df.shape}")

    # Convert string labels into integers and keep target names
    y_factorized, unique_classes = pd.factorize(df[target_column])
    df[target_column] = y_factorized
    target_names = list(unique_classes)

    # -------- Initial Exploratory Data Analysis (EDA) Plots --------
    print("Generating Exploratory Data Analysis Plots...")

    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()

    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[numeric_cols[:10]])
    plt.title("Boxplot of First 10 Features")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(plot_dir, "boxplot_before.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    df[numeric_cols[0]].hist(bins=50)
    plt.title(f"Histogram of {numeric_cols[0]}")
    plt.savefig(os.path.join(plot_dir, "histogram_before.png"))
    plt.close()

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

    # Returning required assets for Real-time export
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, scaler, target_names, X_test