import os
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents plot popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def save_eda_plots(df, y, plot_dir):
    """Generates and saves Countplot, Correlation Heatmap, Boxplot, and Histogram before sampling."""
    print("Saving EDA plots (Pre-sampling)...")
    
    # 1. Count plot for class distribution (Fixed to avoid Seaborn errors)
    plt.figure(figsize=(10, 6))
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    counts = y.value_counts().sort_index()
    labels = counts.index.astype(str)
    values = counts.values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    
    plt.bar(labels, values, color=colors)
    plt.title("Class Distribution (Before Balancing)")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_class_distribution_raw.png"))
    plt.close()

    # Sample data for heavy plots to avoid memory crash
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    
    # 2. Correlation Heatmap (Top 15 features)
    corr = df_sample.iloc[:, :15].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Top 15 Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_correlation_heatmap.png"))
    plt.close()

    # 3. Boxplot (First 5 features to check outliers)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_sample.iloc[:, :5], palette="Set2")
    plt.title("Boxplot of First 5 Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_boxplot.png"))
    plt.close()

    # 4. Histogram (First 6 features)
    df_sample.iloc[:, :6].hist(figsize=(12, 10), bins=30, color='skyblue', edgecolor='black')
    plt.suptitle("Histogram of First 6 Features")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_histogram.png"))
    plt.close()

def load_and_preprocess(file_path, target_column, plot_dir):
    print(f"Opening dataset: {file_path}")
    
    # Memory optimization with systematic chunking
    total_rows = 27000000 
    chunk_size = 1000000 
    chunks = []
    
    print("Scanning dataset for multi-class diversity...")
    for i in range(0, total_rows, chunk_size):
        try:
            # Taking 50,000 samples from every 1 million rows
            chunk = pd.read_csv(file_path, skiprows=i, nrows=100000, low_memory=True)
            if i == 0:
                headers = [c.strip() for c in chunk.columns]
            chunk.columns = headers
            chunks.append(chunk)
        except Exception as e:
            break
            
    df = pd.concat(chunks, axis=0)
    df.columns = df.columns.str.strip()
    del chunks
    gc.collect() 
    
    print(f"Initial Dataset shape after chunking: {df.shape}")
    
    # Encode categorical target -  added to fix the factorize tuple issue
    print(f"Unique classes in '{target_column}': {df[target_column].unique()}")
    target_series = pd.factorize(df[target_column])[0]
    y = pd.Series(target_series)
    
    # Leakage Fix: Drop Timestamp, IP, and Port columns
    cols_to_drop = [
        'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 
        'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'FLOW_DURATION_MILLISECONDS'
    ]
    
    df_numeric = df.select_dtypes(include=[np.number])
    existing_drops = [c for c in cols_to_drop if c in df_numeric.columns]
    df_numeric.drop(columns=existing_drops, inplace=True, errors='ignore')
    print(f"Dropped columns to prevent data leakage: {existing_drops}")

    X = df_numeric
    if target_column in X.columns:
        X = X.drop(columns=[target_column])
    y = target_series
    
    del df, df_numeric
    gc.collect()

    # Handle missing values and inf
    print("Handling missing and infinite values...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X = X.astype('float32')

    # Save initial plots
    save_eda_plots(X, y, plot_dir)

    # Train, Validation, Test Split (Stratified to maintain class ratio)
    print("Splitting data into Train, Validation, and Test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Scale the features (Fit only on Train to prevent Data Leakage)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    del X_temp
    gc.collect()

    # Returning un-sampled scaled data
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X.columns