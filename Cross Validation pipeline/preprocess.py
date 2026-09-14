import matplotlib
matplotlib.use('Agg') 

import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import TomekLinks, InstanceHardnessThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

BASE_PATH = r"D:\Capstone project\Cross Validation pipeline\plots"
os.makedirs(BASE_PATH, exist_ok=True)

# ----------------- VISUALIZATION HELPERS -----------------
def save_plot(title):
    """Saves plots directly to disk and clears RAM."""
    file_path = os.path.join(BASE_PATH, f"{title.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.clf()
    plt.close('all')

def plot_class_distribution(y, title):
    plt.figure(figsize=(8, 5))
    if isinstance(y, np.ndarray): y = pd.Series(y)
    sns.countplot(x=y, hue=y, palette="coolwarm", legend=False)
    plt.title(title)
    plt.xlabel("Class (0: Benign, 1: Attack)")
    plt.ylabel("Count")
    save_plot(title)

def plot_correlation_heatmap(df, title):
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) == 0: return
    corr = numeric_df.sample(n=min(10000, len(numeric_df)), random_state=42).corr()
    sns.heatmap(corr, cmap="RdBu_r", annot=False, fmt=".2f", vmin=-1, vmax=1)
    plt.title(title)
    save_plot(title)

def plot_hist_box(df, title_prefix):
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) == 0: return
    top_features = numeric_df.var().nlargest(min(5, len(numeric_df.columns))).index
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(numeric_df[col].sample(n=min(10000, len(numeric_df))), bins=30, kde=True, color='skyblue')
        plt.title(f"Histogram: {col[:15]}")
    save_plot(f"{title_prefix}_Histograms")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=numeric_df[col].sample(n=min(10000, len(numeric_df))), color='lightgreen')
        plt.title(f"Boxplot: {col[:15]}")
    save_plot(f"{title_prefix}_Boxplots")

def plot_all_eda(df, y, stage_name):
    """Generates and saves all requested plots for a specific stage."""
    print(f"   -> Saving EDA plots for: {stage_name}")
    plot_class_distribution(y, f"{stage_name}_Class_Distribution")
    
    # Drop labels from DF for correlation/histograms if present
    df_features = df.select_dtypes(include=[np.number])
    if not df_features.empty:
        plot_correlation_heatmap(df_features, f"{stage_name}_Correlation_Heatmap")
        plot_hist_box(df_features, f"{stage_name}")

# ----------------- DATA PREPROCESSING HELPERS -----------------
def clean_labels(series):
    series = series.astype(str).str.lower().str.strip()
    return np.where(series.str.contains('benign|normal|^0$', regex=True), 0, 1)

def drop_leakage_columns(df):
    leakage_keywords = ['ip', 'mac', 'time', 'timestamp', 'flow_id', 'flow id', 'unnamed', 'port']
    cols_to_drop = [col for col in df.columns if any(key in col.lower() for key in leakage_keywords)]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def align_features(df_train, df_test):
    df_train.columns = df_train.columns.str.strip().str.lower().str.replace(' ', '_')
    df_test.columns = df_test.columns.str.strip().str.lower().str.replace(' ', '_')
    mapping = {
        'protocol': 'protocol', 'flow_duration': 'flow_duration_milliseconds',
        'total_fwd_packets': 'in_pkts', 'tot_fwd_pkts': 'in_pkts',
        'total_backward_packets': 'out_pkts', 'tot_bwd_pkts': 'out_pkts',
        'total_length_of_fwd_packets': 'in_bytes', 'totlen_fwd_pkts': 'in_bytes',
        'total_length_of_bwd_packets': 'out_bytes', 'totlen_bwd_pkts': 'out_bytes',
        'fwd_packet_length_max': 'longest_flow_pkt', 'fwd_pkt_len_max': 'longest_flow_pkt',
        'fwd_packet_length_min': 'shortest_flow_pkt', 'fwd_pkt_len_min': 'shortest_flow_pkt'
    }
    df_train.rename(columns=mapping, inplace=True)
    common_features = list(set(df_train.columns) & set(df_test.columns))
    return df_train[common_features], df_test[common_features]

def apply_feature_engineering(X_train, X_test):
    var_thres = VarianceThreshold(threshold=0.01)
    var_thres.fit(X_train)
    constant_columns = [col for col in X_train.columns if col not in X_train.columns[var_thres.get_support()]]
    X_train.drop(columns=constant_columns, inplace=True, errors='ignore')
    X_test.drop(columns=constant_columns, inplace=True, errors='ignore')
    
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    X_train.drop(columns=to_drop, inplace=True, errors='ignore')
    X_test.drop(columns=to_drop, inplace=True, errors='ignore')
    return X_train, X_test

def load_csv_folder_sampled(folder_path, sample_fraction=0.10, chunksize=200000):
    chunks = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            full_path = os.path.join(folder_path, file)
            for chunk in pd.read_csv(full_path, chunksize=chunksize, low_memory=True):
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk.sample(frac=sample_fraction, random_state=42))
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else pd.DataFrame()

def load_parquet_folder_sampled(folder_path, sample_per_file=20000):
    chunks = []
    for file in [f for f in os.listdir(folder_path) if f.endswith('.parquet')]:
        try:
            df = pd.read_parquet(os.path.join(folder_path, file), engine='pyarrow')
            if len(df) > sample_per_file: df = df.sample(n=sample_per_file, random_state=42)
            chunks.append(df)
        except: pass
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else pd.DataFrame()

def load_sampled_csv(file_path, sample_fraction=0.05, chunksize=200000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=True):
        chunks.append(chunk.sample(frac=sample_fraction, random_state=42))
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else pd.DataFrame()

# ----------------- MAIN PIPELINE -----------------
def load_and_preprocess(path_2017, path_2018, path_ton_iot):
    print("\n>>> PREPARING SOURCE TRAINING DATA (CIC-IDS 2017 & 2018) <<<")
    df_2017 = load_csv_folder_sampled(path_2017, sample_fraction=0.10)
    y_2017_raw = clean_labels(df_2017.iloc[:, -1])
    plot_all_eda(df_2017.iloc[:, :-1], y_2017_raw, "1A_CIC_IDS_2017_Individual")
    
    df_2018 = load_parquet_folder_sampled(path_2018, sample_per_file=20000)
    y_2018_raw = clean_labels(df_2018.iloc[:, -1])
    plot_all_eda(df_2018.iloc[:, :-1], y_2018_raw, "1B_CIC_IDS_2018_Individual")
    
    y_2017 = clean_labels(df_2017.pop(df_2017.columns[-1]))
    y_2018 = clean_labels(df_2018.pop(df_2018.columns[-1]))
    X_train_raw = pd.concat([df_2017, df_2018], axis=0, ignore_index=True)
    y_train_raw = pd.concat([pd.Series(y_2017), pd.Series(y_2018)], axis=0, ignore_index=True)
    del df_2017, df_2018; gc.collect()

    print("\n>>> PREPARING TARGET DATA (ToN-IoT) <<<")
    X_ton_raw = load_sampled_csv(path_ton_iot, sample_fraction=0.05)
    target_ton = 'Attack' if 'Attack' in X_ton_raw.columns else X_ton_raw.columns[-1]
    y_ton_raw_plot = clean_labels(X_ton_raw[target_ton])
    plot_all_eda(X_ton_raw.drop(columns=[target_ton]), y_ton_raw_plot, "1C_ToN_IoT_Individual")
    
    y_ton_raw = clean_labels(X_ton_raw.pop(target_ton))

    print("\n--- Feature Alignment & Preprocessing ---")
    X_train_aligned, X_ton_aligned = align_features(X_train_raw, X_ton_raw)
    del X_train_raw, X_ton_raw; gc.collect()

    X_train_clean = drop_leakage_columns(X_train_aligned)
    X_ton_clean = drop_leakage_columns(X_ton_aligned)

    X_train_clean = X_train_clean.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    X_ton_clean = X_ton_clean.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

    X_train_clean, X_ton_clean = apply_feature_engineering(X_train_clean, X_ton_clean)
    feature_names = X_train_clean.columns.tolist()

    print("\n--- Generating Plots After Split & Processing ---")
    plot_all_eda(X_train_clean, y_train_raw, "2A_Train_Combined_After_Processing")
    plot_all_eda(X_ton_clean, y_ton_raw, "2B_Test_ToN_IoT_After_Processing")

    print("\n--- Scaling (RobustScaler) ---")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_ton_scaled = scaler.transform(X_ton_clean)

    print("\n--- Generating Plots BEFORE Tomek+IHT ---")
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    plot_all_eda(X_train_scaled_df, y_train_raw, "3_Train_BEFORE_Tomek_IHT")

    print("\n--- Applying Tomek Links & IHT ---")
    tl = TomekLinks(n_jobs=-1)
    X_train_tl, y_train_tl = tl.fit_resample(X_train_scaled, y_train_raw)
    
    iht = InstanceHardnessThreshold(estimator=RandomForestClassifier(n_estimators=10, random_state=42), random_state=42, n_jobs=-1)
    X_train_res, y_train_res = iht.fit_resample(X_train_tl, y_train_tl)

    print("\n--- Generating Plots AFTER Tomek+IHT ---")
    X_train_res_df = pd.DataFrame(X_train_res, columns=feature_names)
    plot_all_eda(X_train_res_df, y_train_res, "4_Train_AFTER_Tomek_IHT")

    print("\n--- Splitting ToN-IoT for Transfer Learning ---")
    X_ton_train, X_ton_test, y_ton_train, y_ton_test = train_test_split(
        X_ton_scaled, y_ton_raw, test_size=0.80, random_state=42, stratify=y_ton_raw
    )
    
    return X_train_res, y_train_res, X_ton_train, X_ton_test, y_ton_train, y_ton_test, feature_names