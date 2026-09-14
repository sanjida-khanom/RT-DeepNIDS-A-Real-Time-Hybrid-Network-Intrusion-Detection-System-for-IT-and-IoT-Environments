import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_random_forest():
    """Builds a Random Forest model. Max depth limited to prevent overfitting."""
    return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

def build_decision_tree():
    """Builds a Decision Tree baseline model."""
    return DecisionTreeClassifier(max_depth=10, random_state=42)

def build_xgboost():
    """Builds an XGBoost model optimized for binary classification."""
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, eval_metric='logloss', random_state=42, n_jobs=-1)

def build_cnn_gru(timesteps, n_features):
    """Builds a CNN + GRU Hybrid Model with high dropout for cross-dataset generalization."""
    inp = layers.Input(shape=(timesteps, n_features))
    x = layers.Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_transformer(timesteps, n_features):
    """Builds a CNN + Transformer (Self-Attention) Hybrid Model."""
    inp = layers.Input(shape=(timesteps, n_features))
    x = layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attention_output = layers.Dropout(0.4)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    ffn = layers.Dense(64, activation='relu')(x)
    ffn = layers.Dropout(0.3)(ffn)
    ffn = layers.Dense(64)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model