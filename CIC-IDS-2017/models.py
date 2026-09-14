# models.py
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- Classical Models ----------
def build_random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

def build_decision_tree():
    return DecisionTreeClassifier(random_state=42)

def build_xgboost():
    return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, eval_metric='mlogloss', random_state=42)

# ---------- Deep Learning: CNN + GRU ----------
def build_cnn_gru(timesteps, n_features, n_classes):
    inp = layers.Input(shape=(timesteps, n_features))
    x = layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.GRU(64)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model