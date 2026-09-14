import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Build Random Forest
def build_random_forest():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Build Decision Tree
def build_decision_tree():
    return DecisionTreeClassifier(random_state=42)

# Build XGBoost
def build_xgboost():
    return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, eval_metric='mlogloss', random_state=42, n_jobs=-1)

# Build CNN+GRU
def build_cnn_gru(timesteps, n_features, n_classes):
    inp = layers.Input(shape=(timesteps, n_features))
    
    x = layers.Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model