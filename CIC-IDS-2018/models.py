import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def build_random_forest():
    # FIX: Added max_depth and min_samples_split to prevent overfitting
    return RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=10, criterion='entropy', random_state=42, n_jobs=-1)

def build_decision_tree():
    # FIX: Added max_depth and min_samples_split to prevent overfitting
    return DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=10, random_state=42)

def build_xgboost():
    # FIX: Reduced max_depth to prevent overfitting
    return XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, tree_method='hist', random_state=42)

def build_cnn_gru(timesteps, n_features, n_classes):
    inp = layers.Input(shape=(timesteps, n_features))
    
    # CNN Layer
    x = layers.Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_normal')(inp)
    x = layers.BatchNormalization()(x)
    
    # Stacked GRU Block
    x = layers.GRU(128, return_sequences=True, dropout=0.2)(x)
    x = layers.GRU(128, return_sequences=False, dropout=0.2)(x)
    
    # Deep Dense Block
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    out = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model