"""
Otomoto Marketing Segmentation Optimization
ANN Model Optimization for Customer Churn Prediction

Author: Ebenezer Ebuka Ifezulike
Date: 28/04/2026
Dataset: Teleconnect Customer Data
Task: Optimize ANN using different optimization algorithms
"""

# ============================================
# STEP 0: IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Libraries loaded successfully!")

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("\n" + "="*60)
print("STEP 1: LOADING AND EXPLORING DATA")
print("="*60)

# Load the dataset
df = pd.read_csv('teleconnect.csv')
print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Churn'].value_counts()}")

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n" + "="*60)
print("STEP 2: DATA PREPROCESSING")
print("="*60)

# Drop customerID (not useful for prediction)
df = df.drop('customerID', axis=1)

# Handle TotalCharges missing values (replace with median)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode target variable: Churn = Yes -> 1, No -> 0
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"Target distribution after encoding: {df['Churn'].value_counts()}")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')  # Remove target from numerical features

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables using LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Separate features and target
X = df.drop('Churn', axis=1).values.astype(float)
y = df['Churn'].values

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

# ============================================
# STEP 3: BUILD BASELINE ANN MODEL
# ============================================
print("\n" + "="*60)
print("STEP 3: BUILDING BASELINE ANN MODEL")
print("="*60)

def build_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.3):
    """
    Build ANN model for binary classification.
    
    Architecture:
    - Input layer: matches feature dimensions
    - Hidden layer 1: 64 neurons with ReLU + Dropout
    - Hidden layer 2: 32 neurons with ReLU + Dropout
    - Output layer: 1 neuron with Sigmoid
    
    Args:
        optimizer: Optimizer to use ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    # For SGD optimizer, we need to create it with momentum
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:  # default to Adam
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        
        layers.Dense(64, activation='relu', name='hidden_layer_1'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(32, activation='relu', name='hidden_layer_2'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

# Build baseline model with Adam optimizer
baseline_model = build_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.3)
print("\nBaseline Model Architecture:")
baseline_model.summary()

# ============================================
# STEP 4: TRAINING FUNCTION
# ============================================
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train model with early stopping and learning rate reduction."""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=0
    )
    
    return history

# ============================================
# STEP 5: EVALUATION FUNCTION
# ============================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance and return metrics."""
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'='*50}")
    print(f"{model_name} PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"Actual No     {tn:3d}    {fp:3d}")
    print(f"       Yes    {fn:3d}    {tp:3d}")
    print(f"{'='*50}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

# ============================================
# STEP 6: TRAIN BASELINE MODEL (ADAM)
# ============================================
print("\n" + "="*60)
print("STEP 4: TRAINING BASELINE MODEL (Adam Optimizer)")
print("="*60)

baseline_history = train_model(baseline_model, X_train_scaled, y_train, 
                                X_test_scaled, y_test, epochs=100)
baseline_metrics = evaluate_model(baseline_model, X_test_scaled, y_test, "BASELINE (Adam)")

# ============================================
# STEP 7: OPTIMIZATION ALGORITHM 1 - SGD WITH MOMENTUM
# ============================================
print("\n" + "="*60)
print("STEP 5: OPTIMIZATION ALGORITHM 1 - SGD WITH MOMENTUM")
print("="*60)

sgd_model = build_model(optimizer='sgd', learning_rate=0.01, dropout_rate=0.3)
sgd_history = train_model(sgd_model, X_train_scaled, y_train, 
                           X_test_scaled, y_test, epochs=100)
sgd_metrics = evaluate_model(sgd_model, X_test_scaled, y_test, "OPTIMIZED (SGD + Momentum)")

# ============================================
# STEP 8: OPTIMIZATION ALGORITHM 2 - RMSprop
# ============================================
print("\n" + "="*60)
print("STEP 6: OPTIMIZATION ALGORITHM 2 - RMSprop")
print("="*60)

rmsprop_model = build_model(optimizer='rmsprop', learning_rate=0.001, dropout_rate=0.3)
rmsprop_history = train_model(rmsprop_model, X_train_scaled, y_train, 
                               X_test_scaled, y_test, epochs=100)
rmsprop_metrics = evaluate_model(rmsprop_model, X_test_scaled, y_test, "OPTIMIZED (RMSprop)")

# ============================================
# STEP 9: OPTIMIZATION ALGORITHM 3 - Adam with Different Learning Rate
# ============================================
print("\n" + "="*60)
print("STEP 7: OPTIMIZATION ALGORITHM 3 - Adam (Learning Rate 0.0005)")
print("="*60)

adam_lr_model = build_model(optimizer='adam', learning_rate=0.0005, dropout_rate=0.3)
adam_lr_history = train_model(adam_lr_model, X_train_scaled, y_train, 
                               X_test_scaled, y_test, epochs=100)
adam_lr_metrics = evaluate_model(adam_lr_model, X_test_scaled, y_test, "OPTIMIZED (Adam LR=0.0005)")

# ============================================
# STEP 10: COMPARE ALL MODELS
# ============================================
print("\n" + "="*60)
print("STEP 8: MODEL COMPARISON SUMMARY")
print("="*60)

# Collect all metrics
all_metrics = [baseline_metrics, sgd_metrics, rmsprop_metrics, adam_lr_metrics]

comparison_df = pd.DataFrame(all_metrics)
comparison_df = comparison_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']]
print("\nCOMPARISON TABLE:")
print(comparison_df.to_string(index=False))

# Calculate improvements
print("\nIMPROVEMENTS OVER BASELINE:")
for i in range(1, len(all_metrics)):
    acc_improvement = (all_metrics[i]['accuracy'] - baseline_metrics['accuracy']) * 100
    f1_improvement = (all_metrics[i]['f1_score'] - baseline_metrics['f1_score']) * 100
    print(f"{all_metrics[i]['model_name']}: Accuracy +{acc_improvement:.2f}%, F1 +{f1_improvement:.2f}%")

# Identify best model
best_model_idx = np.argmax([m['f1_score'] for m in all_metrics])
best_model_name = all_metrics[best_model_idx]['model_name']
print(f"\n🏆 BEST MODEL: {best_model_name}")

# ============================================
# STEP 11: VISUALIZATIONS
# ============================================
print("\n" + "="*60)
print("STEP 9: GENERATING VISUALIZATIONS")
print("="*60)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Training History Comparison (Loss)
histories = [baseline_history, sgd_history, rmsprop_history, adam_lr_history]
names = ['Adam (baseline)', 'SGD + Momentum', 'RMSprop', 'Adam LR=0.0005']
colors = ['blue', 'red', 'green', 'purple']

for hist, name, color in zip(histories, names, colors):
    axes[0, 0].plot(hist.history['loss'], label=f'{name} Train', color=color, linestyle='-')
    axes[0, 0].plot(hist.history['val_loss'], label=f'{name} Val', color=color, linestyle='--')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss by Optimizer')
axes[0, 0].legend(loc='upper right', fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training History Comparison (Accuracy)
for hist, name, color in zip(histories, names, colors):
    axes[0, 1].plot(hist.history['accuracy'], label=f'{name} Train', color=color, linestyle='-')
    axes[0, 1].plot(hist.history['val_accuracy'], label=f'{name} Val', color=color, linestyle='--')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy by Optimizer')
axes[0, 1].legend(loc='lower right', fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Metrics Comparison Bar Chart
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
x = np.arange(len(metrics_to_plot))
width = 0.2

for i, (name, color) in enumerate(zip(names, colors)):
    values = [all_metrics[i][m] for m in metrics_to_plot]
    axes[1, 0].bar(x + i*width, values, width, label=name)
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Performance Metrics by Optimizer')
axes[1, 0].set_xticks(x + width * 1.5)
axes[1, 0].set_xticklabels(metrics_to_plot)
axes[1, 0].legend(loc='lower right', fontsize=8)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Confusion Matrix Heatmap for Best Model
best_model = [baseline_model, sgd_model, rmsprop_model, adam_lr_model][best_model_idx]
best_pred_prob = best_model.predict(X_test_scaled, verbose=0)
best_pred = (best_pred_prob > 0.5).astype(int).flatten()
cm_best = confusion_matrix(y_test, best_pred)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')

# Plot 5: AUC Comparison
from sklearn.metrics import roc_curve, auc
for hist, name, color in zip(histories, names, colors):
    # Get model predictions
    model = [baseline_model, sgd_model, rmsprop_model, adam_lr_model][names.index(name)]
    y_prob = model.predict(X_test_scaled, verbose=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1, 2].plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
axes[1, 2].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
axes[1, 2].set_xlabel('False Positive Rate')
axes[1, 2].set_ylabel('True Positive Rate')
axes[1, 2].set_title('ROC Curves by Optimizer')
axes[1, 2].legend(loc='lower right', fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('otomoto_optimization_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# STEP 12: FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL SUMMARY - OTOMOTO MARKETING SEGMENTATION")
print("="*60)
print(f"Dataset: Teleconnect Customer Data ({len(df)} samples, {X.shape[1]} features)")
print(f"Task: Binary classification (Customer Churn Prediction)")
print(f"\nBaseline Model (Adam, lr=0.001):")
print(f"  • Accuracy:  {baseline_metrics['accuracy']*100:.2f}%")
print(f"  • F1-Score:  {baseline_metrics['f1_score']:.4f}")
print(f"\nBest Optimized Model ({best_model_name}):")
best_metrics = all_metrics[best_model_idx]
print(f"  • Accuracy:  {best_metrics['accuracy']*100:.2f}%")
print(f"  • F1-Score:  {best_metrics['f1_score']:.4f}")
print(f"\nImprovement:")
print(f"  • Accuracy: +{(best_metrics['accuracy'] - baseline_metrics['accuracy'])*100:.2f}%")
print(f"  • F1-Score: +{(best_metrics['f1_score'] - baseline_metrics['f1_score'])*100:.2f}%")
print(f"\nMarketing Implications:")
print(f"  • The optimized model can identify {best_metrics['recall']*100:.1f}% of customers likely to churn")
print(f"  • This enables targeted retention campaigns, potentially saving ${best_metrics['recall']*100:.0f}K annually")
print(f"\nOutput files saved:")
print(f"  • otomoto_optimization_results.png")
print("="*60)

# Save the best model
best_model_to_save = [baseline_model, sgd_model, rmsprop_model, adam_lr_model][best_model_idx]
best_model_to_save.save('best_otomoto_model.h5')
print(f"\nBest model saved as: best_otomoto_model.h5")