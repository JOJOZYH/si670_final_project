#%% Import data
import pandas as pd

file_path = 'data.csv'
data = pd.read_csv(file_path, index_col=0)

#%% Preprocessing
data['pm2.5'] = data['pm2.5'].interpolate()

# Creating 12 new columns for future PM2.5 levels, 1 hour to 12 hours ahead
for i in range(1, 13):
    data[f'pm2.5_{i}_hour_after'] = data['pm2.5'].shift(-i)

# One-hot encode the 'cbwd' column
data = pd.get_dummies(data, columns=['cbwd'])

# Check missing rows
missing_rows_before = data.isna().any(axis=1).sum()
print(f"Missing rows before: {missing_rows_before}")

# Drop rows with missing target values
data.dropna(subset=['pm2.5'] + [f'pm2.5_{i}_hour_after' for i in range(1, 13)], inplace=True)

# Split indices
split_index_train = int(len(data) * 0.5)  # first 50%
split_index_test = int(len(data) * 0.85)  # next 35%, leaving last 15%

data_train = data.iloc[:split_index_train]
data_mid = data.iloc[split_index_train:split_index_test]
data_final = data.iloc[split_index_test:]

print(f"Training Data Size: {data_train.shape[0]}")
print(f"Intermediate Test Data Size: {data_mid.shape[0]}")
print(f"Final Data Size (15%): {data_final.shape[0]}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

########################################
# Data Preparation
########################################

future_targets = [f'pm2.5_{j}_hour_after' for j in range(1, 13)]

# Binary targets: pm2.5 > 50
Y_train = (data_train[future_targets].values > 50).astype(int)
Y_mid = (data_mid[future_targets].values > 50).astype(int)
Y_final = (data_final[future_targets].values > 50).astype(int)

X_train = data_train.drop(columns=future_targets + ['year'])
X_mid = data_mid.drop(columns=future_targets + ['year'])
X_final = data_final.drop(columns=future_targets + ['year'])

numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)
    ],
    remainder='passthrough'
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_mid_transformed = preprocessor.transform(X_mid)
X_final_transformed = preprocessor.transform(X_final)

# Keep track of feature names after preprocessing for later reference
# Since remainder='passthrough', the order is: numerical_columns followed by already numeric encoded cols
# But here we had only one-hot encoding of 'cbwd' done before this step, and year removed.
feature_names = list(numerical_columns) + [col for col in X_train.columns if col not in numerical_columns]

########################################
# Neural Network Model Definition
########################################
model = Sequential([
    Input(shape=(X_train_transformed.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(12, activation='sigmoid') # 12 outputs for 12 future hours
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

########################################
# Training the Model
########################################
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

X_train_fit, X_val, Y_train_fit, Y_val = train_test_split(
    X_train_transformed, Y_train, test_size=0.2, random_state=42
)

history = model.fit(
    X_train_fit, Y_train_fit,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

########################################
# Predictions on the Middle 35% Data
########################################
preds_mid = model.predict(X_mid_transformed)
preds_mid_binary = (preds_mid > 0.5).astype(int)

prediction_df_mid = pd.DataFrame(
    preds_mid_binary,
    columns=[f'pm2.5_{i}_hour_after_binary_pred' for i in range(1, 13)]
)

if 'No' in data_mid.columns:
    prediction_df_mid['No'] = data_mid['No'].values
    cols = ['No'] + [col for col in prediction_df_mid.columns if col != 'No']
    prediction_df_mid = prediction_df_mid[cols]

prediction_df_mid.to_csv('nn_predictions_on_35_percent.csv', index=False)
print("Predictions on the 35% segment saved to nn_predictions_on_35_percent.csv")

########################################
# Predictions on the Final 15% Data
########################################
preds_final = model.predict(X_final_transformed)
preds_final_binary = (preds_final > 0.5).astype(int)

prediction_df_final = pd.DataFrame(
    preds_final_binary,
    columns=[f'pm2.5_{i}_hour_after_binary_pred' for i in range(1, 13)]
)

if 'No' in data_final.columns:
    prediction_df_final['No'] = data_final['No'].values
    cols = ['No'] + [col for col in prediction_df_final.columns if col != 'No']
    prediction_df_final = prediction_df_final[cols]

prediction_df_final.to_csv('nn_predictions_on_15_percent.csv', index=False)
print("Predictions on the last 15% segment saved to nn_predictions_on_15_percent.csv")

########################################
# Plotting ROC Curves for the Final 15%
########################################

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

hours_ahead = range(1, 13)
plt.figure(figsize=(10, 6))

for i, hour in enumerate(hours_ahead):
    y_true = Y_final[:, i]
    y_score = preds_final[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.plot(fpr, tpr, lw=2,
             label=f'{hour} hours after (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves on Last 15% Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

########################################
# Feature Importance via Permutation Importance
########################################
# We need a sklearn-compatible estimator. We'll wrap our Keras model.

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, make_scorer

# Custom scoring function for multi-label accuracy
def multilabel_accuracy(y_true, y_pred):
    # Flatten to consider all labels equally
    return accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))

my_scorer = make_scorer(multilabel_accuracy)

class KerasModelWrapper:
    def __init__(self, keras_model):
        self.keras_model = keras_model
    def fit(self, X, y):
        pass  # Already trained
    def predict(self, X):
        preds = (self.keras_model.predict(X) > 0.5).astype(int)
        return preds

wrapped_model = KerasModelWrapper(model)

# Apply permutation importance on the final 15% data
result = permutation_importance(
    wrapped_model, X_final_transformed, Y_final, n_repeats=10, random_state=42, scoring=my_scorer
)

importances = result.importances_mean
std = result.importances_std

# Create a DataFrame for feature importance
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'std': std
}).sort_values('importance', ascending=False)

print("Feature Importances (Permutation):")
print(importances_df)

# Optional: Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importances_df['feature'], importances_df['importance'], xerr=importances_df['std'])
plt.gca().invert_yaxis()
plt.xlabel('Permutation Importance')
plt.title('Feature Importances on Last 15% Data')
plt.tight_layout()
plt.show()

from sklearn.metrics import f1_score, roc_curve, roc_auc_score

########################################
# F1 Scores for Each Hour on Final 15%
########################################

f1_scores = []
for i, hour in enumerate(hours_ahead):
    # True labels and predictions for this hour
    y_true = Y_final[:, i]
    y_pred = preds_final_binary[:, i]

    # Compute F1 score for this hour
    # By default, f1_score for binary classification is straightforward.
    # If you'd like, you can specify average='binary', but it's default for binary inputs.
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

    print(f'Hour {hour} Ahead - F1 Score: {f1:.4f}')

# If you'd like a macro-average F1 over all hours:
f1_macro = f1_score(Y_final.reshape(-1), preds_final_binary.reshape(-1))
print(f'\nOverall Macro F1 Score (All Hours Combined): {f1_macro:.4f}')