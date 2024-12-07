"""
SVC Classifier with Nested Cross-Validation
"""
#%% Import data
import pandas as pd

file_path = '../data/data.csv'
data = pd.read_csv(file_path, index_col=0)

#%% Preprocessing
data['pm2.5'] = data['pm2.5'].interpolate()

# Creating 12 new columns for future PM2.5 levels, 1 hour to 12 hours ahead
for i in range(1, 13):
    data[f'pm2.5_{i}_hour_after'] = data['pm2.5'].shift(-i)

# One-hot encode the 'cbwd' column
data = pd.get_dummies(data, columns=['cbwd'])

# Calculate the total number of rows with any missing values before dropping
missing_rows_before = data.isna().any(axis=1).sum()
print(f"Missing rows before: {missing_rows_before}")

# Drop rows where any cell from 'pm2.5' to 'pm2.5_12_hour_after' is missing
data.dropna(subset=['pm2.5'] + [f'pm2.5_{j}_hour_after' for j in range(1, 13)], inplace=True)

# Calculate the index to split the data at 85% for training and 15% for testing
split_index_train = int(len(data) * 0.5)
split_index_test = int(len(data) * 0.85)

# Split the data into training and test sets
data_train = data.iloc[:split_index_train]
data_test = data.iloc[split_index_train:split_index_test]

# Display sizes of the new datasets
print(f"Training Data Size: {data_train.shape[0]}")
print(f"Test Data Size: {data_test.shape[0]}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import joblib

# Prepare training and test data (drop the labels and 'year' columns)
X_train = data_train.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])
X_test = data_test.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])

# Define numerical columns
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Set up ColumnTransformer to scale numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)
    ],
    remainder='passthrough'
)

#%% Trainning
# Initialize lists to store overall results
f1_scores = []
roc_aucs = []
best_params = []  # To store best parameters for each hour

num_hours = 12  # Adjust based on loop range

# Define parameter grid for SVC
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf'],
    'classifier__gamma': ['scale', 'auto']
}

# Create a single plot for all ROC curves
plt.figure(figsize=(10, 8))

# Initialize a dictionary to store predictions
predictions = {}

for i in range(1, num_hours + 1):
    print(f"\nProcessing {i} hour(s) after...")

    # Define target variables
    y_train = (data_train[f'pm2.5_{i}_hour_after'] >= 50).astype(int)
    y_test = (data_test[f'pm2.5_{i}_hour_after'] >= 50).astype(int)

    # Split the training data into training and validation sets
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
    )

    # Define the outer and inner cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    # Create a pipeline with preprocessing and the SVC model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=1))
    ])

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    # Initialize metrics lists for nested CV
    nested_f1 = []
    nested_roc_auc = []
    current_best_params = []

    # Outer cross-validation loop
    for train_idx, valid_idx in outer_cv.split(X_train_split, y_train_split):
        X_outer_train, X_outer_valid = X_train_split.iloc[train_idx], X_train_split.iloc[valid_idx]
        y_outer_train, y_outer_valid = y_train_split.iloc[train_idx], y_train_split.iloc[valid_idx]

        # Fit GridSearchCV on the outer training data
        grid_search.fit(X_outer_train, y_outer_train)

        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_
        current_best_params.append(grid_search.best_params_)

        # Predict on the outer validation data
        y_pred_outer = best_model.predict(X_outer_valid)
        y_prob_outer = best_model.predict_proba(X_outer_valid)[:, 1]

        # Calculate metrics
        f1 = f1_score(y_outer_valid, y_pred_outer)
        roc_auc = roc_auc_score(y_outer_valid, y_prob_outer)

        # Store metrics
        nested_f1.append(f1)
        nested_roc_auc.append(roc_auc)

    # Calculate average metrics from nested CV
    avg_f1 = np.mean(nested_f1)
    avg_roc_auc = np.mean(nested_roc_auc)
    f1_scores.append(avg_f1)
    roc_aucs.append(avg_roc_auc)

    # Determine the most common best_params from outer folds
    best_params_counter = {}
    for param in param_grid.keys():
        param_values = [params[param] for params in current_best_params]
        most_common = Counter(param_values).most_common(1)[0][0]
        best_params_counter[param] = most_common

    best_params.append(best_params_counter)
    print(f"Best parameters for {i} hour(s) after: {best_params_counter}")

    # Retrain the pipeline with the most common best parameters on the entire training set
    pipeline.set_params(**best_params_counter)
    pipeline.fit(X_train_split, y_train_split)

    # Save the trained model to a file
    joblib.dump(pipeline.named_steps['classifier'], f'../outputs/svc_hour_{i}.joblib')
    print(f'Model saved for {i} hour(s) after')

    # Predictions on the test data
    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Store the predicted values for the current hour
    predictions[f'Hour_{i}_Predictions'] = y_pred_test

    # Calculate test metrics
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # Store test metrics
    f1_scores.append(f1_test)
    roc_aucs.append(roc_auc_test)

    # Output the test F1 Score and ROC-AUC for the current hour
    print(f"{i} hours after: Test F1 Score: {f1_test:.2f}, Test ROC-AUC: {roc_auc_test:.2f}")

    # ROC Curve for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    plt.plot(fpr_test, tpr_test, label=f'{i} hours after (AUC = {roc_auc_test:.2f})')

# Plot the diagonal line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves SVC')
plt.legend(loc="lower right")

# Save the ROC plot as svc.png
plt.savefig('../outputs/svc.png')
plt.show()

# Output overall results
print(f"Averaged F1 Score: {np.mean(f1_scores):.2f}")
print(f"Averaged ROC-AUC: {np.mean(roc_aucs):.2f}")

# Save all predictions to a CSV file
prediction_df = pd.DataFrame(predictions)
prediction_df.to_csv('../outputs/svc_pred.csv', index=False)
print("Predicted values have been saved to svc_pred.csv")

# Save Best Parameters to svc_params.txt
with open('../outputs/svc_params.txt', 'w') as f:
    for idx, params in enumerate(best_params):
        f.write(f"Hour {idx + 1} Best Parameters: {params}\n")

print("Best parameters for each hour have been saved to svc_params.txt")
print("ROC curves have been saved as svc.png")
