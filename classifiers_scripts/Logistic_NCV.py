"""
Logistic
"""
#%% import data
import pandas as pd

file_path = 'data/data.csv'
data = pd.read_csv(file_path, index_col=0)

#%% preprocessing
data['pm2.5'] = data['pm2.5'].interpolate()

# Creating 12 new columns for future PM2.5 levels, 1 hour to 12 hours ahead
for i in range(1, 13):
    data[f'pm2.5_{i}_hour_after'] = data['pm2.5'].shift(-i)

# One-hot encode the 'cbwd' column
data = pd.get_dummies(data, columns=['cbwd'])

# Calculate the total number of rows with any missing values before dropping
missing_rows_before = data.isna().any(axis=1).sum()
print(f'missing rows before: {missing_rows_before}')
# Drop rows where any cell from 'pm2.5' to 'pm2.5_12_hour_after' is missing
data.dropna(subset=['pm2.5'] + [f'pm2.5_{i}_hour_after' for i in range(1, 13)], inplace=True)

# Calculate the index to split the data at 85% for training and 15% for testing
split_index = int(len(data) * 0.85)

# Split the data into training and test sets
data_train = data.iloc[:split_index]
data_test = data.iloc[split_index:]

# Displaying sizes of the new datasets
print(f"Training Data Size: {data_train.shape[0]}")
print(f"Test Data Size: {data_test.shape[0]}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Prepare training and test data (drop the labels and year columns)
X_train = data_train.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])
X_test = data_test.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])

# Define numerical columns (replace these with the actual numerical column names in your dataset)
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Set up columnTransformer to normalize numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)  # Normalize numerical columns
    ],
    remainder='passthrough'  # Keep other columns as is
)
# %% Logistic Regression with Nested Cross-Validation

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store overall results
f1_scores = []
roc_aucs = []
num_hours = 12  # Adjust based on your loop range

# Define parameter grid for hyperparameter tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']  # 'liblinear' supports both 'l1' and 'l2'
}

# Create a single plot for all ROC curves
plt.figure(figsize=(10, 8))

for i in range(1, num_hours + 1):
    print(f"\nProcessing {i} hour(s) after...")

    # Define target variables
    y_train = data_train[f'pm2.5_{i}_hour_after'] >= 50
    y_test = data_test[f'pm2.5_{i}_hour_after'] >= 50

    # Define the outer and inner cross-validation strategies
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    # Create a pipeline with preprocessing and the Logistic Regression model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Preprocess numerical columns
        ('classifier', LogisticRegression(max_iter=1000, random_state=1))
    ])

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    # Initialize metrics lists for nested CV
    nested_f1 = []
    nested_roc_auc = []

    # Outer cross-validation loop
    for train_idx, valid_idx in outer_cv.split(X_train, y_train):
        # Split data into outer training and validation sets
        X_outer_train, X_outer_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_outer_train, y_outer_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Fit GridSearchCV on the outer training data
        grid_search.fit(X_outer_train, y_outer_train)

        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_

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

    # Retrain the best model on the entire training set
    best_model.fit(X_train, y_train)

    # Predictions on the test data
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]

    # Calculate test metrics
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # Output the test F1 Score and ROC-AUC for the current hour
    print(f"{i} hours after: Test F1 Score: {f1_test:.2f}, Test ROC-AUC: {roc_auc_test:.2f}")

    # ROC Curve for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

    # Plot the ROC curve on the single plot
    plt.plot(fpr_test, tpr_test, label=f'{i} hours after (AUC = {roc_auc_test:.2f})')

# Plot the diagonal line
plt.plot([0, 1], [0, 1], 'k--')

# Configure plot
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.show()

# Output overall results
print(f"Averaged F1 Score: {np.mean(f1_scores):.2f}")
print(f"Averaged ROC-AUC: {np.mean(roc_aucs):.2f}")