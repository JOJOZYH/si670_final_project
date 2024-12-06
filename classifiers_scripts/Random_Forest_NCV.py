"""
Runs RF with nested CV
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

#%% Random Forest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

# Initialize lists to store results for each hour
f1_scores = []
roc_aucs = []

# Create a single plot for all ROC curves
plt.figure(figsize=(10, 8))

# Use tqdm to add a progress bar to the loop
for i in range(1, 13):
    # The current ith hour after current hour as label
    y_train = data_train[f'pm2.5_{i}_hour_after'] >= 50
    y_test = data_test[f'pm2.5_{i}_hour_after'] >= 50

    # Create a pipeline with preprocessing and the Random Forest model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Normalize numerical columns
        ('classifier', RandomForestClassifier(random_state=1))  # Random Forest model
    ])

    # Define parameter grid for Random Forest hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Wrap the pipeline in GridSearchCV
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    # Train the pipeline with grid search
    grid.fit(X_train, y_train)

    # Get the best model after grid search
    model = grid.best_estimator_

    # Store best estimator to lst
    best_models[i - 1]['RF'] = model

    # Predictions on the test data
    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Calculate scores for the test data
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # Store the results
    f1_scores.append(f1_test)
    roc_aucs.append(roc_auc_test)

    # Output the test F1 Score and ROC-AUC for the current hour
    print(f"{i} hours after: Test F1 Score: {f1_test:.2f}, Test ROC-AUC: {roc_auc_test:.2f}")

    # ROC Curve for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

    # Plot the ROC curve on the single plot
    plt.plot(fpr_test, tpr_test, label=f'{i} hours after (area = {roc_auc_test:.2f})')

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
print(f"Averaged F1 Score: {sum(f1_scores)/len(f1_scores):.2f}")
print(f"Averaged ROC-AUC: {sum(roc_aucs)/len(roc_aucs):.2f}")
