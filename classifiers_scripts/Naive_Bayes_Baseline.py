"""
Naive Bayes Classifier Evaluation as a Baseline
with Hourly F1 Scores and ROC Curves
"""

#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

#%% Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path, index_col=0)

# Interpolate missing values in PM2.5
data['pm2.5'] = data['pm2.5'].interpolate()

# Generate future PM2.5 levels (1 to 12 hours ahead)
for i in range(1, 13):
    data[f'pm2.5_{i}_hour_after'] = data['pm2.5'].shift(-i)

# One-hot encode categorical column 'cbwd'
data = pd.get_dummies(data, columns=['cbwd'])

# Remove rows with missing values in future PM2.5 predictions
data.dropna(
    subset=['pm2.5'] + [f'pm2.5_{i}_hour_after' for i in range(1, 13)],
    inplace=True
)

# Define splits:
#  - First 50% for training
#  - Last 15% for testing
#  (The middle 35% was used previously for an intermediate step, but here we focus on final testing.)
split_index_train = int(len(data) * 0.5)
split_index_test = int(len(data) * 0.85)

data_train = data.iloc[:split_index_train]
data_test = data.iloc[split_index_test:]  # Using the last 15% for testing as in your reference code

print(f"Training Data Size: {data_train.shape[0]}")
print(f"Test Data Size: {data_test.shape[0]}")

#%% Prepare training and test data
future_targets = [f'pm2.5_{j}_hour_after' for j in range(1, 13)]
X_train = data_train.drop(columns=future_targets + ['year'])
X_test = data_test.drop(columns=future_targets + ['year'])

# Define numerical columns
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Reconstruct X_train and X_test with scaled numerical columns
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index)

X_train_final = X_train.copy()
X_test_final = X_test.copy()

X_train_final[numerical_columns] = X_train_scaled_df[numerical_columns]
X_test_final[numerical_columns] = X_test_scaled_df[numerical_columns]

feature_names = X_test_final.columns.tolist()

#%% Initialize lists to store evaluation metrics
f1_scores = []
roc_aucs = []
predictions = {}
num_hours = 12  # Total hours to evaluate

plt.figure(figsize=(10, 8))  # For combined ROC plot

#%% Train and evaluate Naive Bayes model for each hour
for i in range(1, num_hours + 1):
    print(f"\nEvaluating Naive Bayes model for {i} hour(s) after...")

    # Define target variable for the i-th hour (binary: pm2.5 >= 50)
    y_train = (data_train[f'pm2.5_{i}_hour_after'] >= 50).astype(int)
    y_test = (data_test[f'pm2.5_{i}_hour_after'] >= 50).astype(int)

    # Initialize and train Naive Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train_final, y_train)

    # Make predictions on the test data
    y_pred_test = nb_classifier.predict(X_test_final)
    # predict_proba gives probabilities for each class, we take the positive class (index 1)
    y_prob_test = nb_classifier.predict_proba(X_test_final)[:, 1]

    # Store predictions
    predictions[f'Hour_{i}_Predictions'] = y_pred_test

    # Calculate metrics
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # Store evaluation metrics
    f1_scores.append(f1_test)
    roc_aucs.append(roc_auc_test)

    # Output the metrics for the current hour
    print(f"{i} hours after: Test F1 Score: {f1_test:.2f}, Test ROC-AUC: {roc_auc_test:.2f}")

    # Generate ROC curve for the current hour
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    plt.plot(fpr_test, tpr_test, label=f'{i} hours after (AUC = {roc_auc_test:.2f})')

#%% Plot a diagonal reference line
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves - Naive Bayes Baseline')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('naive_bayes_baseline_roc.png')
plt.show()

#%% Output averaged metrics
if f1_scores:
    print(f"Averaged F1 Score: {np.mean(f1_scores):.2f}")
else:
    print("No F1 scores to average.")

if roc_aucs:
    print(f"Averaged ROC-AUC: {np.mean(roc_aucs):.2f}")
else:
    print("No ROC-AUC scores to average.")

#%% Save predictions to CSV
if predictions:
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv('naive_bayes_predictions.csv', index=False)
    print("Predicted values have been saved to naive_bayes_predictions.csv")
else:
    print("No predictions to save.")