"""
Runs XGBoost on dataset with no CV
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

#%%
"""XGBoost Classifier without Cross-Validation"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize lists to store overall results
f1_scores = []
roc_aucs = []
num_hours = 12  # Adjust based on your loop range

# Create a single plot for all ROC curves
plt.figure(figsize=(10, 8))

for i in range(1, num_hours + 1):  # Adjust the range based on your data
    print(f"\nProcessing {i} hour(s) after...")
    
    # Define target variables
    y_train = data_train[f'pm2.5_{i}_hour_after'] >= 50
    y_train = y_train.astype(int)
    y_test = data_test[f'pm2.5_{i}_hour_after'] >= 50
    y_test = y_test.astype(int)
    
    # Create a pipeline with preprocessing and the XGBoost model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Preprocess numerical and categorical columns
        ('classifier', XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=1))
    ])
    
    # Fit the model on the entire training data
    pipeline.fit(X_train, y_train)
    
    # Predictions on the test data
    y_pred_test = pipeline.predict(X_test)
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate test metrics
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    
    # Store test metrics
    f1_scores.append(f1_test)
    roc_aucs.append(roc_auc_test)
    
    # ROC Curve for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    
    # Plot the ROC curve on the single plot
    plt.plot(fpr_test, tpr_test, label=f'{i} hours after (AUC = {roc_auc_test:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], 'k--')

# Configure plot aesthetics
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()
# %%
