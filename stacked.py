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
data.dropna(subset=['pm2.5'] + [f'pm2.5_{i}_hour_after' for i in range(1, 13)], inplace=True)

# Calculate the index to split the data at 85% for training and 15% for testing
split_index_train = int(len(data) * 0.5)
split_index_test = int(len(data) * 0.85)

# Split the datasplit_index into training and test sets
data_train = data.iloc[split_index_train:split_index_test]
data_test = data.iloc[split_index_test:]

# Display sizes of the new datasets
print(f"Training Data Size: {data_train.shape[0]}")
print(f"Test Data Size: {data_test.shape[0]}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler # Use StandardScaler for XGB, for KNN and SVC use MinMax instead

# Prepare training and test data (drop the labels and 'year' columns)
X_train = data_train.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])
X_test = data_test.drop(columns=[f'pm2.5_{j}_hour_after' for j in range(1, 13)] + ['year'])

# Define numerical columns
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Set up ColumnTransformer to scale numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)  # Scale numerical columns to [0, 1]
    ],
    remainder='passthrough'  # Keep other columns as is
)