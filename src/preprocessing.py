import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def load_data():
    # Construct the path to the raw data file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'creditcard.csv')
    data = pd.read_csv(raw_data_path)
    return data

def preprocess_data(data):
    # Normalize 'Amount' and 'Time'
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Apply SMOTE to handle imbalanced data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res

if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    os.makedirs(os.path.join(base_dir, 'data', 'processed'), exist_ok=True)

    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Save processed data
    X.to_csv(os.path.join(base_dir, 'data', 'processed', 'X.csv'), index=False)
    y.to_csv(os.path.join(base_dir, 'data', 'processed', 'y.csv'), index=False)