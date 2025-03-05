import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import randint
import joblib
import os

def train_models():
    # Construct paths to processed data files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    X_path = os.path.join(base_dir, 'data', 'processed', 'X.csv')
    y_path = os.path.join(base_dir, 'data', 'processed', 'y.csv')

    # Load processed data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Subsample the data for faster debugging (optional)
    X_train = X_train.sample(frac=0.1, random_state=42)
    y_train = y_train.loc[X_train.index]

    # Train Random Forest with Grid Search
    print("Training Random Forest with Grid Search...")
    rf = RandomForestClassifier(random_state=42)

    # Define parameter grid for Grid Search
    param_grid_rf = {
        'n_estimators': [50, 100],  # Fewer values
        'max_depth': [None, 10],     # Fewer values
        'min_samples_split': [2, 5]  # Fewer values
    }

    # Perform Grid Search
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=2, scoring='f1', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    # Best parameters for Random Forest
    print("Best Parameters for Random Forest:", grid_search_rf.best_params_)

    # Save the best Random Forest model
    best_rf = grid_search_rf.best_estimator_
    joblib.dump(best_rf, os.path.join(base_dir, 'models', 'random_forest_tuned.pkl'))
    print("Random Forest training complete.")

    # Train XGBoost with Random Search
    print("Training XGBoost with Random Search...")
    xgb_model = xgb.XGBClassifier(random_state=42)

    # Define parameter distribution for Random Search
    param_dist_xgb = {
        'n_estimators': randint(50, 100),  # Smaller range
        'max_depth': randint(3, 5),        # Smaller range
        'learning_rate': [0.1, 0.2]        # Fewer values
    }

    # Perform Random Search
    random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist_xgb, n_iter=5, cv=2, scoring='f1', n_jobs=-1)
    random_search_xgb.fit(X_train, y_train)

    # Best parameters for XGBoost
    print("Best Parameters for XGBoost:", random_search_xgb.best_params_)

    # Save the best XGBoost model
    best_xgb = random_search_xgb.best_estimator_
    joblib.dump(best_xgb, os.path.join(base_dir, 'models', 'xgboost_tuned.pkl'))
    print("XGBoost training complete.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)

    # Train models
    train_models()