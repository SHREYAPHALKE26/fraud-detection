# Fraud Detection in Financial Transactions

## Project Overview
This project aims to detect fraudulent transactions using machine learning techniques. It includes:
- Data preprocessing (SMOTE, normalization).
- Model training (Random Forest, XGBoost).
- Model evaluation (precision, recall, F1-score, ROC AUC).
- Anomaly detection (Isolation Forest, Autoencoders).
- Model deployment (Flask API).

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Folder Structure
```bash
fraud-detection/
├── data/
│ ├── raw/ # Raw dataset (creditcard.csv)
│ └── processed/ # Processed data (X.csv, y.csv)
│
├── models/ # Trained models
│ ├── random_forest_tuned.pkl # Best-tuned Random Forest model
│ ├── xgboost_tuned.pkl # Best-tuned XGBoost model
│ ├── isolation_forest.pkl # Trained Isolation Forest model
│ └── autoencoder.h5 # Trained Autoencoder model
│
├── results/ # Evaluation results
│ ├── confusion_matrix_rf.png # Confusion matrix for Random Forest
│ ├── confusion_matrix_xgb.png # Confusion matrix for XGBoost
│ ├── roc_curve_rf.png # ROC curve for Random Forest
│ ├── roc_curve_xgb.png # ROC curve for XGBoost
│ └── metrics.txt # Classification reports and ROC AUC scores
│
├── src/ # Source code
│ ├── preprocessing.py # Data preprocessing script
│ ├── train.py # Model training script
│ ├── evaluate.py # Model evaluation script
│ ├── anomaly_detection.py # Anomaly detection script
│ └── utils.py # Utility functions (if any)
│
├── app.py # Flask API for model deployment
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SHREYAPHALKE26/fraud-detection.git
   cd fraud-detection
   ```
2. Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Usage
1. Data Preprocessing
Run the preprocessing script to normalize the data and handle class imbalance:
```bash
python src/preprocessing.py
```
2. Model Training
Train the Random Forest and XGBoost models with hyperparameter tuning:
```bash
python src/train.py
```
3. Model Evaluation
Evaluate the trained models and generate performance metrics:
```bash
python src/evaluate.py
```
4. Anomaly Detection
Train Isolation Forest and Autoencoder models for anomaly detection:
```bash
python src/anomaly_detection.py
```
5. Model Deployment
Deploy the best-performing model as a Flask API:
```bash
python app.py
```
6. Test the API
Use Postman or curl to send a POST request to the API:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "features": [
    1.0, -1.2, 0.5, -0.8, 1.5, -0.3, 0.7, -1.0, 0.2, -0.5,
    1.1, -0.9, 0.4, -0.6, 1.3, -0.7, 0.8, -1.1, 0.3, -0.4,
    1.2, -0.2, 0.6, -0.1, 1.4, -0.8, 0.9, -1.3, 0.0, 1.0
  ]
}' http://127.0.0.1:5000/predict
```
## Example Response
json
```bash
{
  "prediction": 0
}
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Dataset: Credit Card Fraud Detection Dataset
- Libraries: Scikit-learn, XGBoost, TensorFlow, Flask

