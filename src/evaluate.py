import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models():
    # Construct paths to processed data files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    X_path = os.path.join(base_dir, 'data', 'processed', 'X.csv')
    y_path = os.path.join(base_dir, 'data', 'processed', 'y.csv')

    # Load test data
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path)

    # Load models
    rf = joblib.load(os.path.join(base_dir, 'models', 'random_forest_tuned.pkl'))
    xgb_model = joblib.load(os.path.join(base_dir, 'models', 'xgboost_tuned.pkl'))

    # Evaluate Random Forest
    print("Evaluating Random Forest...")
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    print("Random Forest ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))

    # Confusion Matrix for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig(os.path.join(base_dir, 'results', 'confusion_matrix_rf.png'))
    plt.show()

    # ROC Curve for Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba_rf)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'results', 'roc_curve_rf.png'))
    plt.show()

    # Evaluate XGBoost
    print("Evaluating XGBoost...")
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb))

    print("XGBoost ROC AUC Score:", roc_auc_score(y_test, y_pred_proba_xgb))

    # Confusion Matrix for XGBoost
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost Confusion Matrix')
    plt.savefig(os.path.join(base_dir, 'results', 'confusion_matrix_xgb.png'))
    plt.show()

    # ROC Curve for XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba_xgb)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'results', 'roc_curve_xgb.png'))
    plt.show()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
    evaluate_models()