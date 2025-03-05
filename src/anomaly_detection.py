import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import joblib
import os

def detect_anomalies():
    # Construct paths to processed data files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    X_path = os.path.join(base_dir, 'data', 'processed', 'X.csv')
    y_path = os.path.join(base_dir, 'data', 'processed', 'y.csv')

    # Load processed data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # Train Isolation Forest
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)  # Contamination = expected proportion of anomalies
    iso_forest.fit(X)

    # Save model
    joblib.dump(iso_forest, os.path.join(base_dir, 'models', 'isolation_forest.pkl'))
    print("Isolation Forest training complete.")

    # Train Autoencoder
    print("Training Autoencoder...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define Autoencoder
    input_dim = X_scaled.shape[1]
    encoding_dim = 14  # Size of the encoded representation

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile and train Autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

    # Save model
    autoencoder.save(os.path.join(base_dir, 'models', 'autoencoder.h5'))
    print("Autoencoder training complete.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    detect_anomalies()