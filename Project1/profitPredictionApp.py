import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Paths
DATA_PATH = './50_startups.csv'    # CSV with columns: R&D Spend, Administration, Marketing Spend, State, Profit
MODEL_PATH = 'profit_model.h5'
PREPROCESSOR_PATH = 'preprocessor.pkl'


def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Features and target
    X = df.drop('Profit', axis=1)
    y = df['Profit']

    # Identify numeric and categorical features
    numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    categorical_features = ['State']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Build ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_proc.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Train
    model.fit(X_train_proc, y_train, epochs=100, batch_size=8, validation_split=0.1)

    # Evaluate
    loss, mae = model.evaluate(X_test_proc, y_test)
    print(f"Test MAE: {mae:.2f}")

    # Save model and preprocessor
    model.save(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")


def predict():
    # Load model and preprocessor
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)

    # Gather user input
    print("Enter company data to predict profit:")
    rd_spend = float(input("R&D Spend: "))
    admin = float(input("Administration: "))
    mkt_spend = float(input("Marketing Spend: "))
    state = input("State (e.g., California, New York, Florida): ")

    # Prepare input DataFrame
    df_input = pd.DataFrame([[rd_spend, admin, mkt_spend, state]],
                             columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State'])

    # Preprocess
    X_proc = preprocessor.transform(df_input)

    # Predict
    pred = model.predict(X_proc)
    print(f"Predicted Profit: ${pred[0][0]:.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python profit_prediction_app.py [train|predict]")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == 'train':
        train_model()
    elif cmd == 'predict':
        predict()
    else:
        print("Unknown command. Use 'train' or 'predict'.")
