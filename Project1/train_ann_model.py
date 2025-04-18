
# This script trains an ANN model on company spending and location data,
# then saves the model and preprocessing pipeline for later inference.


#pip install pandas scikit-learn

#Run this on terminal:
#python train_ann_model.py


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


num_epochs=1000

print(f"✅ Training model started with {num_epochs} epochs...")

# Paths
data_path = '50_Startups.csv'       # CSV with columns: R&D Spend, Administration, Marketing Spend, State, Profit
model_path = 'tf_profit_model.h5'
preproc_path = 'preprocessor.pkl'

# 1. Load the dataset
df = pd.read_csv(data_path)
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']

# 2. Define preprocessing
numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
])

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Fit and transform
t_X_train = preprocessor.fit_transform(X_train)
t_X_test = preprocessor.transform(X_test)

# 5. Build ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(t_X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
# Use class-based loss and metric to ensure serialization compatibility
model.compile(
    optimizer='adam',
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# 6. Train
model.fit(t_X_train, y_train, epochs=num_epochs, batch_size=8, validation_split=0.1)

# 7. Evaluate
loss, mae = model.evaluate(t_X_test, y_test)
print(f"Test MAE: {mae:.2f}")

# 8. Save
model.save(model_path)

with open(preproc_path, 'wb') as f:
    pickle.dump(preprocessor, f)

print(f"Model saved to {model_path}")
print(f"Preprocessor saved to {preproc_path}")

print("✅ DONE!")
