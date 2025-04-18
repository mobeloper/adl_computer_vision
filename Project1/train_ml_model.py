
# Train a scikit-learn regression model on 50_Startups.csv
# to predict company Profit based on spending and location.


#pip install pandas scikit-learn

#Run this on terminal:
#python train_ml_model.py



import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


num_estimators=100

print(f"✅ Training model started with {num_estimators} estimators...")

# Paths
data_path = '50_Startups.csv'       # CSV with columns: R&D Spend, Administration, Marketing Spend, State, Profit
model_path = 'sk_profit_model.pkl'
preproc_path = 'preprocessor.pkl'

# 1. Load the dataset
df = pd.read_csv(data_path)
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = df['Profit']


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 3. Define and fit the preprocessor
numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']



# numeric_transformer = Pipeline([
#     ('scaler', StandardScaler())
# ])
numeric_transformer = StandardScaler()

# categorical_transformer = Pipeline([
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)


preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Setup model
# model = Pipeline([
#     ('preproc', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
# ])
model = RandomForestRegressor(n_estimators=num_estimators, random_state=42)
# model = LinearRegression()

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 5. Fit the model
# model.fit(X_train, y_train)
model.fit(X_train_processed, y_train)


# 6. Predict and evaluate
# y_pred = model.predict(X_test)
y_pred = model.predict(X_test_processed)


print(f"Train score: {model.score(X_train_processed,y_train)}")
print(f"Test score: {model.score(X_test_processed,y_test)}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 7. Save the trained pipeline >> Save preprocessor and model separately
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    
with open(preproc_path, 'wb') as f:
    pickle.dump(preprocessor, f)

# print(f" Model pipeline saved to {model_path}")
print(f"Model saved to {model_path}")
print(f"Preprocessor saved to {preproc_path}")

print("✅ DONE!")



# 8. (Optional) CLI prediction example

def predict_profit(rd_spend, admin, mkt_spend, state):
    example_df = pd.DataFrame(
        [[rd_spend, admin, mkt_spend, state]],
        columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    )
    # Load preprocessor and model if not in memory
    with open('preprocessor.pkl', 'rb') as f:
        prep = pickle.load(f)
    with open('sk_profit_model.pkl', 'rb') as f:
        mod = pickle.load(f)
    proc = prep.transform(example_df)
    return mod.predict(proc)[0]

if __name__ == '__main__':
    # Example usage
    sample = predict_profit(160000, 130000, 300000, 'California')
    print(f"Example predicted profit: ${sample:,.2f}")
