import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import os
import xgboost
from sklearn.preprocessing import PolynomialFeatures


# Define a file path
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "house_data.csv")
ACCURACY_FILE_PATH = os.path.join(os.path.dirname(__file__), "accuracy.txt")

try:
    # Load the CSV file into a pandas DataFrame
    house_df = pd.read_csv(CSV_FILE_PATH)

    # Select the columns that we need
    feature_columns = ['bedrooms', 'bathrooms', 'yr_built', 'grade', 'floors', 'sqft_living', 'sqft_lot', 'yr_renovated', 'price']
    house_df = house_df[feature_columns]

    # Handle missing values by dropping them
    house_df = house_df.dropna()

    # Separate features (X) and target (y)
    X = house_df[['bedrooms', 'bathrooms', 'yr_built', 'grade', 'floors', 'sqft_living', 'sqft_lot', 'yr_renovated']]
    y = house_df['price']
    # Feature Engineering: Create 'age' from 'yr_built'
    X['age'] = 2024 - X['yr_built']
    X = X.drop('yr_built', axis=1) # Drop the yr_built after we have created age

    # Add Polynomial features:
    poly = PolynomialFeatures(degree=2, include_bias=False) # Generate polynomial features with degree = 2
    X = poly.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Create a XGBoost regressor model
    lr = xgboost.XGBRegressor(random_state = 42)

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Save the scaler
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # Save the trained model to a file
    pickle.dump(lr, open('model.pkl', 'wb'))

    # Save accuracy
    with open(ACCURACY_FILE_PATH, 'w') as f:
      f.write(f"{r2:.2f}")
    with open(ACCURACY_FILE_PATH, 'a') as f:
        f.write(f"\n{y_pred[0]}")
    print("Model trained and saved successfully.")

except FileNotFoundError:
    print(f"Error: The file at {CSV_FILE_PATH} was not found. Make sure the CSV file path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")