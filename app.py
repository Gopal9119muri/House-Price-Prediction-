from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
import xgboost
import locale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define file path for accuracy file
ACCURACY_FILE_PATH = os.path.join(os.path.dirname(__file__), "accuracy.txt")

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

except FileNotFoundError:
    print("Error: Model or Scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are present.")
    model = None
    scaler = None

def preprocess_input(bedrooms, bathrooms, yr_built, grade, floors, sqft_living, sqft_lot, yr_renovated, scaler):
    """Preprocess the input data for prediction."""
    input_df = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'yr_built': [yr_built],
        'grade': [grade],
        'floors': [floors],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'yr_renovated': [yr_renovated]
    })
        # Feature Engineering: Create 'age' from 'yr_built'
    input_df['age'] = 2024 - input_df['yr_built']
    input_df = input_df.drop('yr_built', axis=1)

     # Add Polynomial features:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    input_poly = poly.fit_transform(input_df)

    # Scale features
    scaled_input = scaler.transform(input_poly)

    return scaled_input

@app.route('/')
def home():
    # Read the accuracy score from file
    try:
        with open(ACCURACY_FILE_PATH, 'r') as f:
            accuracy = float(f.readline().strip())
            accuracy = f"{accuracy * 100:.2f}"
    except FileNotFoundError:
        accuracy = "Accuracy not available"
    return render_template('home.html', accuracy=accuracy)


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    errors = {}
    if request.method == 'POST':
        # Get data from the form
        try:
             location = request.form['location']
             bedrooms = request.form['bedrooms']

             bathrooms = request.form['bathrooms']
             yr_built = request.form['yr_built']
             grade = request.form['grade']
             floors = request.form['floors']
             sqft_living = request.form['sqft_living']

             sqft_lot = request.form['sqft_lot']

             yr_renovated = request.form['yr_renovated']
             if float(bedrooms) < 0:
                  errors['bedrooms'] = "Bedrooms cannot be negative."

             if float(bathrooms) < 0:
                 errors['bathrooms'] = "Bathrooms cannot be negative."

             if float(grade) < 0 :
                errors['grade'] = "Grade cannot be negative"
             if float(floors) < 0 :
                 errors['floors'] = "Floors cannot be negative."

             if float(sqft_living) < 0:
                  errors['sqft_living'] = "Living space cannot be negative."
             if float(sqft_lot) < 0 :
                 errors['sqft_lot'] = "Lot Size cannot be negative"
             if float(yr_renovated) < 0:
                  errors['yr_renovated'] = "Year Renovated cannot be negative."
        except (ValueError, KeyError):
            errors['general'] = "Invalid input. Please enter valid numbers."

        if errors:
             return render_template('index.html', errors=errors,
                location=request.form.get('location'),
                 bedrooms=request.form.get('bedrooms'),
                 bathrooms=request.form.get('bathrooms'),
                 yr_built=request.form.get('yr_built'),
                 grade=request.form.get('grade'),
                floors=request.form.get('floors'),
                  sqft_living=request.form.get('sqft_living'),
                sqft_lot=request.form.get('sqft_lot'),
                yr_renovated=request.form.get('yr_renovated')
            )

        if model is None or scaler is None:
              errors['model_error'] = "Model not loaded. Please check the console for errors"
              return render_template('index.html', errors=errors,
                   location=location,
                  bedrooms=bedrooms,
                 bathrooms=bathrooms,
                 yr_built=yr_built,
                 grade=grade,
                floors=floors,
                  sqft_living=sqft_living,
                sqft_lot=sqft_lot,
                yr_renovated=yr_renovated
            )

        # Preprocess the input
        processed_input = preprocess_input(float(bedrooms), float(bathrooms), float(yr_built), float(grade), float(floors), float(sqft_living), float(sqft_lot), float(yr_renovated), scaler)

        # Make prediction
        price = model.predict(processed_input)[0]

         # Adjust price based on location
        location_adjustments = {
           "Dandeli": 1034760,
            "Haliyal": 1564034,
            "Dharwad": 2065478,
            "Hubli": 2509897,
            "Belgaum": 3698676
           }

        if location in location_adjustments:
              price = price + location_adjustments[location]

        # Return raw price (no formatting)
        formatted_price = f"₹ {int(round(price))}"

        return render_template('index.html', data=formatted_price,
            location=location,
            bedrooms=bedrooms,
                 bathrooms=bathrooms,
                 yr_built=yr_built,
                 grade=grade,
                floors=floors,
                  sqft_living=sqft_living,
                sqft_lot=sqft_lot,
                yr_renovated=yr_renovated)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)