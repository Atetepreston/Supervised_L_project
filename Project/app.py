import streamlit as st
#import joblib
import numpy as np

# Load trained model
#model_filename = "finance_linear_regression_using_supervised_machine_learning.plk"
#lr_model = joblib.load(model_filename)

# Function to load the model
def load_model():
    model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return model


# Streamlit UI
def main():
    st.title("S&P 500 Close Price Prediction")
    st.markdown("Enter the financial indicators below to predict the S&P 500 closing price.")

    # Input fields for user
    sp500_open = st.number_input("S&P 500 Open Price", min_value=0.0, format="%.2f")
    nasdaq_close = st.number_input("Nasdaq Close Price", min_value=0.0, format="%.2f")
    gold_close = st.number_input("Gold Close Price", min_value=0.0, format="%.2f")
    oil_close = st.number_input("Oil Close Price", min_value=0.0, format="%.2f")
    eur_usd = st.number_input("EUR/USD Exchange Rate", min_value=0.0, format="%.4f")

    # Predict button
    if st.button("Predict S&P 500 Close Price"):
        features = np.array([[sp500_open, nasdaq_close, gold_close, oil_close, eur_usd]])
        prediction = lr_model.predict(features)[0]
        st.success(f"Predicted S&P 500 Close Price: {prediction:.2f}")

if _name_ == "_main_":
    main()