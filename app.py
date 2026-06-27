import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📦",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
artifact = joblib.load("best_demand_forecasting_model.pkl")

model = artifact["model"]
product_codes = artifact["product_codes"]
warehouses = artifact["warehouses"]
categories = artifact["categories"]
feature_order = artifact["feature_order"]

# -------------------------------
# Title
# -------------------------------
st.title("📦 Product Demand Forecasting")
st.markdown(
    "Predict future product demand using a **Random Forest Regression** model."
)

st.divider()

# -------------------------------
# Inputs
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    product_code = st.selectbox(
        "Product Code",
        product_codes
    )

    warehouse = st.selectbox(
        "Warehouse",
        warehouses
    )

    category = st.selectbox(
        "Product Category",
        categories
    )

    selected_date = st.date_input(
        "Forecast Date",
        value=date.today()
    )

with col2:

    lag_1 = st.number_input(
        "Previous Day Demand (lag_1)",
        min_value=0.0,
        value=100.0
    )

    lag_7 = st.number_input(
        "Demand 7 Days Ago (lag_7)",
        min_value=0.0,
        value=120.0
    )

    lag_30 = st.number_input(
        "Demand 30 Days Ago (lag_30)",
        min_value=0.0,
        value=140.0
    )

    rolling_mean_7 = st.number_input(
        "7-Day Average Demand",
        min_value=0.0,
        value=110.0
    )

    rolling_std_7 = st.number_input(
        "7-Day Demand Std Dev",
        min_value=0.0,
        value=25.0
    )

# -------------------------------
# Date Features
# -------------------------------

year = selected_date.year
month = selected_date.month
day = selected_date.day
weekday = selected_date.weekday()

quarter = (month - 1) // 3 + 1

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

weekday_sin = np.sin(2 * np.pi * weekday / 7)
weekday_cos = np.cos(2 * np.pi * weekday / 7)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Demand", use_container_width=True):

    input_data = pd.DataFrame([{
        "Product_Code": product_code,
        "Warehouse": warehouse,
        "Product_Category": category,
        "Year": year,
        "Month": month,
        "Weekday": weekday,
        "Quarter": quarter,
        "Day": day,
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Weekday_sin": weekday_sin,
        "Weekday_cos": weekday_cos,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_30": lag_30,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7
    }])

    # Ensure correct feature order
    input_data = input_data[feature_order]

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Forecast Result")

    st.metric(
        "Predicted Demand",
        f"{prediction:.0f} units"
    )

    if prediction < 500:
        st.success("🟢 Low Expected Demand")

    elif prediction < 3000:
        st.warning("🟡 Moderate Expected Demand")

    else:
        st.error("🔴 High Expected Demand")

    st.info(
        f"""
**Forecast Summary**

- Product Code: {product_code}
- Warehouse: {warehouse}
- Category: {category}
- Forecast Date: {selected_date}

The model predicts approximately **{prediction:.0f} units** of demand for the selected product and date.
"""
    )

st.divider()

st.caption(
    "This application uses a Random Forest Regression model trained on historical product demand data. Predictions should be used to support inventory planning and operational decision-making."
)