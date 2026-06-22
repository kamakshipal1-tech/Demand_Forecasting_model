# 📈 Demand Forecasting Using Machine Learning

## Business Problem

Accurate demand forecasting is a critical component of inventory management and supply chain planning. Businesses that underestimate demand frequently experience stockouts, lost sales, and reduced customer satisfaction. On the other hand, overestimating demand results in excess inventory, increased warehousing costs, and inefficient capital allocation.

The objective of this project is to develop a machine learning-driven demand forecasting system capable of predicting future product demand using historical transaction records. The resulting forecasts can support inventory planning, procurement decisions, warehouse allocation, and overall supply chain optimization.

---

## Dataset Overview

The dataset contains historical product demand records across multiple warehouses and product categories.

### Available Features

* Product_Code
* Warehouse
* Product_Category
* Date
* Order_Demand

The target variable is **Order_Demand**, making this a supervised regression problem where the goal is to predict a continuous numerical value rather than classify observations into categories.

---

## Data Quality Assessment

Before model development, a detailed inspection of the dataset was performed.

### Missing Values

Missing values were identified and handled appropriately because machine learning models cannot learn meaningful patterns from incomplete observations.

### Duplicate Records

More than 12,000 duplicate records were detected in the dataset. These duplicates were removed to prevent biased learning and inflated demand estimates.

### Date Inconsistencies

The Date column required conversion into datetime format to enable time-based analysis and feature extraction.

### Demand Value Cleaning

Order demand values contained formatting inconsistencies that required preprocessing and conversion into numerical form before model training.

---

## Exploratory Data Analysis

Several exploratory analyses were performed to understand customer purchasing behaviour and overall demand patterns.

### Demand Distribution Analysis

The distribution of demand was highly right-skewed. Most orders contained relatively small demand values, while a small number of observations represented extremely large orders.

This finding is important because a few large transactions can significantly influence inventory decisions and forecasting performance.

### Product Category Analysis

Certain product categories consistently contributed a disproportionate share of total demand.

#### Business Insight

Organizations can prioritize inventory allocation and procurement planning for high-demand categories to reduce stockout risk and improve service levels.

### Warehouse-Level Demand Analysis

Demand was not uniformly distributed across warehouses.

#### Business Insight

Regional demand variation suggests that inventory should be distributed strategically rather than equally across all warehouse locations.

### Monthly Demand Trend Analysis

Monthly demand patterns revealed fluctuations over time, indicating the presence of temporal demand behaviour.

#### Business Insight

Historical demand trends can be leveraged to improve forecasting accuracy and support seasonal inventory planning.

---

## Feature Engineering

To capture temporal demand behaviour, several date-based features were generated:

* Year
* Month
* Quarter
* Day
* Day of Week

These features enable machine learning models to learn seasonal and calendar-based demand patterns.

Categorical variables were encoded using Label Encoding before model training.

---

## Model Development

Multiple regression algorithms were trained and evaluated.

### Linear Regression

A baseline model used to establish a performance benchmark.

### Decision Tree Regressor

Captures non-linear relationships between product characteristics and demand.

### Random Forest Regressor

Reduces overfitting by combining predictions from multiple decision trees.

### XGBoost Regressor

Gradient boosting model capable of capturing complex interactions and non-linear demand behaviour.

---

## Model Evaluation

Since demand forecasting is a regression problem, classification metrics such as Accuracy, Precision, Recall, and F1 Score are not appropriate.

The models were evaluated using:

### RMSE (Root Mean Squared Error)

Measures the magnitude of forecasting errors while penalizing larger mistakes more heavily.

### MAE (Mean Absolute Error)

Measures average prediction error in demand units.

### R² Score

Measures how effectively the model explains variation in demand.

The final forecasting model was selected based on:

* Lowest RMSE
* Lowest MAE
* Highest R² Score

This ensured that the chosen model produced the most reliable demand estimates while minimizing forecasting error.

XGBoost achieved the highest R² score and lowest forecasting error, making it the final production model.

---

## Business Impact

The forecasting framework can provide significant operational benefits:

### Inventory Optimization

Reduce excess stock while maintaining product availability.

### Supply Chain Efficiency

Improve procurement and replenishment planning.

### Warehouse Management

Enable demand-driven inventory allocation across warehouses.

### Reduced Operational Cost

Minimize storage costs associated with overstocking.

### Improved Customer Satisfaction

Decrease stockout events by anticipating future demand more accurately.

---

## Key Takeaways

* Historical demand is a strong predictor of future demand.
* Demand patterns vary significantly across products and warehouses.
* Data cleaning and preprocessing have a major impact on forecasting quality.
* Machine learning models can provide actionable forecasts that support inventory and supply chain decisions.
* Forecasting should be evaluated using regression metrics such as RMSE, MAE, and R² rather than classification metrics.

## Tech Stack

Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-Learn • XGBoost • Jupyter Notebook
