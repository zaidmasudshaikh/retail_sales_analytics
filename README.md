# Retail Sales Intelligence & Profitability Analytics

An interactive Data Analytics and Visualization project built using Python, Pandas, Plotly, and Streamlit to analyze retail sales, profitability, customer behavior, regional performance, and monthly sales forecasts.

🚀 Live Demo: https://retailsalesintelligence.streamlit.app/

## Project Objective

Analyze historical retail transactions to understand sales, profitability, customers, regions and discount impact, then demonstrate monthly sales prediction using machine learning.

## Features

- Executive dashboard with business KPIs
- Interactive filters for year, category, region and customer segment
- Monthly sales and profit trends
- Category and sub-category sales analysis
- Top products by sales
- Profitability and discount-band analysis
- Discount vs. profit analysis
- Customer sales and profitability analysis
- Regional and state-level analysis
- Data-quality checks and downloadable filtered data
- Monthly sales prediction using three ML regression models
- Model comparison using MAE, RMSE, R² and MAPE
- Actual vs. predicted sales visualization
- 6-month sales forecast
- Forecast details and model interpretation

## Machine Learning Methodology

The sales-prediction module aggregates transactions into monthly sales and creates calendar/time features: year, month, time index, and cyclical month sine/cosine features.

Three regression models are compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

A chronological **80/20 train-test split** is used so future observations are not used to train the model before evaluation. The model with the lowest RMSE is selected as the best model and retrained on all available historical months for the 6-month forecast.

> **Important:** The forecast is an academic/exploratory estimate, not a production-grade time-series forecasting system.

## Dataset

The project uses the **Sample Superstore** dataset containing order, customer, product, sales, discount and profit information. The supplied dataset contains 9,994 transaction rows covering 2014–2017.

## Technology Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn

## Project Structure

```text
retail_sales_analytics/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── Sample - Superstore.csv
```

## Run on macOS

Open Terminal in the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser should open the Streamlit application automatically.

## Project Status

**Completed.** The dashboard, analytics modules, data-quality checks, machine-learning comparison, evaluation metrics, actual-vs-predicted analysis and 6-month forecast are implemented.

## Academic Note

This project demonstrates descriptive analytics, diagnostic business analysis, interactive visualization and introductory predictive analytics in a single Streamlit application.
