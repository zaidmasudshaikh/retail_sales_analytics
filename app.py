import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Machine Learning
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.4rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "Sample - Superstore.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="cp1252")

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)

    df["Profit Margin"] = (
        df["Profit"] / df["Sales"].replace(0, pd.NA) * 100
    )

    df["Profit Status"] = df["Profit"].apply(
        lambda x: "Profit" if x >= 0 else "Loss"
    )

    df["Discount Band"] = pd.cut(
        df["Discount"],
        bins=[-0.001, 0, 0.10, 0.20, 0.30, 0.40, 1.0],
        labels=["0%", "1–10%", "11–20%", "21–30%", "31–40%", "40%+"],
        include_lowest=True,
    )

    return df


df = load_data()

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x:.2f}%"


def filter_data(data):
    st.sidebar.header("🎛️ Filters")

    years = sorted(data["Year"].unique())
    selected_years = st.sidebar.multiselect(
        "Year", years, default=years
    )

    categories = sorted(data["Category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Category", categories, default=categories
    )

    regions = sorted(data["Region"].unique())
    selected_regions = st.sidebar.multiselect(
        "Region", regions, default=regions
    )

    segments = sorted(data["Segment"].unique())
    selected_segments = st.sidebar.multiselect(
        "Customer Segment", segments, default=segments
    )

    filtered = data[
        data["Year"].isin(selected_years)
        & data["Category"].isin(selected_categories)
        & data["Region"].isin(selected_regions)
        & data["Segment"].isin(selected_segments)
    ].copy()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Filtered records: {len(filtered):,} / {len(data):,}"
    )

    return filtered


def kpis(data):
    sales = data["Sales"].sum()
    profit = data["Profit"].sum()
    orders = data["Order ID"].nunique()
    customers = data["Customer ID"].nunique()
    quantity = data["Quantity"].sum()
    margin = profit / sales * 100 if sales else 0
    loss_rows = (data["Profit"] < 0).sum()

    cols = st.columns(7)
    values = [
        ("💰", "Sales", money(sales)),
        ("📈", "Profit", money(profit)),
        ("📦", "Orders", f"{orders:,}"),
        ("👥", "Customers", f"{customers:,}"),
        ("🛒", "Quantity", f"{quantity:,}"),
        ("📊", "Profit Margin", pct(margin)),
        ("🔴", "Loss Records", f"{loss_rows:,}"),
    ]

    for col, (icon, label, value) in zip(cols, values):
        with col:
            st.metric(f"{icon} {label}", value)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("📊 Retail Analytics")
st.sidebar.caption("Superstore Intelligence System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "📈 Sales Analysis",
        "💰 Profitability",
        "👥 Customer Analysis",
        "🌎 Regional Analysis",
        "🤖 Sales Prediction",
        "🧹 Data Quality",
        "ℹ️ Project Information",
    ],
)

data = filter_data(df)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">Retail Sales Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Interactive analytics and profitability analysis using the Superstore dataset</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# EXECUTIVE DASHBOARD
# ---------------------------------------------------------
if page == "🏠 Executive Dashboard":

    kpis(data)
    st.divider()

    st.markdown(
        '<div class="section-title">📌 Business Overview</div>',
        unsafe_allow_html=True,
    )

    monthly = (
        data.groupby("Year-Month", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly,
            x="Year-Month",
            y=["Sales", "Profit"],
            markers=True,
            title="Monthly Sales & Profit Trend",
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            legend_title="Metric",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        category = (
            data.groupby("Category", as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        )

        fig = px.bar(
            category,
            x="Category",
            y=["Sales", "Profit"],
            barmode="group",
            title="Sales & Profit by Category",
        )
        fig.update_layout(yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        region = (
            data.groupby("Region", as_index=False)
            .agg(Profit=("Profit", "sum"))
            .sort_values("Profit")
        )

        fig = px.bar(
            region,
            x="Profit",
            y="Region",
            orientation="h",
            title="Profit by Region",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        segment = (
            data.groupby("Segment", as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        )

        fig = px.bar(
            segment,
            x="Segment",
            y=["Sales", "Profit"],
            barmode="group",
            title="Customer Segment Performance",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Automatic insights
    category_profit = data.groupby("Category")["Profit"].sum()
    region_profit = data.groupby("Region")["Profit"].sum()
    sub_profit = data.groupby("Sub-Category")["Profit"].sum()

    best_category = category_profit.idxmax()
    worst_category = category_profit.idxmin()
    best_region = region_profit.idxmax()
    worst_subcategory = sub_profit.idxmin()

    st.markdown("### 💡 Key Insights")

    i1, i2, i3, i4 = st.columns(4)

    i1.info(f"**Most profitable category**\n\n{best_category}")
    i2.info(f"**Lowest category profit**\n\n{worst_category}")
    i3.success(f"**Top region by profit**\n\n{best_region}")
    i4.warning(f"**Lowest sub-category**\n\n{worst_subcategory}")

# ---------------------------------------------------------
# SALES ANALYSIS
# ---------------------------------------------------------
elif page == "📈 Sales Analysis":

    st.markdown("## 📈 Sales Analysis")
    kpis(data)
    st.divider()

    monthly = (
        data.groupby("Year-Month", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Orders=("Order ID", "nunique"),
            Quantity=("Quantity", "sum"),
        )
    )

    fig = px.line(
        monthly,
        x="Year-Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        category = (
            data.groupby("Category", as_index=False)
            .agg(Sales=("Sales", "sum"))
            .sort_values("Sales", ascending=False)
        )

        fig = px.bar(
            category,
            x="Category",
            y="Sales",
            title="Sales by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        subcategory = (
            data.groupby("Sub-Category", as_index=False)
            .agg(Sales=("Sales", "sum"))
            .sort_values("Sales", ascending=False)
        )

        fig = px.bar(
            subcategory,
            x="Sales",
            y="Sub-Category",
            orientation="h",
            title="Sales by Sub-Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Top 10 Products by Sales")

    products = (
        data.groupby("Product Name", as_index=False)
        .agg(Sales=("Sales", "sum"))
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    fig = px.bar(
        products.sort_values("Sales"),
        x="Sales",
        y="Product Name",
        orientation="h",
        title="Top 10 Products",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PROFITABILITY
# ---------------------------------------------------------
elif page == "💰 Profitability":

    st.markdown("## 💰 Profitability Analysis")
    kpis(data)
    st.divider()

    st.markdown("### 🏷️ Discount Impact")

    discount = (
        data.groupby("Discount Band", observed=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Avg_Profit=("Profit", "mean"),
            Orders=("Order ID", "nunique"),
        )
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            discount,
            x="Discount Band",
            y="Profit",
            title="Total Profit by Discount Band",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            discount,
            x="Discount Band",
            y="Avg_Profit",
            title="Average Profit by Discount Band",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Discount vs Profit")

    fig = px.scatter(
        data,
        x="Discount",
        y="Profit",
        size="Sales",
        color="Category",
        hover_data=["Product Name", "Sub-Category", "Region"],
        title="Individual Orders: Discount vs Profit",
    )
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        sub = (
            data.groupby("Sub-Category", as_index=False)
            .agg(Profit=("Profit", "sum"))
            .sort_values("Profit")
        )
        fig = px.bar(
            sub,
            x="Profit",
            y="Sub-Category",
            orientation="h",
            title="Profit by Sub-Category",
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        loss_products = (
            data.groupby("Product Name", as_index=False)
            .agg(Profit=("Profit", "sum"))
            .sort_values("Profit")
            .head(10)
        )
        fig = px.bar(
            loss_products,
            x="Profit",
            y="Product Name",
            orientation="h",
            title="10 Most Loss-Making Products",
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# CUSTOMER ANALYSIS
# ---------------------------------------------------------
elif page == "👥 Customer Analysis":

    st.markdown("## 👥 Customer Analysis")
    kpis(data)
    st.divider()

    customer = (
        data.groupby(["Customer ID", "Customer Name"], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
            Quantity=("Quantity", "sum"),
        )
    )

    customer["Profit Margin"] = (
        customer["Profit"] / customer["Sales"] * 100
    )

    left, right = st.columns(2)

    with left:
        top_customers = customer.nlargest(10, "Sales").sort_values("Sales")
        fig = px.bar(
            top_customers,
            x="Sales",
            y="Customer Name",
            orientation="h",
            title="Top 10 Customers by Sales",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_profit = customer.nlargest(10, "Profit").sort_values("Profit")
        fig = px.bar(
            top_profit,
            x="Profit",
            y="Customer Name",
            orientation="h",
            title="Top 10 Customers by Profit",
        )
        st.plotly_chart(fig, use_container_width=True)

    segment = (
        data.groupby("Segment", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Customers=("Customer ID", "nunique"),
        )
    )

    fig = px.bar(
        segment,
        x="Segment",
        y=["Sales", "Profit"],
        barmode="group",
        title="Customer Segment Performance",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customer Profitability Table")

    st.dataframe(
        customer.sort_values("Profit", ascending=False).head(25),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# REGIONAL ANALYSIS
# ---------------------------------------------------------
elif page == "🌎 Regional Analysis":

    st.markdown("## 🌎 Regional & Geographic Analysis")
    kpis(data)
    st.divider()

    region = (
        data.groupby("Region", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "nunique"),
        )
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            region,
            x="Region",
            y="Sales",
            title="Sales by Region",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            region,
            x="Region",
            y="Profit",
            title="Profit by Region",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    state = (
        data.groupby("State", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        .sort_values("Profit", ascending=False)
    )

    st.markdown("### State-Level Profitability")

    fig = px.bar(
        state.head(15).sort_values("Profit"),
        x="Profit",
        y="State",
        orientation="h",
        title="Top 15 States by Profit",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        state,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# MACHINE LEARNING - SALES PREDICTION
# ---------------------------------------------------------
elif page == "🤖 Sales Prediction":

    st.markdown("## 🤖 Sales Prediction using Machine Learning")
    st.markdown(
        "Predicting monthly sales using historical Superstore sales data."
    )

    st.divider()

    # -----------------------------------------------------
    # PREPARE MONTHLY DATA
    # -----------------------------------------------------

    monthly_ml = (
        data.groupby("Year-Month", as_index=False)
        .agg(Sales=("Sales", "sum"))
    )

    monthly_ml["Date"] = pd.to_datetime(
        monthly_ml["Year-Month"] + "-01"
    )

    monthly_ml = monthly_ml.sort_values("Date").reset_index(drop=True)

    # Time-based features
    monthly_ml["Year"] = monthly_ml["Date"].dt.year
    monthly_ml["Month"] = monthly_ml["Date"].dt.month
    monthly_ml["TimeIndex"] = np.arange(len(monthly_ml))

    # Cyclical month features
    monthly_ml["Month_Sin"] = np.sin(
        2 * np.pi * monthly_ml["Month"] / 12
    )

    monthly_ml["Month_Cos"] = np.cos(
        2 * np.pi * monthly_ml["Month"] / 12
    )

    feature_columns = [
        "Year",
        "Month",
        "TimeIndex",
        "Month_Sin",
        "Month_Cos",
    ]

    X = monthly_ml[feature_columns]
    y = monthly_ml["Sales"]

    # -----------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------

    if len(monthly_ml) < 12:

        st.warning(
            "Not enough monthly data available for reliable machine learning prediction."
        )

    else:

        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # Time-based split - avoids data leakage
        # -------------------------------------------------

        split_index = int(len(monthly_ml) * 0.80)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # -------------------------------------------------
        # MODELS
        # -------------------------------------------------

        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
        }

        results = []
        predictions = {}

        # -------------------------------------------------
        # TRAIN MODELS
        # -------------------------------------------------

        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, pred)

            rmse = np.sqrt(
                mean_squared_error(y_test, pred)
            )

            r2 = r2_score(y_test, pred)

            # Avoid division by zero
            non_zero = y_test != 0

            mape = np.mean(
                np.abs(
                    (y_test[non_zero] - pred[non_zero])
                    / y_test[non_zero]
                )
            ) * 100

            results.append(
                {
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R² Score": r2,
                    "MAPE": mape,
                }
            )

            predictions[name] = pred

        results_df = pd.DataFrame(results)

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        best_model_name = results_df.loc[
            results_df["RMSE"].idxmin(),
            "Model"
        ]

        best_model = models[best_model_name]

        # -------------------------------------------------
        # KPI CARDS
        # -------------------------------------------------

        best_row = results_df[
            results_df["Model"] == best_model_name
        ].iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🏆 Best Model",
            best_model_name
        )

        c2.metric(
            "MAE",
            f"${best_row['MAE']:,.0f}"
        )

        c3.metric(
            "RMSE",
            f"${best_row['RMSE']:,.0f}"
        )

        c4.metric(
            "R² Score",
            f"{best_row['R² Score']:.3f}"
        )

        st.divider()

        # -------------------------------------------------
        # MODEL COMPARISON
        # -------------------------------------------------

        st.markdown("### 📊 Model Performance Comparison")

        display_results = results_df.copy()

        display_results["MAE"] = display_results["MAE"].map(
            lambda x: f"${x:,.2f}"
        )

        display_results["RMSE"] = display_results["RMSE"].map(
            lambda x: f"${x:,.2f}"
        )

        display_results["R² Score"] = display_results["R² Score"].map(
            lambda x: f"{x:.4f}"
        )

        display_results["MAPE"] = display_results["MAPE"].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # ACTUAL VS PREDICTED
        # -------------------------------------------------

        st.markdown("### 📈 Actual vs Predicted Sales")

        comparison = pd.DataFrame(
            {
                "Date": monthly_ml.iloc[split_index:]["Date"],
                "Actual Sales": y_test.values,
                "Predicted Sales": predictions[best_model_name],
            }
        )

        fig = px.line(
            comparison,
            x="Date",
            y=["Actual Sales", "Predicted Sales"],
            markers=True,
            title=f"Actual vs Predicted Sales — {best_model_name}",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend_title="Metric",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # FUTURE FORECAST
        # -------------------------------------------------

        st.markdown("### 🔮 Next 6 Months Sales Forecast")
        st.caption(
            "Forecast uses the selected best model trained on the complete historical "
            "monthly dataset. It is an academic/exploratory estimate based on calendar "
            "and time-trend features."
        )

        # Train best model on complete historical dataset
        best_model.fit(X, y)

        last_date = monthly_ml["Date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=6,
            freq="MS"
        )

        future = pd.DataFrame(
            {
                "Date": future_dates
            }
        )

        future["Year"] = future["Date"].dt.year
        future["Month"] = future["Date"].dt.month

        future["TimeIndex"] = np.arange(
            len(monthly_ml),
            len(monthly_ml) + len(future)
        )

        future["Month_Sin"] = np.sin(
            2 * np.pi * future["Month"] / 12
        )

        future["Month_Cos"] = np.cos(
            2 * np.pi * future["Month"] / 12
        )

        future["Predicted Sales"] = best_model.predict(
            future[feature_columns]
        )

        # -------------------------------------------------
        # FORECAST CHART
        # -------------------------------------------------

        historical_plot = monthly_ml[
            ["Date", "Sales"]
        ].copy()

        historical_plot["Type"] = "Historical"
        historical_plot.rename(
            columns={"Sales": "Value"},
            inplace=True
        )

        future_plot = future[
            ["Date", "Predicted Sales"]
        ].copy()

        future_plot["Type"] = "Forecast"
        future_plot.rename(
            columns={"Predicted Sales": "Value"},
            inplace=True
        )

        forecast_plot = pd.concat(
            [historical_plot, future_plot],
            ignore_index=True
        )

        fig = px.line(
            forecast_plot,
            x="Date",
            y="Value",
            color="Type",
            markers=True,
            title="Historical Sales & 6-Month Forecast",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -------------------------------------------------
        # FORECAST TABLE
        # -------------------------------------------------

        forecast_table = future[
            ["Date", "Predicted Sales"]
        ].copy()

        forecast_table["Date"] = forecast_table[
            "Date"
        ].dt.strftime("%B %Y")

        forecast_table["Predicted Sales"] = forecast_table[
            "Predicted Sales"
        ].map(
            lambda x: f"${x:,.2f}"
        )

        st.markdown("### 📋 Forecast Details")

        st.dataframe(
            forecast_table,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.markdown("### 💡 Model Interpretation")

        avg_forecast = future["Predicted Sales"].mean()

        historical_avg = monthly_ml["Sales"].mean()

        change = (
            (avg_forecast - historical_avg)
            / historical_avg
            * 100
        )

        if change >= 0:

            st.success(
                f"📈 The model predicts an average monthly sales "
                f"of **${avg_forecast:,.0f}**, approximately "
                f"**{change:.1f}% higher** than the historical "
                f"monthly average."
            )

        else:

            st.warning(
                f"📉 The model predicts an average monthly sales "
                f"of **${avg_forecast:,.0f}**, approximately "
                f"**{abs(change):.1f}% lower** than the historical "
                f"monthly average."
            )

        st.info(
            f"Best performing model based on RMSE: "
            f"**{best_model_name}**"
        )
        
# ---------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------
elif page == "🧹 Data Quality":

    st.markdown("## 🧹 Data Quality & Dataset Overview")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Columns", f"{len(df.columns):,}")
    q3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    q4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.divider()

    st.markdown("### Dataset Information")

    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Unique Values": [df[c].nunique() for c in df.columns],
        }
    )

    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("### Numeric Summary")

    st.dataframe(
        df[["Sales", "Quantity", "Discount", "Profit"]].describe().T,
        use_container_width=True,
    )

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=csv,
        file_name="filtered_superstore.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------
elif page == "ℹ️ Project Information":

    st.markdown("## ℹ️ About the Project")

    st.markdown(
        """
        ### Retail Sales Intelligence & Profitability Analytics

        **Objective:** Analyze retail sales data to identify sales trends,
        profitable products, loss-making products, customer behavior,
        regional performance, and the impact of discounts on profitability.

        ### Technology Stack

        - **Python** — programming language
        - **Pandas** — data loading, cleaning and analysis
        - **NumPy** — numerical feature engineering
        - **Plotly** — interactive visualization
        - **Streamlit** — interactive web application
        - **Scikit-learn** — machine learning and model evaluation

        ### Dataset

        The project uses the Sample Superstore retail dataset containing
        order, customer, product, sales, discount, and profit information.

        ### Machine Learning Module

        The application includes **monthly sales prediction** using three
        regression models:

        - Linear Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor

        Models are evaluated with a chronological 80/20 train-test split
        using:

        - MAE
        - RMSE
        - R² Score
        - MAPE

        The model with the lowest RMSE is selected as the best model and
        retrained on the complete historical dataset for the 6-month forecast.

        ### Forecasting Note

        The forecast is a project-level predictive estimate based on
        historical monthly sales and calendar/time features. It is intended
        for academic demonstration and exploratory business analysis, not as
        a production forecasting system.

        ### Project Status

        **Completed:** data preparation, interactive analytics,
        profitability analysis, customer analysis, regional analysis,
        data-quality checks, machine-learning model comparison, model
        evaluation, actual-vs-predicted visualization, and 6-month sales
        forecasting.
        """
    )

    st.success(
        "Project status: Completed — interactive analytics, profitability analysis, "
        "customer and regional analysis, data-quality checks, and machine-learning "
        "sales prediction are included."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Retail Sales Intelligence • Data Analytics & Visualization Project • "
    "Built with Python, Pandas, Plotly and Streamlit"
)
