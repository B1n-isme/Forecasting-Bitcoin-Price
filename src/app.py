from datetime import datetime

from scipy.stats import norm
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.arima_garch_pred import arima_garch_eval, arima_garch_forecast
from utils.data_preparation import load_data, scale_data, create_lstm_dataset, create_tft_dataset
from utils.dataset import fetch_new_data  # noqa: F401
from utils.lstm_model import *  # noqa: F403
from utils.lstm_pred import LSTM_eval_new, LSTM_forecast
from utils.pca_preprocessing import apply_log_transform, transform_dataset
from utils.tft_model import *  # noqa: F403
from utils.tft_pred import TFT_eval_new, TFT_forecast

st.set_page_config(page_title="Bitcoin Prediction", layout="wide")  # Enables wide mode

# Paths
model_dir = "../models"
results_dir = "../results"
metrics_dir = "../results/metrics"
data_dir = "../data/final"

look_back = 7
model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]
model_options = [
    "LSTM",
    "BiLSTM",
    "Attention-LSTM",
    "Attention-BiLSTM",
    "Ensemble-LSTM",
    "Temporal-Fusion-Transformer",
]

# Load data
if "dataset" not in st.session_state:
    st.session_state.dataset = load_data(f"{data_dir}/dataset.csv")
    st.session_state.val_pca_df = load_data(f"{data_dir}/val_pca_df.csv")
    st.session_state.test_pca_df = load_data(f"{data_dir}/test_pca_df.csv")
    st.session_state.train_residuals_df = load_data(f"{data_dir}/train_residuals_df.csv")
    st.session_state.test_residuals_df = load_data(f"{data_dir}/test_residuals_df.csv")

    st.session_state.scaler = joblib.load("../models/scaler.pkl")
    st.session_state.pca = joblib.load("../models/pca.pkl")
    st.session_state.residual_scaler = joblib.load("../models/residual_scaler.pkl")

    st.session_state.test_lstm_predictions = load_data(
        "../results/predictions/test/lstm_predictions.csv"
    )
    st.session_state.test_lstm_uncertainties = load_data(
        "../results/predictions/test/lstm_uncertainties.csv"
    )
    st.session_state.test_tft_predictions = load_data(
        "../results/predictions/test/tft_predictions.csv"
    )
    st.session_state.test_tft_uncertainties = load_data(
        "../results/predictions/test/tft_uncertainties.csv"
    )

    st.session_state.arima_metrics = pd.read_csv(
        f"{metrics_dir}/test_arima_metrics.csv", index_col="Model"
    ).T
    st.session_state.lstm_metrics = pd.read_csv(
        f"{metrics_dir}/lstm_metrics.csv", index_col="Model_Type"
    )
    st.session_state.tft_metrics = pd.read_csv(
        f"{metrics_dir}/tft_metrics.csv", index_col="Model_Type"
    )


# Load scaler
test_residual_scaled = scale_data(
    st.session_state.test_residuals_df["Residuals"], st.session_state.residual_scaler
)

# Create LSTM test set
LSTM_X_test, LSTM_y_test = create_lstm_dataset(test_residual_scaled, look_back)

# Create TFT test set
TFT_X_test, TFT_y_test = create_tft_dataset(test_residual_scaled, look_back)

# Load ARIMA-GARCH predictions
val_arima_log = st.session_state.train_residuals_df["SARIMA Prediction"]
test_arima_log = st.session_state.test_residuals_df["SARIMA Prediction"]
val_garch = st.session_state.train_residuals_df["GARCH Volatility"]
test_garch = st.session_state.test_residuals_df["GARCH Volatility"]

# Undo the log transformation
val_arima_pred = np.expm1(val_arima_log)
test_arima_pred = np.expm1(test_arima_log)

# Load exogenous variables
val_exog = st.session_state.val_pca_df.drop(columns=["btc_close"])
test_exog = st.session_state.test_pca_df.drop(columns=["btc_close"])
exog = pd.concat([val_exog, test_exog])


def fetch_data(start_date):
    # Step 1: Fetch new data
    # new_dataset = fetch_new_data(start_date)
    new_dataset = pd.read_csv(
        f"{data_dir}/new_dataset.csv", parse_dates=["Date"], index_col="Date"
    )

    # Impute remaining missing values using backward fill method
    new_dataset.ffill(inplace=True)
    new_dataset.bfill(inplace=True)

    # Append new data to the existing dataset
    new_dataset = new_dataset[~new_dataset.index.isin(st.session_state.dataset.index)]
    st.session_state.dataset = pd.concat([st.session_state.dataset, new_dataset])
    st.session_state.dataset = st.session_state.dataset.sort_index()
    # st.session_state.dataset.to_csv(f"{data_dir}/dataset.csv")

    # Step 1: Apply PCA to the new_dataset
    suitable_col = [
        "hash_rate_blockchain",
        "btc_sma_14",
        "btc_ema_14",
        "btc_bb_high",
        "btc_bb_low",
        "btc_bb_mid",
        "btc_bb_width",
        "btc_atr_14",
        "btc_trading_volume",
        "btc_volatility_index",
        "ARK Innovation ETF",
        "CBOE Volatility Index",
        "Shanghai Composite Index",
        "btc_close",
    ]
    new_dataset = apply_log_transform(new_dataset, suitable_col)
    new_pca_df = transform_dataset(
        new_dataset, st.session_state.scaler, st.session_state.pca, "btc_close"
    )

    # Update the test_pca_df with new_pca_df
    new_pca_df = new_pca_df[~new_pca_df.index.isin(st.session_state.test_pca_df.index)]
    st.session_state.test_pca_df = pd.concat([st.session_state.test_pca_df, new_pca_df])
    st.session_state.test_pca_df = st.session_state.test_pca_df.sort_index()
    # st.session_state.test_pca_df.to_csv(f"{data_dir}/test_pca_df.csv") #######################################

    # Step 2: Predict new_pca_df on ARIMA-GARCH
    new_exog = new_pca_df.drop(columns=["btc_close"])
    new_arima_metrics_df, new_residuals_df = arima_garch_eval(
        model_dir, results_dir, len(new_exog), new_pca_df["btc_close"], "test", new_exog
    )
    # concat test_residuals_df and new_residuals_df
    new_residuals_df = new_residuals_df[
        ~new_residuals_df.index.isin(st.session_state.test_residuals_df.index)
    ]
    # Add look_back of test_residuals_df to new_residuals_df to maintain continuity when create time series dataset
    new_residuals_df_2 = pd.concat(
        [st.session_state.test_residuals_df[-look_back:], new_residuals_df]
    )

    # Update the test_residuals_df with new_residuals_df
    st.session_state.test_residuals_df = pd.concat(
        [st.session_state.test_residuals_df, new_residuals_df]
    )
    st.session_state.test_residuals_df = st.session_state.test_residuals_df.sort_index()
    # st.session_state.test_residuals_df.to_csv(f"{data_dir}/test_residuals_df.csv") #######################################

    # Step 3: Predict new_residuals_df_2 on LSTM and TFT
    # Load scaler
    new_residual_scaled = scale_data(
        new_residuals_df_2["Residuals"], st.session_state.residual_scaler
    )
    data_index = new_residuals_df_2["Residuals"].index[look_back:]

    # LSTM predictions
    LSTM_X_new, LSTM_y_new = create_lstm_dataset(new_residual_scaled, look_back)
    new_lstm_pred, new_lstm_uncertainty, lstm_metrics = LSTM_eval_new(
        model_dir,
        results_dir,
        model_types,
        data_index,
        new_pca_df["btc_close"],
        new_residuals_df_2["SARIMA Prediction"][look_back:],
        LSTM_X_new,
        LSTM_y_new,
        st.session_state.residual_scaler,
        n_simulations=100,
    )
    # Update lstm_predictions
    new_lstm_pred = new_lstm_pred[
        ~new_lstm_pred.index.isin(st.session_state.test_lstm_predictions.index)
    ]
    st.session_state.test_lstm_predictions = pd.concat(
        [st.session_state.test_lstm_predictions, new_lstm_pred]
    )
    st.session_state.test_lstm_predictions = (
        st.session_state.test_lstm_predictions.sort_index()
    )
    # st.session_state.test_lstm_predictions.to_csv("../results/predictions/test/lstm_predictions.csv") #######################################
    # Update lstm_uncertainties
    new_lstm_uncertainty = new_lstm_uncertainty[
        ~new_lstm_uncertainty.index.isin(st.session_state.test_lstm_uncertainties.index)
    ]
    st.session_state.test_lstm_uncertainties = pd.concat(
        [st.session_state.test_lstm_uncertainties, new_lstm_uncertainty]
    )
    st.session_state.test_lstm_uncertainties = (
        st.session_state.test_lstm_uncertainties.sort_index()
    )
    # st.session_state.test_lstm_uncertainties.to_csv("../results/predictions/test/lstm_uncertainties.csv") #######################################

    # Update lstm_metrics
    # Set "Model_Type" as index
    lstm_metrics.set_index("Model_Type", inplace=True)
    st.session_state.lstm_metrics = lstm_metrics
    # st.session_state.lstm_metrics.to_csv(f"{results_dir}/metrics/test_lstm_metrics.csv", index=True)

    # TFT predictions
    TFT_X_new, TFT_y_new = create_tft_dataset(new_residual_scaled, look_back)
    new_tft_pred, new_tft_uncertainty, tft_metrics = TFT_eval_new(
        model_dir,
        results_dir,
        data_index,
        new_pca_df["btc_close"],
        new_residuals_df_2["SARIMA Prediction"][look_back:],
        TFT_X_new,
        TFT_y_new,
        look_back,
        st.session_state.residual_scaler,
        n_simulations=100,
    )

    # Update tft_predictions
    new_tft_pred = new_tft_pred[
        ~new_tft_pred.index.isin(st.session_state.test_tft_predictions.index)
    ]
    st.session_state.test_tft_predictions = pd.concat(
        [st.session_state.test_tft_predictions, new_tft_pred]
    )
    st.session_state.test_tft_predictions = (
        st.session_state.test_tft_predictions.sort_index()
    )
    # st.session_state.test_tft_predictions.to_csv("../results/predictions/test/tft_predictions.csv") #######################################
    # Update tft_uncertainties
    new_tft_uncertainty = new_tft_uncertainty[
        ~new_tft_uncertainty.index.isin(st.session_state.test_tft_uncertainties.index)
    ]
    st.session_state.test_tft_uncertainties = pd.concat(
        [st.session_state.test_tft_uncertainties, new_tft_uncertainty]
    )
    st.session_state.test_tft_uncertainties = (
        st.session_state.test_tft_uncertainties.sort_index()
    )
    # st.session_state.test_tft_uncertainties.to_csv("../results/predictions/test/tft_uncertainties.csv") #######################################

    # Update tft_metrics
    tft_metrics.set_index("Model_Type", inplace=True)
    st.session_state.tft_metrics = tft_metrics
    # st.session_state.tft_metrics.to_csv(f'{results_dir}/metrics/test_tft_metrics.csv', index=True)


# Streamlit App
st.title("Future Bitcoin Price Prediction")

# Sidebar inputs
st.sidebar.image("../img/logo.png", width=200)

# Fetch latest data
if st.sidebar.button("Fetch Data"):
    # Get the last date from the dataset
    last_date = (
        st.session_state.dataset.index[-1].date().strftime("%Y-%m-%d")
    )  # Get the last date in main dataset
    last_33th_date = (
        st.session_state.dataset.index[-33].date().strftime("%Y-%m-%d")
    )  # Get the last date in main dataset
    today = datetime.now().date().strftime("%Y-%m-%d")  # Get today's date

    # Check if we need to fetch new data
    if last_date < today:
        st.write(f"Fetching new data from {last_date}...")
        # Fetch and update the dataset
        fetch_data(last_33th_date)
        st.success("Dataset updated successfully!")
    else:
        st.info("No need to fetch new data; the dataset is up-to-date.")

# Display dataset
st.line_chart(st.session_state.dataset["btc_close"])

st.sidebar.title("Input Parameters")
future_days = st.sidebar.number_input(
    "Enter the number of days for future prediction:",
    min_value=1,
    max_value=731,
    value=7,
    step=1,
)
selected_model = st.sidebar.selectbox("Select a Model:", model_options)


if st.sidebar.button("Predict"):
    st.write(f"Generating predictions for the next {future_days} days...")

    # Generate ARIMA-GARCH future predictions
    arimax_garch_future_df = arima_garch_forecast(exog, model_dir, future_days)

    print(arimax_garch_future_df)

    # Generate future predictions
    if selected_model == "Temporal-Fusion-Transformer":
        future_predictions_df = TFT_forecast(
            X_test=TFT_X_test,
            test_residuals_df=st.session_state.test_residuals_df,
            scaler=st.session_state.residual_scaler,
            arimax_garch_future=arimax_garch_future_df,
            model_path=model_dir,
            look_back=7,
            future_days=future_days,
            device="cpu",
        )
    else:
        future_predictions_df = LSTM_forecast(
            model_dir=model_dir,
            model_types=model_types,
            look_back=look_back,
            scaler=st.session_state.residual_scaler,
            test_residual_scaled=test_residual_scaled,
            test_arima_garch_log=test_arima_log,
            future_days=future_days,
            save_path=None,
            arimax_garch_future=arimax_garch_future_df,
        )

    # Display summary metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Future Days", value=future_days)
    with col2:
        st.metric(label="Selected Model", value=selected_model)

    st.markdown("---")  # Add a horizontal line to separate sections

    # Visualization for the selected model
    st.write(
        f"### Predicted Prices Across Train, Validation, Test, and Future ({selected_model} Model)"
    )
    fig = go.Figure()

    # Add actual prices (using black for good contrast on light background)
    fig.add_trace(
        go.Scatter(
            x=st.session_state.dataset.index,
            y=st.session_state.dataset["btc_close"],
            mode="lines",
            name="Actual Prices",
            line=dict(color="black", width=2),
        )
    )

    # Add ARIMA-GARCH predictions (Validation) with a brighter blue
    fig.add_trace(
        go.Scatter(
            x=val_arima_pred.index,
            y=val_arima_pred,
            mode="lines",
            name="SARIMA Predictions (Val)",
            line=dict(color="dodgerblue", width=2),
        )
    )

    z = 1.96

    lower_bound1 = np.expm1(val_arima_log - z * np.sqrt(val_garch))
    upper_bound1 = np.expm1(val_arima_log + z * np.sqrt(val_garch))

    # Add uncertainty bands for validation predictions
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([val_arima_pred.index, val_arima_pred.index[::-1]]),
            y=np.concatenate([lower_bound1, upper_bound1[::-1]]),
            fill="toself",
            fillcolor="rgba(211,211,211,0.5)",  # subtle gray fill
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="x+y",
            name="GARCH Predict (Val)",
        )
    )

    # Add SARIMA predictions (Test) using a medium blue
    fig.add_trace(
        go.Scatter(
            x=test_arima_pred.index,
            y=test_arima_pred,
            mode="lines",
            name="SARIMA Predictions (Test)",
            line=dict(color="mediumblue", width=2),
        )
    )

    lower_bound2 = np.expm1(test_arima_log - z * np.sqrt(test_garch))
    upper_bound2 = np.expm1(test_arima_log + z * np.sqrt(test_garch))

    # Add uncertainty bands for test predictions
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([test_arima_pred.index, test_arima_pred.index[::-1]]),
            y=np.concatenate([lower_bound2, upper_bound2[::-1]]),
            fill="toself",
            fillcolor="rgba(211,211,211,0.5)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="x+y",
            name="GARCH Predict (Test)",
        )
    )

    # Choose test predictions from the appropriate model
    if selected_model == "Temporal-Fusion-Transformer":
        pred = st.session_state.test_tft_predictions[selected_model]
    else:
        pred = st.session_state.test_lstm_predictions[selected_model]

    # Add Hybrid Model Test Predictions in forest green
    fig.add_trace(
        go.Scatter(
            x=st.session_state.test_residuals_df.index[look_back:],
            y=pred,
            mode="lines",
            name=f"SARIMA {selected_model} Hybrid Model Test Predictions",
            line=dict(color="forestgreen", width=2),
        )
    )

    # Retrieve uncertainties for the hybrid model
    if selected_model == "Temporal-Fusion-Transformer":
        uncertainty = st.session_state.test_tft_uncertainties[selected_model]
    else:
        uncertainty = st.session_state.test_lstm_uncertainties[selected_model]

    lower_bound = pred * np.exp(-z * uncertainty)
    upper_bound = pred * np.exp(z * uncertainty)
    uncertainty_date = st.session_state.test_residuals_df.index[look_back:]

    # Add uncertainty bands for hybrid model predictions
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([uncertainty_date, uncertainty_date[::-1]]),
            y=np.concatenate([lower_bound, upper_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(211,211,211,0.5)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="x+y",
            name="LSTM Uncertainty (±2 std)",
        )
    )

    # concat last values of pred to future_predictions_df
    future_predictions_df = pd.concat([pred[-1:], future_predictions_df])

    # Add future predictions with a striking red color
    fig.add_trace(
        go.Scatter(
            x=future_predictions_df.index,
            y=future_predictions_df[selected_model],
            mode="lines",
            name=f"{selected_model} Future Predictions",
            line=dict(color="crimson", width=2),
        )
    )

    # Update layout with a light theme and enhanced styling
    fig.update_layout(
        title=f"Bitcoin Price Predictions ({selected_model} Model)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all", label="All"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )


    # Display interactive chart
    st.plotly_chart(fig)

    # Modify metrics dataframes
    lstm_metrics = st.session_state.lstm_metrics.drop(columns=["Best params"])
    tft_metrics = st.session_state.tft_metrics.drop(columns=["Length"])
    # st.session_state.lstm_metrics.set_index("Model_Type", inplace=True)
    # st.session_state.tft_metrics.set_index("Model_Type", inplace=True)
    lstm_metrics.index.name = "Metric"
    tft_metrics.index.name = "Metric"
    lstm_metrics = lstm_metrics.T
    tft_metrics = tft_metrics.T
    # concat lstm_metrics and tft_metrics based on the same index
    metrics = pd.concat(
        [lstm_metrics, tft_metrics], axis=1
    )

    # Display evaluation metrics
    st.markdown("---")
    st.write("### Evaluation Metrics")

    st.write("#### ARIMA")
    st.dataframe(st.session_state.arima_metrics["sarima_test"][:-1])

    st.write("#### LSTM & TFT")
    st.dataframe(metrics.apply(
                pd.to_numeric, errors="coerce"
            )
    )
        
    

    # Display prediction values
    st.markdown("---")
    st.write("### Prediction Data")
    if selected_model == "Temporal-Fusion-Transformer":
        st.dataframe(future_predictions_df)
    else:
        st.dataframe(future_predictions_df[selected_model])

    # Download predictions
    st.download_button(
        label="Download Future Prediction Results",
        data=future_predictions_df.to_csv().encode("utf-8"),
        file_name="future_predictions.csv",
        mime="text/csv",
    )
