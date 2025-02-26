import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress specific ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)


def load_arima_garch_models(model_dir):
    # Load the ARIMAX & GARCH model
    arimax_results = joblib.load(f"{model_dir}/arimax_model.pkl")
    garch_fit = joblib.load(f"{model_dir}/garch_model.pkl")

    return arimax_results, garch_fit


def arima_garch_eval_old(model_dir, steps, actual, split_type, exog=None):
    # Load the ARIMAX & GARCH model
    sarima_model, garch_model = load_arima_garch_models(model_dir)

    sarima_pred = sarima_model.forecast(steps=steps, exog=exog).values.flatten()
    garch_volatility = garch_model.forecast(horizon=steps).variance.values[-1]

    actual_org = np.expm1(actual)
    sarima_pred_org = np.expm1(sarima_pred)

    mse = mean_squared_error(actual_org, sarima_pred_org)  # MSE
    rmse = root_mean_squared_error(actual_org, sarima_pred_org)  # RMSE
    mae = mean_absolute_error(actual_org, sarima_pred_org)  # MAE
    mape = mean_absolute_percentage_error(actual_org, sarima_pred_org)  # MAPE
    r2 = r2_score(actual_org, sarima_pred_org)  # R2

    # Create a new DataFrame for ARIMA metrics
    arima_metrics_df = pd.DataFrame(
        {
            "Model": [f"sarima_{split_type}"],
            "MSE": [mse],
            "RMSE": [rmse],
            "MAE": [mae],
            "MAPE": [mape],
            "R2": [r2],
            "Length": [steps],
        }
    )

    residuals = actual - sarima_pred

    residuals_df = pd.DataFrame(
        {
            "Date": actual.index,
            "SARIMA Prediction": sarima_pred,
            "GARCH Volatility": garch_volatility,
            "Residuals": residuals,
        }
    )

    return arima_metrics_df, residuals_df


def arima_garch_eval(model_dir, results_dir, steps, actual, split_type, exog=None):
    # Load the ARIMAX & GARCH model
    sarima_model, garch_model = load_arima_garch_models(model_dir)

    sarima_pred = sarima_model.forecast(steps=steps, exog=exog).values.flatten()
    garch_volatility = garch_model.forecast(horizon=steps).variance.values[-1]

    actual_org = np.expm1(actual)
    sarima_pred_org = np.expm1(sarima_pred)

    mse = mean_squared_error(actual_org, sarima_pred_org)  # MSE
    rmse = root_mean_squared_error(actual_org, sarima_pred_org)  # RMSE
    mae = mean_absolute_error(actual_org, sarima_pred_org)  # MAE
    mape = mean_absolute_percentage_error(actual_org, sarima_pred_org)  # MAPE
    r2 = r2_score(actual_org, sarima_pred_org)  # R2

    test_arima_metrics_df = pd.read_csv(
        f"{results_dir}/metrics/{split_type}_arima_metrics.csv"
    )
    mse2, rmse2, mae2, mape2, r22, steps2 = test_arima_metrics_df[
        ["MSE", "RMSE", "MAE", "MAPE", "R2", "Length"]
    ].values[0]

    # Compute weighted averages
    total_length = steps + steps2
    combined_mse = (mse * steps + mse2 * steps2) / total_length
    combined_rmse = (rmse * steps + rmse2 * steps2) / total_length
    combined_mae = (mae * steps + mae2 * steps2) / total_length
    combined_mape = (mape * steps + mape2 * steps2) / total_length
    combined_r2 = (r2 * steps + r22 * steps2) / total_length

    # Create a new DataFrame for ARIMA metrics
    arima_metrics_df = pd.DataFrame(
        {
            "Model": [f"sarima_{split_type}"],
            "MSE": [combined_mse],
            "RMSE": [combined_rmse],
            "MAE": [combined_mae],
            "MAPE": [combined_mape],
            "R2": [combined_r2],
            "Length": [total_length],
        }
    )

    # arima_metrics_df.to_csv(f"../results/metrics/{split_type}_arima_metrics.csv", index=False)

    # Residuals
    residuals = actual - sarima_pred
    residuals_df = pd.DataFrame(
        {
            "Date": actual.index,
            "SARIMA Prediction": sarima_pred,
            "GARCH Volatility": garch_volatility,
            "Residuals": residuals,
        }
    ).set_index("Date")

    return arima_metrics_df, residuals_df


def arima_garch_forecast(exog, model_dir, future_days):
    future_dates = pd.date_range(
        start=exog.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="D"
    )

    # Create a DataFrame to store predicted values
    future_exog = pd.DataFrame(index=future_dates, columns=exog.columns)

    # Predict future values for each exogenous variable
    for col in exog.columns:
        # Extract SARIMA parameters for the current indicator from sarima_params
        # Fit ARIMA to the historical data
        model = SARIMAX(exog[col], order=(1, 1, 3))
        model_fit = model.fit(disp=False)

        # Forecast future values
        forecast = model_fit.forecast(steps=future_days, index=future_dates)
        future_exog[col] = forecast

    # Load the ARIMAX & GARCH model
    arimax_results = joblib.load(f"{model_dir}/arimax_model.pkl")
    garch_fit = joblib.load(f"{model_dir}/garch_model.pkl")

    # Forecast ARIMAX for the next 6 days
    arimax_forecast_future = arimax_results.forecast(
        steps=future_days, exog=future_exog
    ).values.flatten()

    # Forecast GARCH for the next 6 days
    garch_forecast_future = garch_fit.forecast(
        horizon=future_days, method="simulation"
    ).variance.values[-1]

    random_noise = np.random.normal(
        loc=0, scale=garch_forecast_future, size=future_days
    )

    # Combine ARIMAX and GARCH forecasts (log-transformed scale)
    arimax_garch_future = arimax_forecast_future + random_noise

    return arimax_garch_future


def main():
    model_dir = "../../models"
    results_dir = "../../results"
    # VAL
    test_pca_df = pd.read_csv(
        "../../data/final/val_pca_df.csv", parse_dates=["Date"], index_col="Date"
    )
    test_exog = test_pca_df.drop(columns=["btc_close"])
    arima_metrics_df, residuals_df = arima_garch_eval_old(
        model_dir, len(test_exog), test_pca_df["btc_close"], "val", test_exog
    )

    print(arima_metrics_df)
    print(residuals_df.head())

    # arima_metrics_df.to_csv("../../results/metrics/val_arima_metrics.csv", index=False)
    # residuals_df.to_csv("../../data/final/train_residuals_df.csv", index=False)

    # TEST
    # test_pca_df = pd.read_csv("../../data/final/test_pca_df.csv", parse_dates=["Date"], index_col="Date")
    # test_exog = test_pca_df.drop(columns=["btc_close"])
    # arima_metrics_df, residuals_df= arima_garch_eval_old(model_dir, len(test_exog), test_pca_df["btc_close"], "test", test_exog)

    # print(arima_metrics_df)
    # print(residuals_df.head())

    # arima_metrics_df.to_csv("../../results/metrics/test_arima_metrics.csv", index=False)
    # residuals_df.to_csv("../../data/final/test_residuals_df.csv", index=False)

    # # NEW TEST
    # test_pca_df = pd.read_csv("../../data/final/new_pca_df.csv", parse_dates=["Date"], index_col="Date")
    # test_exog = test_pca_df.drop(columns=["btc_close"])
    # arima_metrics_df, residuals_df= arima_garch_eval(model_dir, results_dir, len(test_exog), test_pca_df["btc_close"], "test", test_exog)

    # print(arima_metrics_df)
    # print(residuals_df.head())
    # residuals_df.reset_index(inplace=True)

    # arima_metrics_df.to_csv("../../results/metrics/test_arima_metrics.csv", index=False)
    # residuals_df.to_csv("../../data/final/new_residuals_df.csv", index=False)


if __name__ == "__main__":
    main()
