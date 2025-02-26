import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from utils.data_preparation import create_lstm_dataset, load_data, scale_data
from utils.lstm_model import *
# from data_preparation import create_lstm_dataset, load_data, scale_data
# from lstm_model import *


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return (
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )


def perform_mc_dropout(keras_model, X_test, n_simulations=100):
    preds = []
    for _ in range(n_simulations):
        pred = keras_model(X_test, training=True)
        preds.append(pred.numpy())
    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)
    uncertainty = np.std(preds, axis=0)
    return mean_pred, uncertainty


def evaluate_predictions(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, rmse, mae, mape, r2


def ensemble_prediction(predictions, uncertainties, actual):
    ensemble_mean_pred = np.mean(
        [predictions[model_type] for model_type in predictions], axis=0
    )
    ensemble_variance = np.sum(
        [uncertainties[model_type] ** 2 for model_type in uncertainties], axis=0
    )
    ensemble_uncertainty = np.sqrt(ensemble_variance)

    mse, rmse, mae, mape, r2 = evaluate_predictions(actual, ensemble_mean_pred)

    predictions["Ensemble-LSTM"] = ensemble_mean_pred
    uncertainties["Ensemble-LSTM"] = ensemble_uncertainty

    return (
        {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2},
        predictions,
        uncertainties,
    )


def LSTM_eval_test(
    model_dir,
    model_types,
    data_index,
    test_actual,
    test_arima,
    X_test,
    y_test,
    scaler,
    n_simulations=100,
):
    test_actual = np.expm1(test_actual)
    test_arima = test_arima.to_numpy()
    predictions = {}
    uncertainties = {}
    evaluation_metrics = {}

    for model_type in model_types:
        model_file = f"{model_dir}/{model_type}_best_model.pkl"
        param_file = f"{model_dir}/{model_type}_best_params.pkl"

        # Load model and parameters
        best_model = joblib.load(model_file)
        best_params = joblib.load(param_file)
        keras_model = best_model.model_

        # Perform MC Dropout
        mean_pred, uncertainty = perform_mc_dropout(keras_model, X_test, n_simulations)

        # Inverse transform predictions
        mean_pred = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
        uncertainty = scaler.inverse_transform(uncertainty.reshape(-1, 1)).flatten()

        predictions[model_type] = np.expm1(mean_pred + test_arima)
        uncertainties[model_type] = uncertainty

        # Evaluate predictions
        mse, rmse, mae, mape, r2 = evaluate_predictions(
            test_actual, predictions[model_type]
        )

        # Store metrics with best params
        evaluation_metrics[model_type] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "Best params": best_params,
        }

    # Perform ensemble evaluation
    ensemble_metrics, predictions, uncertainties = ensemble_prediction(
        predictions, uncertainties, test_actual
    )
    evaluation_metrics["Ensemble-LSTM"] = {
        "MSE": ensemble_metrics["MSE"],
        "RMSE": ensemble_metrics["RMSE"],
        "MAE": ensemble_metrics["MAE"],
        "MAPE": ensemble_metrics["MAPE"],
        "R2": ensemble_metrics["R2"],
        "Best params": "N/A (Ensemble)",
    }

    evaluation_metrics = pd.DataFrame.from_dict(
        evaluation_metrics, orient="index"
    ).reset_index()
    evaluation_metrics.columns = [
        "Model_Type",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Best params",
    ]

    # Flatten predictions and uncertainties
    predictions = {model: pred.flatten() for model, pred in predictions.items()}
    uncertainties = {model: uncert.flatten() for model, uncert in uncertainties.items()}

    # create dataframe for test predictions and uncertainties with index as data_index
    predictions_df = pd.DataFrame.from_dict(predictions, orient="index").transpose()
    uncertainties_df = pd.DataFrame.from_dict(uncertainties, orient="index").transpose()

    predictions_df.index = data_index
    uncertainties_df.index = data_index

    return predictions_df, uncertainties_df, evaluation_metrics


def LSTM_eval_new(
    model_dir,
    results_dir,
    model_types,
    data_index,
    test_actual,
    test_arima,
    X_test,
    y_test,
    scaler,
    n_simulations=100,
):
    test_actual = np.expm1(test_actual)
    test_arima = test_arima.to_numpy()
    predictions = {}
    uncertainties = {}
    evaluation_metrics = {}

    for model_type in model_types:
        model_file = f"{model_dir}/{model_type}_best_model.pkl"
        param_file = f"{model_dir}/{model_type}_best_params.pkl"

        # Load model and parameters
        best_model = joblib.load(model_file)
        best_params = joblib.load(param_file)
        keras_model = best_model.model_

        # Perform MC Dropout
        mean_pred, uncertainty = perform_mc_dropout(keras_model, X_test, n_simulations)

        # Inverse transform predictions
        mean_pred = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
        uncertainty = scaler.inverse_transform(uncertainty.reshape(-1, 1)).flatten()

        predictions[model_type] = np.expm1(mean_pred + test_arima)
        uncertainties[model_type] = uncertainty

        # Evaluate predictions
        mse, rmse, mae, mape, r2 = evaluate_predictions(
            test_actual, predictions[model_type]
        )

        # Store metrics with best params
        evaluation_metrics[model_type] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "Best params": best_params,
        }

    # Perform ensemble evaluation
    ensemble_metrics, predictions, uncertainties = ensemble_prediction(
        predictions, uncertainties, test_actual
    )
    evaluation_metrics["Ensemble-LSTM"] = {
        "MSE": ensemble_metrics["MSE"],
        "RMSE": ensemble_metrics["RMSE"],
        "MAE": ensemble_metrics["MAE"],
        "MAPE": ensemble_metrics["MAPE"],
        "R2": ensemble_metrics["R2"],
        "Best params": "N/A (Ensemble)",
    }

    evaluation_metrics = pd.DataFrame.from_dict(
        evaluation_metrics, orient="index"
    ).reset_index()
    evaluation_metrics.columns = [
        "Model_Type",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Best params",
    ]

    # Load test eval metrics & test predictions & uncertainties
    test_evaluation_metrics = pd.read_csv(f"{results_dir}/metrics/lstm_metrics.csv")
    test_predictions = pd.read_csv(
        f"{results_dir}/predictions/test/lstm_predictions.csv", index_col="Date"
    )

    # Flatten predictions and uncertainties
    predictions = {model: pred.flatten() for model, pred in predictions.items()}
    uncertainties = {model: uncert.flatten() for model, uncert in uncertainties.items()}

    # create dataframe for test predictions and uncertainties with index as data_index
    predictions_df = pd.DataFrame.from_dict(predictions, orient="index").transpose()
    uncertainties_df = pd.DataFrame.from_dict(uncertainties, orient="index").transpose()

    predictions_df.index = data_index
    uncertainties_df.index = data_index

    # Compute weighted averages of evaluation metrics
    combined_evaluation_metrics = {}
    total_length = len(y_test) + len(test_predictions["LSTM"])
    for model_type in test_evaluation_metrics["Model_Type"].values:
        combined_mse = (
            evaluation_metrics[evaluation_metrics["Model_Type"] == model_type][
                "MSE"
            ].values[0]
            * len(y_test)
            + test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["MSE"].values[0]
            * len(test_predictions[model_type])
        ) / total_length
        combined_rmse = (
            evaluation_metrics[evaluation_metrics["Model_Type"] == model_type][
                "RMSE"
            ].values[0]
            * len(y_test)
            + test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["RMSE"].values[0]
            * len(test_predictions[model_type])
        ) / total_length
        combined_mae = (
            evaluation_metrics[evaluation_metrics["Model_Type"] == model_type][
                "MAE"
            ].values[0]
            * len(y_test)
            + test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["MAE"].values[0]
            * len(test_predictions[model_type])
        ) / total_length
        combined_mape = (
            evaluation_metrics[evaluation_metrics["Model_Type"] == model_type][
                "MAPE"
            ].values[0]
            * len(y_test)
            + test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["MAPE"].values[0]
            * len(test_predictions[model_type])
        ) / total_length
        combined_r2 = (
            evaluation_metrics[evaluation_metrics["Model_Type"] == model_type][
                "R2"
            ].values[0]
            * len(y_test)
            + test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["R2"].values[0]
            * len(test_predictions[model_type])
        ) / total_length
        combined_evaluation_metrics[model_type] = {
            "MSE": combined_mse,
            "RMSE": combined_rmse,
            "MAE": combined_mae,
            "MAPE": combined_mape,
            "R2": combined_r2,
            "Best params": test_evaluation_metrics[
                test_evaluation_metrics["Model_Type"] == model_type
            ]["Best params"].values[0],
        }

    # Convert dict to DataFrame
    combined_evaluation_metrics = pd.DataFrame.from_dict(
        combined_evaluation_metrics, orient="index"
    ).reset_index()
    combined_evaluation_metrics.columns = [
        "Model_Type",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Best params",
    ]
    combined_evaluation_metrics.index = combined_evaluation_metrics["Model_Type"]

    # print(test_evaluation_metrics)
    # print(evaluation_metrics)

    return predictions_df, uncertainties_df, combined_evaluation_metrics


def LSTM_forecast(
    model_dir,
    model_types,
    look_back,
    scaler,
    test_residual_scaled,
    test_arima_garch_log,
    future_days,
    save_path,
    arimax_garch_future,
):
    future_dates = pd.date_range(
        test_arima_garch_log.index[-1], periods=future_days + 1, freq="D"
    )[1:]
    future_predictions = {}

    for model_type in model_types:
        model_file = f"{model_dir}/{model_type}_best_model.pkl"
        best_model = joblib.load(model_file)
        keras_model = best_model.model_

        input_sequence = test_residual_scaled[-look_back:].reshape(1, look_back, 1)
        future_residuals = []

        for _ in range(future_days):
            next_residual = keras_model.predict(input_sequence)[0, 0]
            future_residuals.append(next_residual)
            next_residual = np.array([[next_residual]])
            input_sequence = np.append(
                input_sequence[:, 1:, :], next_residual[:, np.newaxis, :], axis=1
            )

        future_residuals = np.array(future_residuals).reshape(-1, 1)
        future_residuals_inverse = scaler.inverse_transform(future_residuals).flatten()

        # arima_garch_future_pred = test_arima_garch_pred.iloc[-1]
        final_future_forecast_list = []
        # for residual in future_residuals_inverse:
        #     final_future_forecast = arima_garch_future_pred + residual
        #     final_future_forecast_list.append(final_future_forecast)
        final_future_forecast = arimax_garch_future + future_residuals_inverse
        final_future_forecast_list.append(final_future_forecast)

        future_predictions[model_type] = np.exp(final_future_forecast_list) - 1

    ensemble_future_pred = np.mean(
        [future_predictions[model_type] for model_type in future_predictions], axis=0
    )
    future_predictions["Ensemble-LSTM"] = ensemble_future_pred

    future_predictions = {
        key: np.squeeze(value) for key, value in future_predictions.items()
    }

    future_predictions_df = pd.DataFrame(future_predictions, index=future_dates)
    future_predictions_df.to_csv(save_path)

    return future_predictions_df


def main():
    model_dir = "../../models"
    results_dir = "../../results"
    look_back = 7
    scaler = joblib.load("../../models/residual_scaler.pkl")

    # Load arima metrics
    arima_metrics = pd.read_csv("../../results/metrics/test_arima_metrics.csv")

    # # Load test data
    test_pca_df = load_data("../../data/final/test_pca_df.csv")
    test_residual_df = load_data("../../data/final/test_residuals_df.csv")
    test_residual_scaled = scale_data(test_residual_df["Residuals"], scaler)

    model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]

    # Create LSTM test set
    LSTM_X_test, LSTM_y_test = create_lstm_dataset(test_residual_scaled, look_back)

    predictions_df, uncertainties_df, evaluation_metrics = LSTM_eval_test(
        model_dir, model_types, test_residual_df["Residuals"].index[look_back:], test_pca_df["btc_close"][look_back:], test_residual_df["SARIMA Prediction"][look_back:], LSTM_X_test, LSTM_y_test, scaler, n_simulations=100
    )

    print(arima_metrics)
    print(evaluation_metrics)
    print(predictions_df.head())
    print(uncertainties_df.head())

    evaluation_metrics.to_csv(f"{results_dir}/metrics/lstm_metrics.csv", index=False)
    predictions_df.to_csv(f"{results_dir}/predictions/test/lstm_predictions.csv", index=True)
    uncertainties_df.to_csv(f"{results_dir}/predictions/test/lstm_uncertainties.csv", index=True)

    # # Load new test data
    # new_pca_df = load_data("../../data/final/new_pca_df.csv")
    # test_residual_df = load_data("../../data/final/new_residuals_df.csv")
    # test_residual_scaled = scale_data(test_residual_df["Residuals"], scaler)

    # model_types = ["LSTM", "BiLSTM", "Attention-LSTM", "Attention-BiLSTM"]

    # # Create LSTM test set
    # LSTM_X_test, LSTM_y_test = create_lstm_dataset(test_residual_scaled, look_back)

    # predictions_df, uncertainties_df, combined_evaluation_metrics = LSTM_eval_new(
    #     model_dir,
    #     results_dir,
    #     model_types,
    #     test_residual_df["Residuals"].index[look_back:],
    #     new_pca_df["btc_close"][look_back:],
    #     test_residual_df["SARIMA Prediction"][look_back:],
    #     LSTM_X_test,
    #     LSTM_y_test,
    #     scaler,
    #     n_simulations=100,
    # )

    # print(combined_evaluation_metrics)
    # print(predictions_df.head())
    # print(uncertainties_df.head())

    # LSTM_forecast(
    #     model_dir, model_types, look_back, scaler, test_residual_scaled, test_arima_garch_log, future_days,
    #     f"{results_dir}/predictions/future/LSTM_future_pred.csv", arimax_garch_future
    # )


if __name__ == "__main__":
    main()
