from utils.utils import load_df_from_csv
from matplotlib import pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
import random

"""
Theory References:
https://utsavdesai26.medium.com/linear-regression-made-simple-a-step-by-step-tutorial-fb8e737ea2d9
"""


def calculate_mse(y_predictions: list, y_actual: list) -> float:
    sum_diff = 0
    n_data = len(y_predictions)

    for i in range(n_data):
        sum_diff += (y_actual[i] - y_predictions[i]) ** 2

    return sum_diff / n_data


def gradient_loss_m(y_predictions: list, y_actual: list, x_input: list) -> float:
    n_data = len(y_predictions)
    derivative_m = 0

    for i in range(n_data):
        derivative_m += (y_actual[i] - y_predictions[i]) * (x_input[i])

    derivative_m = (-2 / n_data) * derivative_m
    return derivative_m


def gradient_loss_c(y_predictions: list, y_actual: list) -> float:
    n_data = len(y_predictions)
    derivative_c = 0

    for i in range(n_data):
        derivative_c += y_actual[i] - y_predictions[i]

    derivative_c = (-2 / n_data) * derivative_c
    return derivative_c


def predict_input(x: float, m: float, c: float) -> float:
    return m * x + c


def split_data(
    data: pd.DataFrame, test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = data["input"]
    y = data["output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return data_train, data_test


def visualize_learning_error(error: list) -> None:
    OUTPUT_PATH = "regressions/linear_regression/output/learning_error.jpg"
    iterations = list(range(1, len(error) + 1))

    plt.title("Error per epoch")
    plt.plot(iterations, error)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig(OUTPUT_PATH)
    plt.show()


def fit(
    data_train: pd.DataFrame,
    iterations: int = 1000,
    learning_rate: float = 0.001,
    visualize_error: bool = False,
) -> Tuple[float, float]:
    m = random.uniform(0, 1)
    c = random.uniform(0, 1)
    actual_data = data_train["output"].to_list()
    input_data = data_train["input"].to_list()
    error = []

    for i in range(iterations):
        predicted_data = []
        
        for x in input_data:
            y_pred = predict_input(x, m, c)
            predicted_data.append(y_pred)

        loss = calculate_mse(predicted_data, actual_data)
        derivative_m = gradient_loss_m(predicted_data, actual_data, input_data)
        derivative_c = gradient_loss_c(predicted_data, actual_data)
        m = m - (learning_rate * derivative_m)
        c = c - (learning_rate * derivative_c)
        error.append(loss)
        print(f"Loss for iteration {i+1}: {loss}")

    if visualize_error:
        visualize_learning_error(error)

    return m, c


def visualize_predictions(
    data_test: pd.DataFrame, y_predictions: list, m: float, c: float
) -> None:
    OUTPUT_PATH = "regressions/linear_regression/output/test_data_predicted.jpg"

    x_input = data_test["input"].to_list()
    y_actual = data_test["output"].to_list()

    plt.figure(figsize=(10, 5))
    plt.title(f"Regression Line: m = {round(m,2)}, c = {round(c,2)}")
    plt.plot(x_input, y_predictions, color="blue")
    plt.scatter(x_input, y_actual)
    plt.savefig(OUTPUT_PATH)
    plt.show()


def evaluate(
    data_test: pd.DataFrame, m: float, c: float, visualize: bool = False
) -> None:
    actual_data = data_test["output"].to_list()
    predicted_data = []

    for row in data_test.itertuples():
        idx = row.Index
        x = data_test.at[idx, "input"]
        y_pred = predict_input(x, m, c)

        predicted_data.append(y_pred)

    error = calculate_mse(predicted_data, actual_data)
    print(f"Mean Squared Error for Test data : {round(error, 2)}")

    if visualize:
        visualize_predictions(data_test, predicted_data, m, c)


if __name__ == "__main__":
    df = load_df_from_csv("regressions/linear_regression/datasets/linear_reg.csv")
    df_train, df_test = split_data(df)

    iterations = 2000
    learning_rate = 0.005
    m, c = fit(
        df_train,
        iterations=iterations,
        learning_rate=learning_rate,
        visualize_error=True,
    )

    evaluate(df_test, m, c, visualize=True)
