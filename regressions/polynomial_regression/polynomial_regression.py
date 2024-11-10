from utils.utils import load_df_from_csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

"""
Theory References:
https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch7slides-polynomial-regression.pdf
https://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf
"""


def split_data(
    data: pd.DataFrame, test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = data.loc[:, ["input"]]
    y = data["output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return data_train, data_test


def fit(data_train: pd.DataFrame, poly_degree: int) -> np.ndarray:
    X = data_train.loc[:, ["input"]].to_numpy()
    y = data_train["output"].to_numpy()
    beta = get_beta_matrix(X, y, poly_degree)
    return beta


def get_beta_matrix(
    X_input: np.ndarray, y_actual: np.ndarray, poly_degree: int
) -> np.ndarray:
    X = get_polynomial_features(X_input, poly_degree)
    y = y_actual

    b = np.matmul(X.T, X)
    b = np.linalg.inv(b)
    b = np.matmul(b, X.T)
    b = np.matmul(b, y)

    return b


def get_polynomial_features(X_input: np.ndarray, poly_degree: int) -> np.ndarray:
    column_size = poly_degree + 1
    vector_input = X_input.reshape(-1)
    input_size = vector_input.shape[0]
    poly_features = np.empty((input_size, column_size))

    for degree in range(column_size):
        poly_features[:, degree] = vector_input**degree

    return poly_features


def evaluate(
    beta: np.ndarray, data_test: pd.DataFrame, visualize: bool = False
) -> None:
    X = data_test.loc[:, ["input"]].to_numpy()
    y_actual = data_test["output"].to_numpy()

    y_predicted = predict_data(beta, X)
    mae = calculate_mae(y_predicted, y_actual)
    print("Mean Absolute Error: ", mae)

    if visualize:
        visualize_prediction(data_test, beta)


def predict_data(beta: np.ndarray, X_input: np.ndarray) -> np.ndarray:
    degree = beta.shape[0] - 1
    X = get_polynomial_features(X_input, degree)
    b = beta
    return np.matmul(X, b)


def calculate_mae(y_predictions: np.ndarray, y_actual: np.ndarray) -> float:
    return np.mean(np.absolute(y_actual - y_predictions))


def visualize_prediction(data_test: pd.DataFrame, beta: np.ndarray) -> None:
    OUTPUT_PATH = "regressions/polynomial_regression/output/test_data_predicted.jpg"

    x_input = data_test["input"].to_numpy()
    y_actual = data_test["output"].to_numpy()

    upper_x, lower_x = x_input.max(), x_input.min()
    x_space = np.linspace(lower_x, upper_x, 50)
    reggresion_line = np.zeros((len(x_space)))

    for i in range(len(beta)):
        reggresion_line += beta[i] * (x_space**i)

    plt.figure(figsize=(10, 5))
    plt.title("Polynomial Line")
    plt.plot(x_space, reggresion_line, color="blue")
    plt.scatter(x_input, y_actual)
    plt.savefig(OUTPUT_PATH)
    plt.show()


if __name__ == "__main__":
    df = load_df_from_csv("regressions/polynomial_regression/datasets/poly_reg.csv")
    df_train, df_test = split_data(df)
    beta = fit(df_train, poly_degree=2)
    evaluate(beta, df_test, visualize=True)
