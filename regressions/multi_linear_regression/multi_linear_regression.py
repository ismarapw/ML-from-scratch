from utils.utils import load_df_from_csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

"""
Theory References:
https://www.sjsu.edu/faculty/guangliang.chen/Math261a/Ch3slides-multiple-linear-regression.pdf
https://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf
"""


def calculate_mae(y_predictions: np.ndarray, y_actual: np.ndarray) -> float:
    return np.mean(np.absolute(y_actual - y_predictions))


def get_beta_matrix(X_input: np.ndarray, y_actual: np.ndarray) -> np.ndarray:
    X = X_input
    y = y_actual

    b = np.matmul(X.T, X)
    b = np.linalg.inv(b)
    b = np.matmul(b, X.T)
    b = np.matmul(b, y)

    return b


def predict_data(beta: np.ndarray, X_input: np.ndarray) -> np.ndarray:
    X = X_input
    b = beta
    return np.matmul(X, b)


def split_data(
    data: pd.DataFrame, test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = data.loc[:, ["feature_1", "feature_2"]]
    y = data["output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return data_train, data_test


def fit(data_train: pd.DataFrame) -> np.ndarray:
    X = data_train.loc[:, ["feature_1", "feature_2"]].to_numpy()
    y = data_train["output"].to_numpy()
    beta = get_beta_matrix(X, y)
    return beta


def visualize_prediction(
    beta: np.ndarray, y_actual: np.ndarray, X_input: np.ndarray
) -> None:
    OUTPUT_PATH = "regressions/multi_linear_regression/output/test_data_predicted.jpg"

    X = X_input
    upper_x1, lower_x1 = X[:, 0].max(), X[:, 0].min()
    upper_x2, lower_x2 = X[:, 1].max(), X[:, 1].min()

    x1_space = np.linspace(lower_x1, upper_x1, 30)
    x2_space = np.linspace(lower_x2, upper_x2, 30)
    x1_grid, x2_grid = np.meshgrid(x1_space, x2_space)
    y_preds = x1_grid * beta[0] + x2_grid * beta[1]

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot_surface(x1_grid, x2_grid, y_preds, alpha=0.5)
        ax.scatter(X[:, 0], X[:, 1], y_actual)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    fig.suptitle("Predicted data")
    plt.savefig(OUTPUT_PATH)
    plt.show()


def evaluate(
    beta: np.ndarray, data_test: pd.DataFrame, visualize: bool = False
) -> None:
    X = data_test.loc[:, ["feature_1", "feature_2"]].to_numpy()
    y_actual = data_test["output"].to_numpy()

    y_predicted = predict_data(beta, X)
    mae = calculate_mae(y_predicted, y_actual)
    print("Mean Absolute Error: ", mae)

    if visualize:
        visualize_prediction(beta, y_actual, X)


if __name__ == "__main__":
    df = load_df_from_csv(
        "regressions/multi_linear_regression/datasets/multi_linear_reg.csv"
    )
    df_train, df_test = split_data(df)
    beta = fit(df_train)
    evaluate(beta, df_test, visualize=True)
