from utils.utils import load_df_from_csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, Literal, Tuple

"""
References:
https://medium.com/@creatrohit9/k-nearest-neighbors-k-nn-the-distance-based-machine-learning-algorithm-96cfc684412d
https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
"""

def calculate_eucledian_distance(first_point: list, second_point: list) -> float:
    x1, y1 = first_point[0], first_point[1]
    x2, y2 = second_point[0], second_point[0]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_manhattan_distance(first_point: list, second_point: list) -> float:
    x1, y1 = first_point[0], first_point[1]
    x2, y2 = second_point[0], second_point[0]
    return math.sqrt(abs(x2 - x1) + abs(y2 - y1))


def choose_distance_function(
    distance_method: Literal["euclidean", "manhattan"] = "euclidean"
) -> Callable[[list, list], float]:
    if distance_method == "euclidean":
        return calculate_eucledian_distance
    elif distance_method == "manhattan":
        return calculate_manhattan_distance
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = data.copy()
    scaled_df[["feature_1", "feature_2"]] = scaled_data
    return scaled_df


def get_sample_point(data: pd.DataFrame) -> list:
    N_SAMPLE = 1
    sample = data.sample(n=N_SAMPLE)
    point = sample[["feature_1", "feature_2"]].to_numpy()[0]
    return point


def vote_class(labels: list) -> float:
    FIRST_LABEL, SECOND_LABEL = 0, 1
    n_first_label, n_second_label = 0, 0

    for label in labels:
        if label == FIRST_LABEL:
            n_first_label += 1
        elif label == SECOND_LABEL:
            n_second_label += 1
        else:
            raise ValueError(
                f"Label : {label} is not found, accepted labels: [0,1]"
            )

    return FIRST_LABEL if n_first_label >= n_second_label else SECOND_LABEL


def scale_and_split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = data[["feature_1", "feature_2"]]
    y = data["class"]
    X_scaled = scale_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=2
    )
    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return data_train, data_test


def classifiy_point(
    data_train: pd.DataFrame,
    point_target: list,
    k_neighbours: int = 5,
    distance_method: Literal["euclidean", "manhattan"] = "euclidean",
) -> float:
    distance_function = choose_distance_function(distance_method)
    distances_to_point = []

    for row in data_train.itertuples():
        idx = row.Index
        point = [data_train.at[idx, "feature_1"], data_train.at[idx, "feature_2"]]
        distance = distance_function(point, point_target)
        distances_to_point.append(distance)

    data_result = data_train.copy()
    data_result["distance"] = distances_to_point
    data_result = data_result.sort_values(by=["distance"])
    data_closest = data_result[:k_neighbours]

    neighbours_class = data_closest["class"].to_list()
    predicted_class = vote_class(neighbours_class)

    return predicted_class


def evaluate_knn():
    # TODO: calculate accuracy for entire test data
    pass

if __name__ == "__main__":
    df = load_df_from_csv("classifications/knn/datasets/knn_dataset.csv")
    df_train, df_test = scale_and_split_data(df)

    point_target = get_sample_point(df_test)
    label_predicted = classifiy_point(
        df_train, point_target, distance_method="euclidean"
    )
    print(label_predicted)
