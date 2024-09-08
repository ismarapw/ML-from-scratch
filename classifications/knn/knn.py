from utils.utils import load_data_from_csv








if __name__ == "__main__":
    DATA_PATH = 'datasets/crop_recommendation.csv'
    data = load_data_from_csv(DATA_PATH)
    data = data[['humidity', 'rainfall', 'label']]
    print(data)