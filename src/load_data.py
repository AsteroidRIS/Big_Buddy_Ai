import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loading...")
    return df

if __name__ == "__main__":
    path = "C:\\Users\\Krishna Gupta\\PycharmProjects\\Big_Buddy_Ai\\dataset\\Train\\csv\\Combine_Training_Data.csv"
    load_data(path)
