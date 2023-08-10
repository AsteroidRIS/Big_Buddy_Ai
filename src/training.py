from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import time, datetime
from load_data import load_data
from filtering_text import filtering_text


def training(filtered_text):
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(representation_model=representation_model, verbose= True)
    print("Bert model training started...")
    start = datetime.datetime.now()
    topics, probs = topic_model.fit_transform(filtered_text)
    topic_model.save("Model")
    end = datetime.datetime.now()
    diff = (end - start)
    print("Model trains in ")
    print("Bert model training completed...")
    return topics, probs


if __name__ == "__main__":
    path = "C:\\Users\\Krishna Gupta\\PycharmProjects\\Big_Buddy_Ai\\dataset\\Train\\csv\\Combine_Training_Data.csv"
    df = load_data(path)
    filtered_text = filtering_text(df)
    training(filtered_text)
