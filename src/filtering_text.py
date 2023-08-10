import nltk
from nltk.stem import WordNetLemmatizer


from load_data import load_data


def filtering_text(df):
    df['content'] = df['content'].fillna('').astype(str)
    doc = df['content']
    filtered_text = []
    lemmatizer = WordNetLemmatizer()
    for i in doc:
        filtered_text.append(lemmatizer.lemmatize(i))
    print("Filtering of text completed...")
    return filtered_text


if __name__ == "__main__":
    path = "C:\\Users\\Krishna Gupta\\PycharmProjects\\Big_Buddy_Ai\\dataset\\Train\\csv\\Combine_Training_Data.csv"
    df = load_data(path)
    filtered_text = filtering_text(df)
