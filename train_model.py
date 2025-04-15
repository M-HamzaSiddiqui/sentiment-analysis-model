import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import string

df = pd.read_csv('imdb_dataset.csv')
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

custom_data = pd.read_csv('custom_feedback.csv')
custom_data["sentiment"] = custom_data["label"].map({"positive": 1, "negative": 0})
custom_data.drop(columns=["label"], inplace=True)

df = pd.concat([df.rename(columns={"review": "text"}), custom_data], ignore_index=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5),
    LinearSVC()
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'app/model/imdb_sentiment_svm.pkl')

print("Model training complete and saved at app/model/imdb_sentiment_svm.pkl")
