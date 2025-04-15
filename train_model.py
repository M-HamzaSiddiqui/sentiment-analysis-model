import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('imdb_dataset.csv')

df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    LinearSVC()
)

pipeline.fit(X_train, y_train)

print(classification_report(y_test, pipeline.predict(X_test)))

joblib.dump(pipeline, 'app/model/imdb_sentiment_svm.pkl')

print("Model training complete and saved")



