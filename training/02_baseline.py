"""Step 2: a classical baseline (TF-IDF + logistic regression), so the fine-tuned model has something
honest to beat. Output: results/baseline.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).parent
train = pd.read_csv(HERE / "data/train.csv")
test = pd.read_csv(HERE / "data/test.csv")

model = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000),
    LogisticRegression(max_iter=2000, C=4.0),
)
model.fit(train["text"], train["label_id"])
pred = model.predict(test["text"])
result = {
    "model": "TF-IDF (character 2-5 grams) + logistic regression",
    "test_accuracy": round(accuracy_score(test["label_id"], pred), 4),
    "test_macro_f1": round(f1_score(test["label_id"], pred, average="macro"), 4),
    "test_size": len(test),
}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results/baseline.json").write_text(json.dumps(result, indent=2))
print(result)
