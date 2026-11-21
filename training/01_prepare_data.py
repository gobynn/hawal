"""Step 1: download the dataset, clean it, and make fixed train / validation / test splits.

Dataset: nazimali/kurdish-news-headlines (Hugging Face, CC BY 4.0), Central Kurdish (Sorani) news
headlines labelled by topic. Output: data/{train,val,test}.csv and data/labels.json.
"""
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
SEED = 42


def clean(text: str) -> str:
    # Same normalisation the website applies before running the model.
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ة", "ە").replace("ھ", "ه").replace("ـ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


raw = load_dataset("nazimali/kurdish-news-headlines", split="train").to_pandas()
print("raw rows:", len(raw))
df = raw[["text", "label", "text-english"]].rename(columns={"text-english": "english"})
df["text"] = df["text"].astype(str).map(clean)
df = df[df["text"].str.len() >= 8]
df = df.drop_duplicates(subset="text")
print("after cleaning + dedupe:", len(df))
print(df["label"].value_counts().to_string())

labels = sorted(df["label"].unique())
(OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
df["label_id"] = df["label"].map({l: i for i, l in enumerate(labels)})

train, rest = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=SEED)
val, test = train_test_split(rest, test_size=0.5, stratify=rest["label_id"], random_state=SEED)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(part))
