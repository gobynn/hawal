"""Step 5: collect every result into one file the website reads, plus sample test headlines for the demo.
Output: ../web/src/data/results.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
ft = json.loads((R / "finetune.json").read_text())
test = pd.read_csv(HERE / "data/test.csv")
samples = (
    test.groupby("label").sample(3, random_state=7)
    .loc[:, ["text", "english", "label"]]
    .to_dict(orient="records")
)
site = {
    "dataset": {
        "name": "Kurdish News Headlines", "id": "nazimali/kurdish-news-headlines", "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/nazimali/kurdish-news-headlines",
        "raw_rows": 50000, "clean_rows": ft["train_size"] + ft["val_size"] + ft["test_size"],
        "splits": {"train": ft["train_size"], "validation": ft["val_size"], "test": ft["test_size"]},
        "label_counts": pd.concat([pd.read_csv(HERE / f"data/{s}.csv") for s in ("train", "val", "test")])["label"].value_counts().to_dict(),
    },
    "base_model": {"id": ft["base_model"], "url": f"https://huggingface.co/{ft['base_model']}"},
    "training": {k: ft[k] for k in ("epochs", "batch_size", "learning_rate", "max_length", "device", "training_minutes")},
    "labels": ft["labels"],
    "baseline": json.loads((R / "baseline.json").read_text()),
    "finetuned": {k: ft[k] for k in ("test_accuracy", "test_macro_f1")},
    "deployed": json.loads((R / "onnx_int8.json").read_text()),
    "per_label": {l: {m: round(ft["per_label"][l][m], 4) for m in ("precision", "recall", "f1-score", "support")} for l in ft["labels"]},
    "confusion_matrix": ft["confusion_matrix"],
    "log": json.loads((R / "training_log.json").read_text()),
    "experiments": [
        {
            "name": "Run 1: KuBERT's default tokenizer settings (accent stripping on)",
            "epochs": 2, "learning_rate": 3e-5,
            **{k: json.loads((R / "run1/finetune.json").read_text())[k] for k in ("test_accuracy", "test_macro_f1")},
            "note": "Accent stripping removed the hamza from ئ, so 8.1% of tokens became [UNK]. Scored below the baseline.",
        },
        {
            "name": "Run 2: accent stripping off, ى/ھ normalised (final model)",
            "epochs": ft["epochs"], "learning_rate": ft["learning_rate"],
            "test_accuracy": ft["test_accuracy"], "test_macro_f1": ft["test_macro_f1"],
            "note": "Unknown tokens fell to 4.6%, mostly punctuation and digits that KuBERT's vocabulary lacks.",
        },
    ] if (R / "run1/finetune.json").exists() else [],
    "samples": samples,
}
out = HERE.parent / "web/src/data/results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(site, indent=2, ensure_ascii=False))
print("wrote", out, "samples:", len(samples))
