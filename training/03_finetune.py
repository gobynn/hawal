"""Step 3: fine-tune KuBERT (a Central Kurdish BERT, asosoft/KuBERT-Central-Kurdish-BERT-Model) to classify
Kurdish news headlines by topic. Runs on Apple Silicon (MPS), CUDA or CPU.
Output: model/ (fine-tuned weights + tokenizer), results/finetune.json, results/training_log.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup

HERE = Path(__file__).parent
BASE = "asosoft/KuBERT-Central-Kurdish-BERT-Model"
EPOCHS, BATCH, LR, MAX_LEN, SEED = 3, 32, 4e-5, 48, 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

labels = json.loads((HERE / "data/labels.json").read_text())
train_df, val_df, test_df = (pd.read_csv(HERE / f"data/{s}.csv") for s in ("train", "val", "test"))


def clean_tokenizer():
    """KuBERT's published tokenizer config points [UNK]/[PAD] at tokens missing from its vocab (and the first
    vocab line is "[PAD" without the bracket), so build a standard BERT WordPiece tokenizer from its vocab.txt.
    strip_accents=False matters: BERT's default accent stripping removes the hamza from ئ, which turned 8.1% of
    all tokens into [UNK] (4.6% without it, mostly punctuation and digits)."""
    from huggingface_hub import hf_hub_download
    vocab = Path(hf_hub_download(BASE, "vocab.txt")).read_text(encoding="utf-8").split("\n")
    if vocab[0] == "[PAD":
        vocab[0] = "[PAD]"
    fixed = HERE / "data/kubert-vocab.txt"
    fixed.write_text("\n".join(vocab), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(fixed), do_lower_case=True, strip_accents=False, unk_token="[UNK]", pad_token="[PAD]",
                             cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=MAX_LEN)


tok = clean_tokenizer()


class Headlines(Dataset):
    def __init__(self, df):
        self.enc = tok(list(df["text"]), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        self.y = torch.tensor(df["label_id"].values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.y[i]


loaders = {
    "train": DataLoader(Headlines(train_df), batch_size=BATCH, shuffle=True),
    "val": DataLoader(Headlines(val_df), batch_size=128),
    "test": DataLoader(Headlines(test_df), batch_size=128),
}

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)}
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * len(loaders["train"])
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, gold = [], []
    for x, y in loader:
        logits = model(**{k: v.to(device) for k, v in x.items()}).logits
        preds += logits.argmax(-1).cpu().tolist()
        gold += y.tolist()
    model.train()
    return np.array(gold), np.array(preds)


log, step, started = [], 0, time.time()
model.train()
for epoch in range(1, EPOCHS + 1):
    running = 0.0
    for i, (x, y) in enumerate(loaders["train"], 1):
        out = model(**{k: v.to(device) for k, v in x.items()}, labels=y.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        running += out.loss.item(); step += 1
        if step % 100 == 0:
            log.append({"step": step, "epoch": epoch, "train_loss": round(running / i, 4)})
            print(f"epoch {epoch} step {step}/{steps} loss {running / i:.4f} ({time.time() - started:.0f}s)", flush=True)
    g, p = evaluate(loaders["val"])
    log.append({"step": step, "epoch": epoch, "train_loss": round(running / len(loaders['train']), 4),
                "val_accuracy": round(accuracy_score(g, p), 4), "val_macro_f1": round(f1_score(g, p, average="macro"), 4)})
    print("validation:", log[-1], flush=True)

g, p = evaluate(loaders["test"])
result = {
    "base_model": BASE,
    "dataset": "nazimali/kurdish-news-headlines (CC BY 4.0)",
    "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
    "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": LR, "max_length": MAX_LEN, "device": device,
    "training_minutes": round((time.time() - started) / 60, 1),
    "test_accuracy": round(accuracy_score(g, p), 4),
    "test_macro_f1": round(f1_score(g, p, average="macro"), 4),
    "per_label": classification_report(g, p, target_names=labels, output_dict=True),
    "confusion_matrix": confusion_matrix(g, p).tolist(),
    "labels": labels,
}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results/finetune.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
(HERE / "results/training_log.json").write_text(json.dumps(log, indent=2))
model.save_pretrained(HERE / "model")
tok.save_pretrained(HERE / "model")
print("TEST:", {k: result[k] for k in ("test_accuracy", "test_macro_f1", "training_minutes")})
