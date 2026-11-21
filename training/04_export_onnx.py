"""Step 4: export the fine-tuned model to ONNX, quantize it to 8-bit so it runs in the browser, and check the
quantized model's accuracy on the same test set. Output: ../web/public/models/hawal/ and results/onnx_int8.json
"""
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

HERE = Path(__file__).parent
SRC = HERE / "model"
TMP = HERE / "onnx-fp32"
OUT = HERE.parent / "web/public/models/hawal"

ORTModelForSequenceClassification.from_pretrained(SRC, export=True).save_pretrained(TMP)
(OUT / "onnx").mkdir(parents=True, exist_ok=True)
quantize_dynamic(TMP / "model.onnx", OUT / "onnx/model_quantized.onnx", weight_type=QuantType.QInt8)

tok = AutoTokenizer.from_pretrained(SRC)
tok.save_pretrained(OUT)
shutil.copy(SRC / "config.json", OUT / "config.json")
for extra in ("vocab.txt", "added_tokens.json"):
    (OUT / extra).unlink(missing_ok=True)

# Accuracy of the exact file the website runs.
test = pd.read_csv(HERE / "data/test.csv")
session = ort.InferenceSession(str(OUT / "onnx/model_quantized.onnx"))
names = {i.name for i in session.get_inputs()}
preds = []
for start in range(0, len(test), 256):
    enc = tok(list(test["text"][start:start + 256]), truncation=True, max_length=48, padding=True, return_tensors="np")
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
    preds += session.run(None, feed)[0].argmax(-1).tolist()
result = {
    "model": "fine-tuned KuBERT, ONNX, dynamic 8-bit quantization (the file the website runs)",
    "size_mb": round((OUT / "onnx/model_quantized.onnx").stat().st_size / 1e6, 1),
    "test_accuracy": round(accuracy_score(test["label_id"], preds), 4),
    "test_macro_f1": round(f1_score(test["label_id"], preds, average="macro"), 4),
}
(HERE / "results/onnx_int8.json").write_text(json.dumps(result, indent=2))
print(result)
