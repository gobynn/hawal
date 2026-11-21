# Hawal (هەواڵ): a fine-tuned Kurdish news topic classifier

Hawal fine-tunes **KuBERT**, a BERT model pretrained on Central Kurdish, to sort Sorani news headlines into
five topics: economy, health, science & technology, society and sport. The website runs the fine-tuned model
in the visitor's browser (8-bit ONNX with Transformers.js), so no server or API key is involved.

- **Live site:** see the deployment link in the repository description
- **Dataset:** [nazimali/kurdish-news-headlines](https://huggingface.co/datasets/nazimali/kurdish-news-headlines) (CC BY 4.0), 50,000 labelled headlines
- **Base model:** [asosoft/KuBERT-Central-Kurdish-BERT-Model](https://huggingface.co/asosoft/KuBERT-Central-Kurdish-BERT-Model)

## Repository

```
training/
  01_prepare_data.py   download, normalise Kurdish letter forms, dedupe, stratified 80/10/10 split
  02_baseline.py       TF-IDF (character n-grams) + logistic regression baseline
  03_finetune.py       fine-tune KuBERT (PyTorch, Apple MPS / CUDA / CPU)
  04_export_onnx.py    export to ONNX, 8-bit dynamic quantization, re-test the quantized file
  05_site_data.py      collect results for the website
  results/             every metric the website shows (JSON), plus run 1 kept for comparison
web/                   Next.js site; the model lives in web/public/models/hawal
```

## Reproduce

```sh
cd training
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers datasets accelerate scikit-learn "optimum[onnxruntime]" onnx onnxruntime pandas
.venv/bin/python 01_prepare_data.py
.venv/bin/python 02_baseline.py
.venv/bin/python 03_finetune.py      # ~9 minutes on an Apple M4 Pro
.venv/bin/python 04_export_onnx.py
.venv/bin/python 05_site_data.py

cd ../web && pnpm install && pnpm dev
```

## What I found along the way

KuBERT's published tokenizer config is broken: its unknown and padding tokens point at tokens that are not in
its vocabulary, and the first vocabulary line is `[PAD` without the closing bracket. `03_finetune.py` rebuilds a
standard WordPiece tokenizer from the vocabulary file.

The first fine-tuning run (kept in `training/results/run1/`) scored below the baseline. The cause was BERT's
default accent stripping: it decomposes `ئ` and drops the hamza, so 8.1% of all tokens became `[UNK]`. Turning
accent stripping off and normalising `ى`→`ی` and `ھ`→`ه` brought unknown tokens down to 4.6%, mostly punctuation
and digits that KuBERT's vocabulary does not contain.

## Credits

Dataset by nazimali on Hugging Face (CC BY 4.0). KuBERT by AsoSoft. Built with PyTorch, Hugging Face
Transformers, scikit-learn, ONNX Runtime, Transformers.js and Next.js.
