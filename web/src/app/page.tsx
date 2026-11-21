import { CompareBars, Confusion, LossChart } from "@/components/charts"
import { Demo } from "@/components/demo"
import results from "@/data/results.json"
import { pct, topic } from "@/lib/topics"

const GITHUB = "https://github.com/gobynn/hawal"

function Section({ id, eyebrow, title, children }: { id: string; eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="border-t border-line py-16 sm:py-20">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-2">{eyebrow}</p>
      <h2 className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
      <div className="mt-8">{children}</div>
    </section>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-2xl border border-line bg-surface p-5">
      <p className="text-2xl font-semibold tracking-tight tabular-nums sm:text-3xl">{value}</p>
      <p className="mt-1 text-sm text-ink-2">{label}</p>
    </div>
  )
}

export default function Home() {
  const r = results
  const t = r.training
  const steps = [
    { n: "01", title: "Collect and clean the data", body: `Downloaded ${r.dataset.raw_rows.toLocaleString()} labelled Sorani headlines, unified Arabic and Kurdish letter forms (ي→ی, ك→ک), dropped very short lines and duplicates: ${r.dataset.clean_rows.toLocaleString()} remain.`, file: "01_prepare_data.py" },
    { n: "02", title: "Split it fairly", body: `Stratified 80 / 10 / 10 split with a fixed seed: ${r.dataset.splits.train.toLocaleString()} for training, ${r.dataset.splits.validation.toLocaleString()} for validation, ${r.dataset.splits.test.toLocaleString()} held out for the final test.`, file: "01_prepare_data.py" },
    { n: "03", title: "Build a baseline to beat", body: `A classical model (TF-IDF over character n-grams + logistic regression) reaches ${pct(r.baseline.test_accuracy)} on the test set.`, file: "02_baseline.py" },
    { n: "04", title: "Fine-tune KuBERT", body: `Added a 5-topic classification head to KuBERT and trained all layers for ${t.epochs} epochs (batch ${t.batch_size}, learning rate ${t.learning_rate}, max ${t.max_length} tokens, AdamW + linear warm-up) on an Apple M4 Pro GPU in ${t.training_minutes} minutes.`, file: "03_finetune.py" },
    { n: "05", title: "Shrink it for the web", body: `Exported to ONNX and quantized to 8-bit (${r.deployed.size_mb} MB). The quantized file scores ${pct(r.deployed.test_accuracy)} on the same test set, and runs in the browser with Transformers.js.`, file: "04_export_onnx.py" },
  ]

  return (
    <main className="mx-auto max-w-6xl px-5 sm:px-8">
      <header className="flex items-center justify-between py-6">
        <span className="flex items-baseline gap-2 text-lg font-semibold tracking-tight">
          Hawal <span className="font-kurdish text-base font-normal text-ink-2" dir="rtl">هەواڵ</span>
        </span>
        <nav className="flex gap-5 text-sm text-ink-2">
          <a href="#how" className="hover:text-ink">How</a>
          <a href="#results" className="hover:text-ink">Results</a>
          <a href="#data" className="hover:text-ink">Data</a>
          <a href={GITHUB} className="hover:text-ink">GitHub</a>
        </nav>
      </header>

      <section className="pb-16 pt-10 sm:pt-16">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-2">Fine-tuned language model · Central Kurdish</p>
        <h1 className="mt-3 max-w-4xl text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
          Reading Kurdish news the way a newsroom does.
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-ink-2">
          Hawal is KuBERT, a Central Kurdish BERT, fine-tuned on {r.dataset.clean_rows.toLocaleString()} Sorani headlines to tell
          economy, health, science, society and sport apart. It scores <strong className="text-ink">{pct(r.finetuned.test_accuracy)}</strong> on
          headlines it never saw, and it runs entirely in your browser.
        </p>
        <div className="mt-10"><Demo samples={r.samples} /></div>
      </section>

      <Section id="how" eyebrow="Method" title="From raw headlines to a model in your browser">
        <ol className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {steps.map(s => (
            <li key={s.n} className="flex flex-col rounded-2xl border border-line bg-surface p-6">
              <span className="text-sm tabular-nums text-ink-2">{s.n}</span>
              <h3 className="mt-2 text-lg font-semibold">{s.title}</h3>
              <p className="mt-2 flex-1 text-sm leading-relaxed text-ink-2">{s.body}</p>
              <a href={`${GITHUB}/blob/main/training/${s.file}`} className="mt-4 font-mono text-xs text-ink-2 underline-offset-4 hover:text-ink hover:underline">training/{s.file}</a>
            </li>
          ))}
        </ol>
      </Section>

      <Section id="results" eyebrow="Results" title={r.finetuned.test_accuracy > r.baseline.test_accuracy
        ? `Fine-tuning beats the classical baseline by ${((r.finetuned.test_accuracy - r.baseline.test_accuracy) * 100).toFixed(1)} points`
        : "How fine-tuning compares with a classical baseline"}>
        <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
          <Stat value={pct(r.finetuned.test_accuracy)} label="test accuracy, fine-tuned KuBERT" />
          <Stat value={pct(r.finetuned.test_macro_f1)} label="macro F1 across the 5 topics" />
          <Stat value={`${t.training_minutes} min`} label={`training time, ${t.epochs} epochs on Apple M4 Pro`} />
          <Stat value={`${r.deployed.size_mb} MB`} label="8-bit model shipped to the browser" />
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Same {r.dataset.splits.test.toLocaleString()} test headlines, three models</h3>
            <div className="mt-6">
              <CompareBars rows={[
                { name: "Baseline: TF-IDF + logistic regression", note: "Classical machine learning, no pretraining.", accuracy: r.baseline.test_accuracy, f1: r.baseline.test_macro_f1 },
                { name: "Fine-tuned KuBERT (full precision)", note: "The model after fine-tuning, in PyTorch.", accuracy: r.finetuned.test_accuracy, f1: r.finetuned.test_macro_f1, strong: true },
                { name: "Fine-tuned KuBERT, 8-bit (this website)", note: "Quantized for the browser. Measured with ONNX Runtime in Python; the browser's WebAssembly runtime can differ slightly on close calls.", accuracy: r.deployed.test_accuracy, f1: r.deployed.test_macro_f1 },
              ]} />
            </div>
          </div>
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Training loss</h3>
            <p className="mt-1 text-sm text-ink-2">Running average of the cross-entropy loss, logged every 100 steps.</p>
            <div className="mt-4"><LossChart log={r.log} /></div>
            <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-sm text-ink-2">
              {r.log.filter(l => "val_accuracy" in l).map(l => (
                <li key={l.epoch}>Epoch {l.epoch}: validation accuracy <span className="tabular-nums text-ink">{pct((l as { val_accuracy: number }).val_accuracy)}</span></li>
              ))}
            </ul>
          </div>
        </div>

        {r.experiments.length > 0 && (
          <div className="mt-6 rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Experiments: what the first run got wrong</h3>
            <ul className="mt-4 grid gap-4 md:grid-cols-2">
              {r.experiments.map(e => (
                <li key={e.name} className="rounded-2xl border border-line p-5">
                  <p className="font-medium">{e.name}</p>
                  <p className="mt-1 text-sm tabular-nums text-ink-2">{e.epochs} epochs · learning rate {e.learning_rate} · accuracy <span className="text-ink">{pct(e.test_accuracy)}</span> · macro F1 {pct(e.test_macro_f1)}</p>
                  <p className="mt-2 text-sm leading-relaxed text-ink-2">{e.note}</p>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_1.3fr]">
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Per topic</h3>
            <table className="mt-4 w-full text-sm">
              <thead><tr className="text-start text-xs text-ink-2"><th className="pb-2 text-start font-medium">Topic</th><th className="pb-2 text-end font-medium">Precision</th><th className="pb-2 text-end font-medium">Recall</th><th className="pb-2 text-end font-medium">F1</th></tr></thead>
              <tbody>
                {r.labels.map(l => {
                  const m = r.per_label[l as keyof typeof r.per_label]
                  return (
                    <tr key={l} className="border-t border-line">
                      <td className="py-2"><span className="me-2 inline-block size-2 rounded-full" style={{ background: topic(l).color }} />{topic(l).en}</td>
                      <td className="py-2 text-end tabular-nums">{pct(m.precision)}</td>
                      <td className="py-2 text-end tabular-nums">{pct(m.recall)}</td>
                      <td className="py-2 text-end tabular-nums font-medium">{pct(m["f1-score"])}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Confusion matrix (test set)</h3>
            <p className="mt-1 text-sm text-ink-2">Diagonal = correct. Off-diagonal cells show which topics the model mixes up.</p>
            <div className="mt-4"><Confusion labels={r.labels} matrix={r.confusion_matrix} /></div>
          </div>
        </div>
      </Section>

      <Section id="data" eyebrow="Data and credits" title="What it was trained on">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Dataset</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-2">
              <a className="text-ink underline underline-offset-4" href={r.dataset.url}>{r.dataset.name}</a> ({r.dataset.id}), {r.dataset.license}.
              Central Kurdish headlines from Kurdish news sites, each with a topic label and an English translation.
            </p>
            <ul className="mt-5 flex flex-col gap-2">
              {Object.entries(r.dataset.label_counts).map(([l, n]) => (
                <li key={l} className="flex items-center gap-3 text-sm">
                  <span className="w-32 shrink-0">{topic(l).en}</span>
                  <span className="h-2 flex-1 overflow-hidden rounded-full bg-line"><span className="block h-full rounded-full" style={{ width: `${(Number(n) / 10000) * 100}%`, background: topic(l).color }} /></span>
                  <span className="w-14 text-end tabular-nums text-ink-2">{Number(n).toLocaleString()}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-3xl border border-line bg-surface p-6">
            <h3 className="font-semibold">Base model and tools</h3>
            <ul className="mt-2 flex flex-col gap-2 text-sm leading-relaxed text-ink-2">
              <li><a className="text-ink underline underline-offset-4" href={r.base_model.url}>KuBERT</a> ({r.base_model.id}): a BERT model pretrained on Central Kurdish text by AsoSoft. Hawal starts from its weights and fine-tunes all of them.</li>
              <li>Its published tokenizer config pointed the unknown and padding tokens at tokens missing from the vocabulary, so training rebuilds a standard WordPiece tokenizer from its vocab file.</li>
              <li>PyTorch and Hugging Face Transformers for training, scikit-learn for the baseline and metrics, ONNX Runtime for export and quantization, Transformers.js for the browser.</li>
              <li>Every number on this page is read from the training scripts&apos; output files in <a className="text-ink underline underline-offset-4" href={`${GITHUB}/tree/main/training/results`}>training/results</a>.</li>
            </ul>
          </div>
        </div>
      </Section>

      <footer className="border-t border-line py-10 text-sm text-ink-2">
        Hawal · AI course project by Baxan Hamid · <a href={GITHUB} className="underline underline-offset-4 hover:text-ink">source and training code</a>
      </footer>
    </main>
  )
}
