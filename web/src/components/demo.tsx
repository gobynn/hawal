"use client"
// Live demo: the fine-tuned model (8-bit ONNX, in /public/models/hawal) runs inside the visitor's browser
// with Transformers.js. Nothing is sent to a server.
import { useEffect, useRef, useState } from "react"
import { normalize, pct, topic } from "@/lib/topics"

type Score = { label: string; score: number }
type Classifier = (text: string, options: { top_k: number | null }) => Promise<Score[]>
type Sample = { text: string; english: string; label: string }

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2"

let loading: Promise<Classifier> | null = null
function loadModel(): Promise<Classifier> {
  loading ??= (async () => {
    const mod = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ TRANSFORMERS_URL)
    mod.env.allowLocalModels = true
    mod.env.allowRemoteModels = false
    mod.env.localModelPath = "/models/"
    return (await mod.pipeline("text-classification", "hawal", { dtype: "q8" })) as Classifier
  })()
  loading.catch(() => { loading = null })
  return loading
}

export function Demo({ samples }: { samples: Sample[] }) {
  const [text, setText] = useState(samples[0]?.text ?? "")
  const [status, setStatus] = useState<"loading" | "ready" | "running" | "error">("loading")
  const [scores, setScores] = useState<Score[] | null>(null)
  const [ms, setMs] = useState<number | null>(null)
  const [expected, setExpected] = useState<string | null>(samples[0]?.label ?? null)
  const [english, setEnglish] = useState<string | null>(samples[0]?.english ?? null)
  const model = useRef<Classifier | null>(null)

  async function run(value: string) {
    const input = normalize(value)
    if (!input || !model.current) return
    setStatus("running")
    const started = performance.now()
    const out = await model.current(input, { top_k: null })
    setMs(Math.round(performance.now() - started))
    setScores([...out].sort((a, b) => b.score - a.score))
    setStatus("ready")
  }

  useEffect(() => {
    let alive = true
    loadModel()
      .then(clf => {
        if (!alive) return
        model.current = clf
        setStatus("ready")
        void run(text)
      })
      .catch(() => alive && setStatus("error"))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pick = (s: Sample) => {
    setText(s.text); setExpected(s.label); setEnglish(s.english)
    void run(s.text)
  }

  const top = scores?.[0]
  return (
    <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
      <div className="flex flex-col gap-4">
        <label htmlFor="headline" className="text-sm font-medium text-ink-2">Kurdish (Sorani) headline</label>
        <textarea
          id="headline"
          dir="rtl"
          lang="ckb"
          rows={3}
          value={text}
          onChange={e => { setText(e.target.value); setExpected(null); setEnglish(null) }}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void run(text) } }}
          className="font-kurdish w-full resize-none rounded-2xl border border-line bg-surface px-5 py-4 text-2xl leading-relaxed text-ink outline-none transition-colors focus:border-ink"
          placeholder="سەردێڕێکی هەواڵ بنووسە…"
        />
        {english && <p className="-mt-1 text-sm text-ink-2">English: {english}</p>}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => void run(text)}
            disabled={status !== "ready" || !text.trim()}
            className="rounded-full bg-ink px-6 py-3 text-sm font-semibold text-canvas transition-opacity disabled:opacity-40"
          >
            {status === "loading" ? "Loading model…" : status === "running" ? "Reading…" : "Classify"}
          </button>
          <span className="text-xs text-ink-2">
            {status === "loading" && "Downloading the 8-bit model once (about 80 MB), then it runs offline in your browser."}
            {status === "error" && "The model could not load in this browser. Try a recent Chrome, Safari or Edge."}
            {ms !== null && status === "ready" && `Ran in ${ms} ms on this device. Nothing was sent to a server.`}
          </span>
        </div>
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-2">Try a headline from the test set (never seen in training)</p>
          <div className="flex flex-wrap gap-2">
            {samples.map(s => (
              <button
                key={s.text}
                type="button"
                onClick={() => pick(s)}
                dir="rtl"
                className="font-kurdish max-w-full truncate rounded-full border border-line bg-surface px-4 py-2 text-base text-ink transition-colors hover:border-ink"
                title={s.english}
              >
                {s.text}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-3xl border border-line bg-surface p-6" aria-live="polite">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-2">Model prediction</p>
        {top ? (
          <>
            <p className="mt-2 text-4xl font-semibold tracking-tight" style={{ color: topic(top.label).color }}>{topic(top.label).en}</p>
            <p className="font-kurdish mt-1 text-xl text-ink-2" dir="rtl">{topic(top.label).ku}</p>
            <p className="mt-1 text-sm text-ink-2">
              {pct(top.score)} confident
              {expected && (expected === top.label ? " · matches the dataset's label" : ` · dataset label: ${topic(expected).en}`)}
            </p>
            <ul className="mt-6 flex flex-col gap-3">
              {scores!.map(s => (
                <li key={s.label}>
                  <div className="mb-1 flex justify-between text-sm"><span>{topic(s.label).en}</span><span className="tabular-nums text-ink-2">{pct(s.score)}</span></div>
                  <div className="h-2 overflow-hidden rounded-full bg-line">
                    <div className="h-full rounded-full transition-[width] duration-500 ease-out" style={{ width: `${Math.max(1, s.score * 100)}%`, background: topic(s.label).color }} />
                  </div>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <div className="mt-4 flex flex-col gap-3" aria-busy="true">
            {[0, 1, 2, 3, 4].map(i => <div key={i} className="h-6 animate-pulse rounded-full bg-line" />)}
          </div>
        )}
      </div>
    </div>
  )
}
