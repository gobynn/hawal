// Training charts drawn as plain SVG from results.json (no chart library).
import { pct, topic } from "@/lib/topics"

type LogRow = { step: number; epoch: number; train_loss: number; val_accuracy?: number }

export function LossChart({ log }: { log: LogRow[] }) {
  const points = log.filter(r => r.val_accuracy === undefined)
  if (points.length < 2) return null
  const w = 560, h = 200, pad = 32
  const maxStep = points.at(-1)!.step
  const maxLoss = Math.max(...points.map(p => p.train_loss))
  const minLoss = Math.min(...points.map(p => p.train_loss))
  const x = (s: number) => pad + (s / maxStep) * (w - pad * 2)
  const y = (l: number) => pad / 2 + ((maxLoss - l) / (maxLoss - minLoss || 1)) * (h - pad * 1.5)
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.train_loss).toFixed(1)}`).join(" ")
  const epochEnd = log.find(r => r.val_accuracy !== undefined && r.epoch === 1)?.step
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Training loss falling over the training steps">
      <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} stroke="var(--color-line)" />
      {epochEnd && <line x1={x(epochEnd)} x2={x(epochEnd)} y1={8} y2={h - pad} stroke="var(--color-line)" strokeDasharray="4 4" />}
      {epochEnd && <text x={x(epochEnd) + 6} y={20} fontSize="11" fill="var(--color-ink-2)">end of epoch 1</text>}
      <path d={d} fill="none" stroke="var(--color-ink)" strokeWidth="2" strokeLinejoin="round" />
      <text x={pad} y={h - 10} fontSize="11" fill="var(--color-ink-2)">step 0</text>
      <text x={w - pad} y={h - 10} fontSize="11" fill="var(--color-ink-2)" textAnchor="end">step {maxStep}</text>
      <text x={pad + 4} y={y(maxLoss) + 4} fontSize="11" fill="var(--color-ink-2)">loss {maxLoss.toFixed(2)}</text>
      <text x={w - pad} y={y(minLoss) - 8} fontSize="11" fill="var(--color-ink-2)" textAnchor="end">{minLoss.toFixed(2)}</text>
    </svg>
  )
}

export function Confusion({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(...matrix.flat())
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="p-2 text-start text-xs font-medium text-ink-2">true ↓ / predicted →</th>
            {labels.map(l => <th key={l} className="p-2 text-xs font-medium text-ink-2">{topic(l).en}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="p-2 text-start text-xs font-medium text-ink-2">{topic(labels[i]).en}</th>
              {row.map((n, j) => {
                const share = n / max
                return (
                  <td key={j} className="rounded-lg p-2 text-center tabular-nums"
                    style={{ background: i === j ? `color-mix(in srgb, ${topic(labels[i]).color} ${Math.round(20 + share * 70)}%, transparent)` : n ? `color-mix(in srgb, var(--color-ink) ${Math.round(4 + share * 60)}%, transparent)` : "transparent", color: i === j && share > 0.5 ? "#fff" : undefined }}>
                    {n}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function CompareBars({ rows }: { rows: { name: string; note: string; accuracy: number; f1: number; strong?: boolean }[] }) {
  return (
    <ul className="flex flex-col gap-5">
      {rows.map(r => (
        <li key={r.name}>
          <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
            <span className={r.strong ? "font-semibold" : ""}>{r.name}</span>
            <span className="tabular-nums text-sm text-ink-2">accuracy {pct(r.accuracy)} · macro F1 {pct(r.f1)}</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-line">
            <div className="h-full rounded-full" style={{ width: pct(r.accuracy), background: r.strong ? "var(--color-ink)" : "var(--color-ink-2)" }} />
          </div>
          <p className="mt-1 text-xs text-ink-2">{r.note}</p>
        </li>
      ))}
    </ul>
  )
}
