import { useState } from 'react'

import ErrorNotice from '../../components/ErrorNotice'
import TrialCard from '../../components/TrialCard'
import { useMatch } from '../../services/API/BackendToFrontendAPI'

/**
 * A worked example, and the reason it is this long.
 *
 * Measured on 226 criteria across five real trials, changing nothing but the
 * input: a three-sentence vignette left 89% of criteria `unclear`, while one
 * carrying the facts below left 61% — decidable criteria went from 24 to 88.
 *
 * So the prompt beside the box is not hand-holding. It is the single largest
 * lever anyone has over the quality of the answer.
 */
const EXAMPLE = `Synthetic case, not a real patient. 58-year-old woman, ECOG 1, stage IV non-small cell lung adenocarcinoma with EGFR exon 19 deletion, diagnosed 14 months ago. Two prior lines: osimertinib, then carboplatin/pemetrexed; last dose 6 weeks ago. Measurable disease per RECIST 1.1 (liver, lung). No brain metastases on MRI 3 weeks ago. Labs: ANC 3.2 x10^9/L, platelets 210 x10^9/L, haemoglobin 11.4 g/dL, creatinine clearance 82 mL/min, bilirubin 0.7 mg/dL, AST 28 U/L, ALT 31 U/L. No HIV, no hepatitis B or C. No autoimmune disease, no interstitial lung disease. Post-menopausal. No other malignancy.`

/** Ranked by how many of the 11,195 trials actually test each one. */
const FIELDS = [
  ['Diagnosis, histology', '95%'],
  ['Comorbidities, prior malignancy', '87%'],
  ['Age', '83%'],
  ['Prior therapy lines and agents', '78%'],
  ['ECOG performance status', '68%'],
  ['Labs: ANC, platelets, creatinine, bilirubin', '62%'],
  ['Prior surgery or radiotherapy', '60%'],
  ['HIV / hepatitis status', '55%'],
  ['Time since last treatment', '46%'],
  ['Stage, biomarkers, measurable disease', '45%'],
]

function Progress() {
  return (
    <div className="border-line bg-raised rounded-lg border p-5">
      <div className="flex items-center gap-3">
        <span className="border-accent size-4 animate-spin rounded-full border-2 border-t-transparent" />
        <p className="text-ink-strong text-sm">Searching…</p>
      </div>
      {/* Cold, this takes 30-40 seconds: the embedding model loads, then every
          candidate trial costs one model call. A bare spinner for that long reads
          as broken, so the work is named instead. */}
      <ol className="text-ink mt-3 space-y-1 text-sm">
        <li>Embedding the description</li>
        <li>Ranking 351,745 criteria by similarity</li>
        <li>Judging each candidate trial, criterion by criterion</li>
      </ol>
      <p className="text-ink mt-3 text-xs">
        30–40 seconds cold, about 4 warm. Every candidate is a separate model call.
      </p>
    </div>
  )
}

export default function HomePage() {
  const [text, setText] = useState('')
  const [k, setK] = useState(10)
  const { data, isLoading, error, run, reset } = useMatch()

  const tooShort = text.trim().length < 10

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-ink-strong text-3xl tracking-tight">
          Find trials a patient could join
        </h1>
        <p className="mt-2 max-w-3xl">
          Describe the case in your own words. Every criterion of every candidate
          trial is judged separately, with a reason you can check.
        </p>
      </section>

      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault()
          if (!tooShort) void run({ text: text.trim(), k })
        }}
      >
        <div className="border-not-met bg-not-met-bg rounded-md border-l-2 px-4 py-2 text-sm">
          <strong className="text-ink-strong">Synthetic descriptions only.</strong>{' '}
          This text is sent to a third-party model provider. Never enter real
          patient data.
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          maxLength={4000}
          placeholder="Age, diagnosis and stage, ECOG, biomarkers, prior therapy and when it ended, key labs, relevant comorbidities…"
          className="border-line bg-surface text-ink-strong focus:border-accent-line w-full rounded-lg border p-4 text-sm outline-none"
        />

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={tooShort || isLoading}
            className="bg-accent rounded-md px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {isLoading ? 'Searching…' : 'Find trials'}
          </button>

          <label className="text-ink flex items-center gap-2 text-sm">
            Trials to judge
            <select
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="border-line bg-surface text-ink-strong rounded-md border px-2 py-1"
            >
              {[5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            onClick={() => setText(EXAMPLE)}
            className="text-accent text-sm underline underline-offset-2"
          >
            Use a worked example
          </button>

          {(data || error) && (
            <button
              type="button"
              onClick={() => {
                reset()
                setText('')
              }}
              className="text-ink hover:text-ink-strong text-sm"
            >
              Clear
            </button>
          )}
        </div>
      </form>

      {!data && !isLoading && !error && (
        <section className="border-line rounded-lg border p-5">
          <h2 className="text-ink-strong text-base">
            The more of these you give, the more it can decide
          </h2>
          <p className="mt-1 text-sm">
            Measured: a three-sentence description leaves 89% of criteria
            undecidable. One carrying the facts below leaves 61% — three times as
            many answers from the same trials.
          </p>
          <ul className="mt-3 grid gap-x-8 gap-y-1 text-sm sm:grid-cols-2">
            {FIELDS.map(([field, share]) => (
              <li key={field} className="flex justify-between gap-4">
                <span>{field}</span>
                <span className="text-ink font-mono text-xs">{share}</span>
              </li>
            ))}
          </ul>
          <p className="text-ink mt-3 text-xs">
            Percentages are the share of the 11,195 recruiting trials that test
            each one.
          </p>
        </section>
      )}

      {isLoading && <Progress />}

      {error && <ErrorNotice error={error} onRetry={() => void run({ text: text.trim(), k })} />}

      {data && (
        <section className="space-y-4">
          {/* Rendered from the payload, never hardcoded — a refactor here cannot
              drop it, because the server is what supplies it. */}
          <p className="border-line bg-raised text-ink rounded-md border p-4 text-sm">
            {data.disclaimer}
          </p>

          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-ink-strong text-xl">
              {data.trials.length} candidate trials
            </h2>
            <span className="text-ink font-mono text-xs">
              judged by {data.model}
            </span>
          </div>

          {data.trials.map((trial) => (
            <TrialCard key={trial.nct_id} trial={trial} />
          ))}

          {data.trials.length === 0 && (
            <p className="text-sm">
              Nothing matched. The corpus is recruiting, interventional oncology
              trials only — a case outside that scope will return nothing.
            </p>
          )}
        </section>
      )}
    </div>
  )
}
