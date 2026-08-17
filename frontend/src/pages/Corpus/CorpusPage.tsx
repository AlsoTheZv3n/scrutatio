import { useState } from 'react'

import ErrorNotice from '../../components/ErrorNotice'
import {
  useCorpusStats,
  useCriteria,
} from '../../services/API/BackendToFrontendAPI'
import {
  CRITERION_KINDS,
  trialUrl,
  type CriterionKind,
} from '../../services/API/types'

const PAGE = 50

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="border-line rounded-lg border p-4">
      <div className="text-ink text-xs">{label}</div>
      <div className="text-ink-strong mt-1 font-mono text-xl">{value}</div>
      {note && <div className="text-ink mt-0.5 text-xs">{note}</div>}
    </div>
  )
}

/**
 * The honesty surface.
 *
 * Costs nothing per visitor — no model call, no embedding — and it is the only
 * place where what the system *knows* is inspectable rather than inferred from a
 * ranking. Browsing the `other` bucket is the fastest way to see where extraction
 * is weak, which is a thing a demo would hide and a tool should not.
 */
export default function CorpusPage() {
  const stats = useCorpusStats()
  const [kind, setKind] = useState<CriterionKind | ''>('')
  const [exclusion, setExclusion] = useState<'' | 'true' | 'false'>('')
  const [offset, setOffset] = useState(0)

  const page = useCriteria({
    ...(kind ? { kind } : {}),
    ...(exclusion ? { isExclusion: exclusion === 'true' } : {}),
    offset,
    limit: PAGE,
  })

  const fmt = (n: number) => n.toLocaleString('en-US')

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-ink-strong text-3xl tracking-tight">What it knows</h1>
        <p className="mt-2 max-w-3xl">
          ClinicalTrials.gov publishes eligibility as prose in a single field. This
          is that prose split into individually checkable predicates, each typed
          and marked as inclusion or exclusion.
        </p>
      </section>

      {stats.error && <ErrorNotice error={stats.error} onRetry={stats.reload} />}

      {stats.data && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Trials" value={fmt(stats.data.bronze_trials)} note="recruiting, interventional, oncology" />
            <Stat label="Criteria extracted" value={fmt(stats.data.silver_criteria)} note={`across ${fmt(stats.data.silver_trials)} trials`} />
            <Stat label="Vectors" value={fmt(stats.data.gold_vectors)} note={stats.data.embedding_model} />
            <Stat
              label="Failed extraction"
              value={fmt(stats.data.silver_failed)}
              note={
                stats.data.silver_criteria > 0
                  ? `${((stats.data.silver_failed / (stats.data.silver_trials + stats.data.silver_failed)) * 100).toFixed(2)}% of trials`
                  : undefined
              }
            />
          </section>

          <p className="text-ink font-mono text-xs">
            signature {stats.data.signature} · newest trial update{' '}
            {stats.data.newest_update ?? 'unknown'} · {stats.data.database}
          </p>
        </>
      )}

      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={kind}
            onChange={(e) => {
              setKind(e.target.value as CriterionKind | '')
              setOffset(0)
            }}
            className="border-line bg-surface text-ink-strong rounded-md border px-3 py-1.5 text-sm"
          >
            <option value="">All categories</option>
            {CRITERION_KINDS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>

          <select
            value={exclusion}
            onChange={(e) => {
              setExclusion(e.target.value as '' | 'true' | 'false')
              setOffset(0)
            }}
            className="border-line bg-surface text-ink-strong rounded-md border px-3 py-1.5 text-sm"
          >
            <option value="">Inclusion and exclusion</option>
            <option value="false">Inclusion only</option>
            <option value="true">Exclusion only</option>
          </select>

          {page.data && (
            <span className="text-ink text-sm">
              {fmt(page.data.total)} matching
            </span>
          )}
        </div>

        {page.error && <ErrorNotice error={page.error} />}

        {page.isLoading && <p className="text-ink text-sm">Loading…</p>}

        {page.data && (
          <>
            <ul className="border-line divide-line divide-y rounded-lg border">
              {page.data.rows.map((row) => (
                <li key={`${row.nct_id}-${row.ordinal}`} className="p-4">
                  <p className="text-ink-strong text-sm">
                    {row.is_exclusion && (
                      <span className="text-not-met mr-1.5 text-xs font-medium uppercase">
                        exclusion
                      </span>
                    )}
                    {row.text}
                  </p>
                  <p className="text-ink mt-1 font-mono text-xs">
                    <a
                      href={trialUrl(row.nct_id)}
                      target="_blank"
                      rel="noreferrer"
                      className="text-accent underline underline-offset-2"
                    >
                      {row.nct_id}
                    </a>{' '}
                    #{row.ordinal} · {row.kind}
                  </p>
                </li>
              ))}
            </ul>

            <div className="flex items-center justify-between gap-4">
              <button
                type="button"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
                className="border-line text-ink hover:text-ink-strong rounded-md border px-3 py-1.5 text-sm disabled:opacity-40"
              >
                Previous
              </button>
              <span className="text-ink font-mono text-xs">
                {fmt(offset + 1)}–{fmt(Math.min(offset + PAGE, page.data.total))} of{' '}
                {fmt(page.data.total)}
              </span>
              <button
                type="button"
                disabled={offset + PAGE >= page.data.total}
                onClick={() => setOffset(offset + PAGE)}
                className="border-line text-ink hover:text-ink-strong rounded-md border px-3 py-1.5 text-sm disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
