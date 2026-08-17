import { useState } from 'react'

import {
  ADMINISTRATIVE_KINDS,
  trialUrl,
  type CriterionVerdict,
  type TrialMatch,
  type Verdict,
} from '../services/API/types'

/** Colour is the accent; the glyph and the word carry the meaning. */
const STYLE: Record<Verdict, { chip: string; glyph: string; word: string }> = {
  not_met: { chip: 'bg-not-met-bg text-not-met', glyph: '×', word: 'not met' },
  met: { chip: 'bg-met-bg text-met', glyph: '✓', word: 'met' },
  unclear: { chip: 'bg-unclear-bg text-unclear', glyph: '?', word: 'unclear' },
}

/**
 * Decision order, not source order.
 *
 * `not_met` first because an exclusion is the fastest way to rule a trial out —
 * one line can end the review. `met` next as the evidence for. `unclear` last,
 * because it is the work list rather than the answer.
 */
const RANK: Record<Verdict, number> = { not_met: 0, met: 1, unclear: 2 }

function byDecisionValue(a: CriterionVerdict, b: CriterionVerdict) {
  return RANK[a.verdict] - RANK[b.verdict] || a.ordinal - b.ordinal
}

function VerdictBar({ counts }: { counts: TrialMatch['counts'] }) {
  const total = counts.met + counts.not_met + counts.unclear
  if (total === 0) return null

  const segments: Array<[Verdict, number]> = [
    ['not_met', counts.not_met],
    ['met', counts.met],
    ['unclear', counts.unclear],
  ]

  return (
    <div className="flex items-center gap-3">
      <div
        className="bg-line flex h-1.5 w-32 overflow-hidden rounded-full"
        role="img"
        aria-label={`${counts.not_met} not met, ${counts.met} met, ${counts.unclear} unclear`}
      >
        {segments.map(([verdict, n]) =>
          n > 0 ? (
            <span
              key={verdict}
              style={{ width: `${(n / total) * 100}%` }}
              className={
                verdict === 'not_met'
                  ? 'bg-not-met'
                  : verdict === 'met'
                    ? 'bg-met'
                    : 'bg-unclear'
              }
            />
          ) : null,
        )}
      </div>
      <span className="font-mono text-xs">
        <span className="text-not-met">{counts.not_met}×</span>{' '}
        <span className="text-met">{counts.met}✓</span>{' '}
        <span className="text-unclear">{counts.unclear}?</span>
      </span>
    </div>
  )
}

function CriterionRow({ criterion }: { criterion: CriterionVerdict }) {
  const style = STYLE[criterion.verdict]
  return (
    <li className="border-line grid grid-cols-[auto_1fr] gap-x-3 border-t py-3 first:border-t-0">
      <span
        className={`${style.chip} mt-0.5 flex size-6 items-center justify-center rounded-md text-sm font-medium`}
        title={style.word}
        aria-hidden
      >
        {style.glyph}
      </span>
      <div className="min-w-0">
        <p className="text-ink-strong text-sm">
          {criterion.is_exclusion && (
            <span className="text-not-met mr-1.5 text-xs font-medium uppercase">
              exclusion
            </span>
          )}
          {criterion.text}
        </p>
        {/* The rationale is the product. A verdict without one is a badge asking
            to be trusted, which is exactly what this tool must not do. */}
        <p className="text-ink mt-1 text-sm italic">{criterion.rationale}</p>
        <span className="text-ink mt-1 inline-block font-mono text-xs">
          #{criterion.ordinal} · {criterion.kind} · {style.word}
        </span>
      </div>
    </li>
  )
}

export default function TrialCard({ trial }: { trial: TrialMatch }) {
  const [open, setOpen] = useState(false)
  const [showAdmin, setShowAdmin] = useState(false)

  const admin = new Set<string>(ADMINISTRATIVE_KINDS)
  const clinical = trial.criteria.filter((c) => !admin.has(c.kind))
  const administrative = trial.criteria.filter((c) => admin.has(c.kind))

  return (
    <article className="border-line bg-surface rounded-lg border">
      <div className="p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <a
            href={trialUrl(trial.nct_id)}
            target="_blank"
            rel="noreferrer"
            className="text-accent font-mono text-sm underline underline-offset-2"
          >
            {trial.nct_id}
          </a>
          <span className="text-ink text-xs">
            {trial.overall_status.toLowerCase().replace(/_/g, ' ')}
            {trial.phase && ` · ${trial.phase.toLowerCase().replace(/_/g, ' ')}`}
          </span>
        </div>

        <h3 className="mt-1 text-base leading-snug">{trial.title}</h3>

        {/* Why this trial surfaced at all. Without it the ranking is a number
            asking to be believed. */}
        <p className="border-accent-line text-ink mt-3 border-l-2 pl-3 text-sm">
          matched on #{trial.matched_on.ordinal} ({trial.matched_on.score.toFixed(3)}):{' '}
          <span className="text-ink-strong">{trial.matched_on.text}</span>
        </p>

        {trial.locations.length > 0 && (
          <p className="text-ink mt-2 text-xs">
            {trial.locations.slice(0, 3).join(' · ')}
            {trial.locations.length > 3 && ` · +${trial.locations.length - 3} more`}
          </p>
        )}

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <VerdictBar counts={trial.counts} />
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="border-line text-ink hover:text-ink-strong hover:bg-raised rounded-md border px-3 py-1.5 text-sm transition-colors"
            aria-expanded={open}
          >
            {open ? 'Hide' : `Show ${trial.criteria.length} criteria`}
          </button>
        </div>
      </div>

      {open && (
        <div className="border-line bg-raised border-t px-5 py-3">
          <ul>
            {[...clinical].sort(byDecisionValue).map((c) => (
              <CriterionRow key={c.ordinal} criterion={c} />
            ))}
          </ul>

          {administrative.length > 0 && (
            <div className="border-line mt-1 border-t pt-3">
              <button
                type="button"
                onClick={() => setShowAdmin((v) => !v)}
                className="text-ink hover:text-ink-strong text-sm"
                aria-expanded={showAdmin}
              >
                {showAdmin ? 'Hide' : 'Show'} {administrative.length} administrative
                criteria
                {/* Measured: consent, compliance and contraception came back 100%
                    unclear, because no patient description states them. Shown by
                    default they bury the criteria that can actually be decided. */}
                <span className="text-ink ml-1 text-xs">
                  (consent, adherence, contraception — a description cannot answer
                  these)
                </span>
              </button>
              {showAdmin && (
                <ul className="mt-2">
                  {[...administrative].sort(byDecisionValue).map((c) => (
                    <CriterionRow key={c.ordinal} criterion={c} />
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </article>
  )
}
