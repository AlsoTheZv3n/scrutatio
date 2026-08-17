/**
 * What this is, and — at greater length — what it is not.
 *
 * A medical decision-support tool that does not state its limits is a liability.
 * Every number here is measured; nothing on this page is an estimate, and the
 * uncomfortable ones are first rather than last.
 */

function Limit({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-line border-l-2 pl-4">
      <h3 className="text-ink-strong text-base">{title}</h3>
      <div className="mt-1 space-y-2 text-sm">{children}</div>
    </div>
  )
}

export default function AboutPage() {
  return (
    <div className="max-w-3xl space-y-10">
      <section>
        <h1 className="text-ink-strong text-3xl tracking-tight">Limits</h1>
        <p className="mt-2">
          Read this before trusting anything the tool returns.
        </p>
      </section>

      <div className="border-not-met bg-not-met-bg rounded-md border-l-2 p-4 text-sm">
        <strong className="text-ink-strong">
          Decision support, not medical advice.
        </strong>{' '}
        This is a prioritised list of candidate trials with per-criterion evidence
        for review by a qualified healthcare professional — not an eligibility
        determination. Eligibility must always be verified with the study team.
        Never enter real patient data.
      </div>

      <section className="space-y-6">
        <Limit title="Retrieval quality has not been evaluated">
          <p>
            The system has never been measured against a labelled benchmark. It
            has not been shown to be better than the keyword search that
            ClinicalTrials.gov already offers for free, and that comparison is the
            project's own definition of done.
          </p>
          <p className="text-ink">
            An evaluation against TREC Clinical Trials 2021 is in progress. Until
            it reports, every ranking here is plausible rather than proven.
          </p>
        </Limit>

        <Limit title="Most criteria cannot be decided from a description">
          <p>
            Measured over 226 criteria in five real trials: a three-sentence
            description left <strong>89%</strong> of criteria undecidable. A
            description carrying age, stage, biomarkers, prior therapy, labs and
            infection status left <strong>61%</strong>.
          </p>
          <p>
            Some of that floor is permanent. Consent, protocol adherence and
            contraception are <strong>11.9%</strong> of all criteria and no patient
            description can ever answer them — they are collapsed by default in the
            results for that reason.
          </p>
          <p className="text-ink">
            An <em>unclear</em> verdict is the honest answer, not a failure. It
            marks the criterion a human should check.
          </p>
        </Limit>

        <Limit title="The corpus is narrow">
          <p>
            Currently recruiting, interventional, oncology trials only — about
            11,200 of the roughly 560,000 studies on ClinicalTrials.gov. A case
            outside that scope returns nothing, and that is not a bug.
          </p>
          <p>
            Trial data is a snapshot, not a live feed. Status and eligibility can
            have changed since it was taken.
          </p>
        </Limit>

        <Limit title="Extraction is imperfect">
          <p>
            Eligibility prose is split into predicates by a language model.{' '}
            <strong>4.06%</strong> of criteria land in a catch-all category and are
            effectively unmatchable. Five trials failed extraction entirely and are
            absent from search.
          </p>
          <p className="text-ink">
            The Corpus page shows all of this directly, including the catch-all
            bucket.
          </p>
        </Limit>

        <Limit title="Judgements come from a language model">
          <p>
            Each verdict and its reason are generated per criterion. They are not
            checked against a clinician, and the model can be confidently wrong. The
            reason is shown for every verdict precisely so it can be checked rather
            than trusted.
          </p>
        </Limit>
      </section>

      <section>
        <h2 className="text-ink-strong text-xl">How it works</h2>
        <ol className="mt-3 space-y-2 text-sm">
          <li>
            <strong className="text-ink-strong">1. Extract.</strong> Every trial's
            eligibility prose is split into single testable predicates, each typed
            into one of twenty categories and marked inclusion or exclusion.
          </li>
          <li>
            <strong className="text-ink-strong">2. Embed.</strong> Each criterion
            becomes a vector, so a description can match on meaning rather than
            shared words — a query about poor kidney function finds criteria phrased
            as impaired renal function.
          </li>
          <li>
            <strong className="text-ink-strong">3. Judge.</strong> Every criterion of
            every shortlisted trial is judged against the description separately,
            with a stated reason. A trial can rank high on one criterion and still be
            excluded by another, which is why the ranking alone is not the answer.
          </li>
        </ol>
      </section>
    </div>
  )
}
