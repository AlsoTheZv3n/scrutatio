export default function Footer() {
  return (
    <footer className="border-line text-ink mt-16 border-t px-6 py-8 text-xs">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4">
        <p>
          Decision support, not medical advice. Trial data from{' '}
          <a
            href="https://clinicaltrials.gov"
            target="_blank"
            rel="noreferrer"
            className="text-accent underline underline-offset-2"
          >
            ClinicalTrials.gov
          </a>
          . Never enter real patient data.
        </p>
        {/* Stated in the chrome, not only on the About page: a limit nobody
            scrolls to is a limit nobody reads. */}
        <p className="text-ink">Retrieval quality is not yet evaluated.</p>
      </div>
    </footer>
  )
}
