import {
  ScrutatioApiError,
  ScrutatioNetworkError,
} from '../services/API/requests'

/**
 * One error box is not enough.
 *
 * The backend distinguishes eight failure codes, and they call for different
 * things from the reader. "A pipeline run holds the database" means wait thirty
 * seconds. "The index has not been built" means run a command. "The model
 * provider failed" means press the button again. Collapsing all of them into
 * *Something went wrong* throws away work the backend already did — which is why
 * `code` exists and why nothing here branches on `detail`.
 */
const GUIDANCE: Record<string, { title: string; hint: string; retry: boolean }> = {
  database_busy: {
    title: 'The corpus is being updated',
    hint: 'A pipeline run holds the database exclusively — DuckDB allows one process at a time. This usually clears within a minute.',
    retry: true,
  },
  gold_empty: {
    title: 'The search index has not been built',
    hint: 'Criteria have been extracted but not embedded yet. On the server: `uv run scrutatio embed`.',
    retry: false,
  },
  openrouter_unconfigured: {
    title: 'The server has no model credentials',
    hint: 'OPENROUTER_API_KEY is unset, so candidate trials cannot be judged. Retrieval alone would return a ranking with no reasoning, which is not what this tool is for.',
    retry: false,
  },
  encoder_unavailable: {
    title: 'The embedding model is missing',
    hint: 'The server was installed without the optional embeddings extra.',
    retry: false,
  },
  upstream_failed: {
    title: 'The model provider failed',
    hint: 'The fault is upstream, not here. Retrying often works.',
    retry: true,
  },
  storage_error: {
    title: 'The database returned an error',
    hint: 'Not the usual lock. The request id below will appear in the server log.',
    retry: false,
  },
  validation_error: {
    title: 'That input was rejected',
    hint: '',
    retry: false,
  },
  internal_error: {
    title: 'Something failed on the server',
    hint: 'The message is deliberately vague — an unmapped failure can carry internals. Quote the request id.',
    retry: true,
  },
}

interface Props {
  error: ScrutatioApiError | ScrutatioNetworkError
  onRetry?: () => void
}

export default function ErrorNotice({ error, onRetry }: Props) {
  const isApi = error instanceof ScrutatioApiError
  const guidance = isApi ? GUIDANCE[error.code] : undefined

  const title = guidance?.title ?? (isApi ? 'The request failed' : 'Cannot reach the API')
  const hint = guidance?.hint ?? ''
  const canRetry = onRetry && (guidance?.retry ?? true)

  return (
    <div
      role="alert"
      className="border-line bg-raised rounded-lg border p-5"
    >
      <h2 className="text-ink-strong text-base font-medium">{title}</h2>
      <p className="mt-2 text-sm">{error.message}</p>
      {hint && <p className="text-ink mt-2 text-sm">{hint}</p>}

      <div className="mt-4 flex items-center gap-4">
        {canRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="border-accent-line text-accent hover:bg-accent-bg rounded-md border px-3 py-1.5 text-sm transition-colors"
          >
            Try again
          </button>
        )}
        {isApi && error.requestId && (
          <span className="text-ink font-mono text-xs">
            request {error.requestId}
          </span>
        )}
      </div>
    </div>
  )
}
