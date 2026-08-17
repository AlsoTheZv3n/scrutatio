/**
 * React bindings over `requests.ts`.
 *
 * The previous version was a generic `useFetch` from a tutorial. It had three
 * faults that would have surfaced as confusing UI rather than as errors:
 *
 * 1. It attached `body: JSON.stringify(input)` to **every** request, including
 *    GETs. `fetch` rejects a GET with a body outright.
 * 2. It never checked `response.ok`, so a 503 "a pipeline run holds the database"
 *    was parsed as JSON and handed to the component as `data`. The UI would have
 *    rendered an error object as a result.
 * 3. It had no error state at all, so there was nothing to render instead.
 *
 * The hooks below own loading and error state and hand back typed data. Every
 * request is abortable, because `/match` runs 30-40 seconds and a user who
 * navigates away should not have a late response overwrite state on an unmounted
 * component.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  getCorpusStats,
  getCriteria,
  getHealth,
  postMatch,
  ScrutatioApiError,
  ScrutatioNetworkError,
  type CriteriaQuery,
} from "./requests";
import type {
  CriterionPage,
  Health,
  LayerCounts,
  MatchRequest,
  MatchResponse,
} from "./types";

/** What every hook exposes. `error` is the backend's, already typed. */
export interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  error: ScrutatioApiError | ScrutatioNetworkError | null;
}

const IDLE = { data: null, isLoading: false, error: null } as const;

function asKnownError(cause: unknown): ScrutatioApiError | ScrutatioNetworkError {
  if (cause instanceof ScrutatioApiError || cause instanceof ScrutatioNetworkError) {
    return cause;
  }
  // Anything else is a bug in this layer rather than a backend condition, but the
  // component still needs something renderable.
  return new ScrutatioNetworkError(cause);
}

/**
 * Fetch once on mount. For the endpoints that describe the corpus rather than
 * answer a question.
 */
function useOnMount<T>(fetcher: () => Promise<T>): AsyncState<T> & { reload: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ ...IDLE, isLoading: true });
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    setState((previous) => ({ ...previous, isLoading: true }));

    fetcher()
      .then((data) => {
        if (alive.current) setState({ data, isLoading: false, error: null });
      })
      .catch((cause: unknown) => {
        if (alive.current) setState({ data: null, isLoading: false, error: asKnownError(cause) });
      });

    return () => {
      // Not cosmetic: `/corpus/stats` returns a 503 while a pipeline run holds the
      // database, and a component that unmounted meanwhile would otherwise set
      // state after unmount.
      alive.current = false;
    };
    // `fetcher` is intentionally not a dependency — callers pass inline closures,
    // and depending on it would refetch on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}

/** Is the backend up? Independent of the database, so it answers during a run. */
export const useHealth = (): AsyncState<Health> & { reload: () => void } =>
  useOnMount(getHealth);

/** Row counts per layer. 503 with code `database_busy` means a run is in flight. */
export const useCorpusStats = (): AsyncState<LayerCounts> & { reload: () => void } =>
  useOnMount(getCorpusStats);

/** The criterion explorer. Refetches whenever the query changes. */
export function useCriteria(query: CriteriaQuery = {}): AsyncState<CriterionPage> {
  const [state, setState] = useState<AsyncState<CriterionPage>>({ ...IDLE, isLoading: true });
  // Serialised so the effect compares by value; a fresh object literal on every
  // render would otherwise refetch forever.
  const key = JSON.stringify(query);

  useEffect(() => {
    let alive = true;
    setState((previous) => ({ ...previous, isLoading: true }));

    getCriteria(JSON.parse(key) as CriteriaQuery)
      .then((data) => {
        if (alive) setState({ data, isLoading: false, error: null });
      })
      .catch((cause: unknown) => {
        if (alive) setState({ data: null, isLoading: false, error: asKnownError(cause) });
      });

    return () => {
      alive = false;
    };
  }, [key]);

  return state;
}

/**
 * The main event: rank trials for a patient description.
 *
 * Imperative rather than render-driven, because it costs money and 30-40 seconds
 * per call — it must fire on an explicit submit, never on a re-render.
 *
 * **Synthetic vignettes only.** The text leaves the machine and reaches a
 * third-party model provider.
 */
export function useMatch(): AsyncState<MatchResponse> & {
  run: (request: MatchRequest) => Promise<void>;
  reset: () => void;
} {
  const [state, setState] = useState<AsyncState<MatchResponse>>(IDLE);
  const inFlight = useRef(0);

  const run = useCallback(async (request: MatchRequest) => {
    // A second submit while the first is running must not let a stale response
    // land last. Only the newest call is allowed to write state.
    const ticket = ++inFlight.current;
    setState({ data: null, isLoading: true, error: null });
    try {
      const data = await postMatch(request);
      if (ticket === inFlight.current) setState({ data, isLoading: false, error: null });
    } catch (cause: unknown) {
      if (ticket === inFlight.current) {
        setState({ data: null, isLoading: false, error: asKnownError(cause) });
      }
    }
  }, []);

  const reset = useCallback(() => {
    inFlight.current += 1;
    setState(IDLE);
  }, []);

  return { ...state, run, reset };
}
