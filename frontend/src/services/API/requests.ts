/**
 * The typed calls against the Scrutatio backend.
 *
 * This file previously held tutorial scaffolding about "suppliers" pointing at
 * `http://myserver.com/api/suppliers/get`, importing types that do not exist in
 * this project. It did not compile.
 *
 * Plain async functions rather than hooks, because the calls are not all
 * render-driven: `/match` fires on a form submit and the corpus explorer paginates
 * on demand. React bindings live in `BackendToFrontendAPI.tsx` and are built on
 * top of these, so the same functions are usable from a loader, a test, or a
 * script without dragging React in.
 */

import type {
  ApiIndex,
  CriterionKind,
  CriterionPage,
  ErrorBody,
  Health,
  LayerCounts,
  MatchRequest,
  MatchResponse,
} from "./types";

/**
 * Where the backend lives.
 *
 * `uv run scrutatio serve` binds 127.0.0.1:8000 by default, and the backend
 * allows exactly `http://localhost:5173` and `http://127.0.0.1:5173` as origins —
 * so the dev server works out of the box. Override with VITE_API_BASE in `.env`
 * for any other deployment, and remember to add that origin to `api_cors_origins`
 * on the server or the browser will block the response before you see it.
 */
export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

/** Thrown for every non-2xx response. Carries the backend's code and request id. */
export class ScrutatioApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly requestId: string;

  constructor(status: number, body: ErrorBody) {
    super(body.detail);
    this.name = "ScrutatioApiError";
    this.status = status;
    this.code = body.code;
    this.requestId = body.request_id;
  }

  /** 503 means "not yet" rather than "no": a pipeline run holds the database, or
   *  a layer has not been built. The UI should say wait, not say broken. */
  get isTemporary(): boolean {
    return this.status === 503 || this.code === "upstream_failed";
  }
}

/** The network itself failed — the server was never reached, so there is no code. */
export class ScrutatioNetworkError extends Error {
  constructor(cause: unknown) {
    super(
      `Could not reach the API at ${API_BASE}. Is it running? ` +
        `Start it with: uv run scrutatio serve`,
    );
    this.name = "ScrutatioNetworkError";
    this.cause = cause;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        // Only on requests that carry one. A GET with a body is rejected by fetch
        // itself, which is what the previous version of this file did on every call.
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (cause) {
    // A CORS rejection also lands here, indistinguishable from the server being
    // down — the browser refuses to say more. Hence the hint in the message.
    throw new ScrutatioNetworkError(cause);
  }

  if (!response.ok) {
    // The backend answers every error with ErrorBody. If something upstream of it
    // (a proxy) answered instead, synthesise the same shape so callers have one
    // thing to handle.
    let body: ErrorBody;
    try {
      body = (await response.json()) as ErrorBody;
    } catch {
      body = {
        code: `http_${response.status}`,
        detail: response.statusText || "The server returned an unreadable error.",
        request_id: response.headers.get("X-Request-ID") ?? "",
      };
    }
    throw new ScrutatioApiError(response.status, body);
  }

  return (await response.json()) as T;
}

/** What the service is and what it answers. Cheap, and touches no database. */
export const getIndex = (): Promise<ApiIndex> => request<ApiIndex>("/");

/** Liveness. Deliberately independent of the database, so it still answers while
 *  a pipeline run holds the file. Use this for an "is the backend up" indicator. */
export const getHealth = (): Promise<Health> => request<Health>("/health");

/** Row counts per layer, plus the extraction signature the corpus was built with. */
export const getCorpusStats = (): Promise<LayerCounts> =>
  request<LayerCounts>("/corpus/stats");

export interface CriteriaQuery {
  kind?: CriterionKind;
  isExclusion?: boolean;
  offset?: number;
  /** 1..500. The server rejects anything larger with a 422. */
  limit?: number;
}

export const getCriteria = (query: CriteriaQuery = {}): Promise<CriterionPage> => {
  const params = new URLSearchParams();
  if (query.kind !== undefined) params.set("kind", query.kind);
  if (query.isExclusion !== undefined) params.set("is_exclusion", String(query.isExclusion));
  if (query.offset !== undefined) params.set("offset", String(query.offset));
  if (query.limit !== undefined) params.set("limit", String(query.limit));
  const qs = params.toString();
  return request<CriterionPage>(`/corpus/criteria${qs ? `?${qs}` : ""}`);
};

/**
 * Rank trials for a free-text patient description.
 *
 * **Synthetic vignettes only.** This leaves the machine and reaches a third-party
 * model provider. The response carries its own `disclaimer` — render it.
 *
 * Slow by design: cold it is ~30-40 seconds, because every candidate trial costs
 * one model call and they run concurrently. Show progress, not a spinner that
 * looks stuck.
 */
export const postMatch = (body: MatchRequest): Promise<MatchResponse> =>
  request<MatchResponse>("/match", { method: "POST", body: JSON.stringify(body) });
