import { vi } from "vitest";

import health from "./fixtures/health.json";
import results from "./fixtures/results.json";
import scoreAll from "./fixtures/score_all.json";
import scoreLocal from "./fixtures/score_local.json";
import thresholds from "./fixtures/thresholds.json";

export const FIXTURES = { health, results, scoreAll, scoreLocal, thresholds };

/** Make matchMedia answer for the two queries the app reads: narrow screens and reduced motion. */
export function mockMedia({ narrow = false, reduced = true }: { narrow?: boolean; reduced?: boolean } = {}): void {
  window.matchMedia = ((query: string) => ({
    matches: query.includes("max-width") ? narrow : query.includes("prefers-reduced-motion") ? reduced : false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

export interface ScoreBody {
  text: string;
  systems?: string[];
}

export interface ApiMock {
  fetch: ReturnType<typeof vi.fn>;
  /** Bodies of the POST /api/score requests, parsed. */
  scoreBodies: () => ScoreBody[];
  scoreCalls: () => number;
}

interface ApiOptions {
  health?: unknown;
  thresholds?: unknown;
  results?: unknown;
  /** Answer for a score request; receives the parsed body. */
  score?: (body: ScoreBody) => unknown;
  /** Make every request fail as if the API were not running. */
  offline?: boolean;
  /** Make only score requests fail with this HTTP status. */
  scoreStatus?: number;
  /** Make only score requests fail as a network error. */
  scoreNetworkError?: boolean;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Stub fetch with the real captured API responses (or overrides). */
export function mockApi(options: ApiOptions = {}): ApiMock {
  const bodies: ScoreBody[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if (options.offline) throw new TypeError("Failed to fetch");
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (url.endsWith("/api/health")) return jsonResponse(options.health ?? health);
    if (url.endsWith("/api/thresholds")) return jsonResponse(options.thresholds ?? thresholds);
    if (url.endsWith("/api/results")) return jsonResponse(options.results ?? results);
    if (url.endsWith("/api/score")) {
      const body = JSON.parse(String(init?.body ?? "{}")) as ScoreBody;
      bodies.push(body);
      if (options.scoreNetworkError) throw new TypeError("Failed to fetch");
      if (options.scoreStatus !== undefined) return jsonResponse({ detail: [] }, options.scoreStatus);
      const answer = options.score ? options.score(body) : body.systems ? scoreLocal : scoreAll;
      return jsonResponse(answer);
    }
    return jsonResponse({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetch: fetchMock, scoreBodies: () => bodies, scoreCalls: () => bodies.length };
}
