import { afterEach, describe, expect, it, vi } from "vitest";

import {
  HEALTH_TIMEOUT_MS,
  LABELS,
  SCORE_TIMEOUT_MS,
  SYSTEM_IDS,
  getHealth,
  parseHealth,
  parseResults,
  parseScoreResponse,
  parseThresholds,
  postScore,
} from "./api";
import healthFixture from "./fixtures/health.json";
import resultsFixture from "./fixtures/results.json";
import scoreAllFixture from "./fixtures/score_all.json";
import thresholdsFixture from "./fixtures/thresholds.json";
import { mockApi } from "./test-utils";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("parseResults on the real response", () => {
  const payload = parseResults(resultsFixture);

  it("reads every section the pages draw from", () => {
    expect(payload).not.toBeNull();
    expect(payload?.missing_sections).toEqual([]);
    expect(payload?.policy_summary).toHaveLength(3);
    expect(payload?.policy_per_label).toHaveLength(SYSTEM_IDS.length * LABELS.length);
    expect(payload?.policy_contrasts?.length).toBeGreaterThan(0);
    expect(payload?.review_budget?.length).toBeGreaterThan(0);
  });

  it("keeps only the three systems the console shows in the average-precision rows", () => {
    const rows = payload?.per_label_ap ?? [];
    expect(rows).toHaveLength(SYSTEM_IDS.length * LABELS.length);
    expect(new Set(rows.map((row) => row.system))).toEqual(new Set(SYSTEM_IDS));
  });

  it("leaves out the all-label mean in the average-precision differences", () => {
    const rows = payload?.ap_differences ?? [];
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((row) => (LABELS as readonly string[]).includes(row.label))).toBe(true);
  });

  it("returns null for a malformed row instead of guessing", () => {
    expect(parseResults({ missing_sections: [], policy_summary: [{ system: "nope" }] })).toBeNull();
    expect(parseResults({ missing_sections: [], policy_per_label: [{ system: "jev", label: "toxic" }] })).toBeNull();
    expect(parseResults({ review_budget: [{ metric: "recall_rare" }] })).toBeNull();
    expect(parseResults([])).toBeNull();
  });

  it("reads the support floor the thresholds were fitted with, and ignores a malformed one", () => {
    expect(parseResults({ missing_sections: [], policy_fit: { support_floor: 25 } })?.policy_fit).toEqual({ support_floor: 25 });
    expect(parseResults({ missing_sections: [], policy_fit: { support_floor: "many" } })?.policy_fit).toBeUndefined();
    expect(parseResults({ missing_sections: [] })?.policy_fit).toBeUndefined();
  });

  it("treats a section that is absent as absent, not empty", () => {
    const sparse = parseResults({ missing_sections: ["policy_per_label"] });
    expect(sparse?.policy_per_label).toBeUndefined();
    expect(sparse?.missing_sections).toEqual(["policy_per_label"]);
  });
});

describe("other parsers", () => {
  it("reads the health response", () => {
    const health = parseHealth(healthFixture);
    expect(health).not.toBeNull();
    expect(Object.keys(health ?? {}).sort()).toEqual([...SYSTEM_IDS].sort());
    expect(parseHealth({ systems: { jev: { status: "weird" } } })).toBeNull();
    expect(parseHealth({})).toBeNull();
  });

  it("reads the reason code from health and from score issues, and tolerates its absence", () => {
    expect(parseHealth(healthFixture)?.jev.reason_code).toBe("no_api_key");
    expect(parseHealth(healthFixture)?.logreg_word.reason_code).toBeNull();
    const parsed = parseScoreResponse(scoreAllFixture);
    expect(parsed?.issues[0]?.reason_code).toBe("no_api_key");
    const older = parseScoreResponse({ results: [], issues: [{ system: "jev", status: "unavailable", reason: "x" }] });
    expect(older?.issues[0]?.reason_code).toBeNull();
  });

  it("reads thresholds with null HIGH", () => {
    const rows = parseThresholds(thresholdsFixture) ?? [];
    expect(rows.length).toBe(18);
    expect(rows.some((row) => row.high === null)).toBe(true);
    expect(parseThresholds({})).toBeNull();
  });

  it("reads a score response and tolerates a missing time", () => {
    const parsed = parseScoreResponse(scoreAllFixture);
    expect(parsed?.results).toHaveLength(2);
    expect(parsed?.results[0]?.time_ms).toEqual(expect.any(Number));
    const without = parseScoreResponse({
      results: [{ ...(scoreAllFixture.results[0] ?? {}), time_ms: undefined }],
      issues: [],
    });
    expect(without?.results[0]?.time_ms).toBeNull();
    expect(parseScoreResponse({ results: [{}], issues: [] })).toBeNull();
  });
});

describe("getHealth", () => {
  it("reports an unreachable API when the request fails", async () => {
    mockApi({ offline: true });
    expect(await getHealth()).toEqual({ reachable: false });
  });

  it("reports a reachable API with the per-system state", async () => {
    mockApi();
    const result = await getHealth();
    expect(result.reachable).toBe(true);
    expect(result.reachable && result.systems?.jev.status).toBe(healthFixture.systems.jev.status);
  });

  it("reports a reachable API whose answer could not be used", async () => {
    mockApi({ health: { nonsense: true } });
    expect(await getHealth()).toEqual({ reachable: true, systems: null });
  });
});

describe("postScore", () => {
  it("sends only the text when no systems are named", async () => {
    const api = mockApi();
    await postScore("hello");
    expect(api.scoreBodies()).toEqual([{ text: "hello" }]);
  });

  it("names the systems when asked to score only some", async () => {
    const api = mockApi();
    await postScore("hello", { systems: ["logreg_word", "distilbert"] });
    expect(api.scoreBodies()).toEqual([{ text: "hello", systems: ["logreg_word", "distilbert"] }]);
  });

  it("maps failures to outcomes", async () => {
    mockApi({ scoreStatus: 422 });
    expect(await postScore("x")).toEqual({ kind: "validation" });
    mockApi({ scoreStatus: 500 });
    expect(await postScore("x")).toEqual({ kind: "server" });
    mockApi({ scoreNetworkError: true });
    expect(await postScore("x")).toEqual({ kind: "network" });
    mockApi({ score: () => ({ unexpected: true }) });
    expect(await postScore("x")).toEqual({ kind: "server" });
  });

  it("reports an aborted request as aborted, not as a network error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new DOMException("aborted", "AbortError");
      }),
    );
    const controller = new AbortController();
    controller.abort();
    expect(await postScore("x", { signal: controller.signal })).toEqual({ kind: "aborted" });
  });

  it("keeps the text out of storage, the console and the address", async () => {
    const marker = "zq-private-marker-text";
    const api = mockApi();
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const logs = [vi.spyOn(console, "log"), vi.spyOn(console, "info"), vi.spyOn(console, "warn"), vi.spyOn(console, "error")];
    await postScore(marker, { systems: ["logreg_word"] });
    expect(setItem).not.toHaveBeenCalled();
    for (const log of logs) expect(JSON.stringify(log.mock.calls)).not.toContain(marker);
    const urls = api.fetch.mock.calls.map((call) => String(call[0]));
    expect(urls.every((url) => !url.includes(marker))).toBe(true);
  });
});

describe("time limits", () => {
  function hang(): ReturnType<typeof vi.fn> {
    return vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
  }

  afterEach(() => {
    vi.useRealTimers();
  });

  it("gives up on a score request that never answers and says timeout, not network", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", hang());
    const pending = postScore("x");
    await vi.advanceTimersByTimeAsync(SCORE_TIMEOUT_MS - 1);
    let settled = false;
    void pending.then(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(2);
    expect(await pending).toEqual({ kind: "timeout" });
  });

  it("keeps a caller's own abort apart from a timeout", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", hang());
    const controller = new AbortController();
    const pending = postScore("x", { signal: controller.signal });
    controller.abort();
    expect(await pending).toEqual({ kind: "aborted" });
  });

  it("does not leave a timer running after an answer", async () => {
    vi.useFakeTimers();
    mockApi();
    await postScore("x");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("treats a health check that never answers as an unreachable API", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", hang());
    const pending = getHealth();
    await vi.advanceTimersByTimeAsync(HEALTH_TIMEOUT_MS + 1);
    expect(await pending).toEqual({ reachable: false });
  });
});
