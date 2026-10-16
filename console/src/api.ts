export const SYSTEM_IDS = ["logreg_word", "distilbert", "jev"] as const;
export type SystemId = (typeof SYSTEM_IDS)[number];

export const SYSTEM_NAMES: Record<SystemId, string> = {
  logreg_word: "Logistic regression",
  distilbert: "DistilBERT",
  jev: "JEV",
};

/** Systems that run on this machine; JEV sends the comment to the TypeSafe API. */
export const LOCAL_SYSTEM_IDS: readonly SystemId[] = ["logreg_word", "distilbert"];

export const LABELS = [
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate",
] as const;
export type Label = (typeof LABELS)[number];

export type Decision = "ALLOW" | "HUMAN REVIEW" | "ACTION";
export type SystemState = "available" | "loading" | "unavailable";
export type IssueStatus = "unavailable" | "loading" | "error";

/** Why a system is unavailable, as a stable code the page can act on (the text may change). */
export const NO_API_KEY = "no_api_key";

export interface SystemHealth {
  status: SystemState;
  reason: string | null;
  reason_code: string | null;
}

export type Health = Record<SystemId, SystemHealth>;

/** The API answered (reachable) or did not (unreachable); systems is null if the body was unusable. */
export type HealthResult = { reachable: false } | { reachable: true; systems: Health | null };

export interface ThresholdRow {
  system: SystemId;
  label: Label;
  low: number | null;
  high: number | null;
  action_enabled: boolean;
}

export interface SystemScore {
  system: SystemId;
  probabilities: Record<Label, number>;
  decision: Decision;
  triggering_labels: Label[];
  /** Time the API spent scoring this system, in milliseconds. */
  time_ms: number | null;
}

export interface SystemIssue {
  system: SystemId;
  status: IssueStatus;
  reason: string;
  reason_code: string | null;
}

export interface ScoreResponse {
  results: SystemScore[];
  issues: SystemIssue[];
}

export type ScoreOutcome =
  | { kind: "ok"; data: ScoreResponse }
  | { kind: "validation" }
  | { kind: "network" }
  | { kind: "timeout" }
  | { kind: "server" }
  | { kind: "aborted" };

export interface ScoreOptions {
  /** Restrict scoring to these systems; omit to score every system. */
  systems?: readonly SystemId[];
  signal?: AbortSignal;
}

export interface PolicySummaryRow {
  system: SystemId;
  n_rows: number;
  n_allow: number;
  n_review: number;
  n_action: number;
  m1_review_share: number | null;
  m2_action_precision: number | null;
  m3_miss_rate: number | null;
  m4_rare_miss_rate: number | null;
  action_coverage: number | null;
}

export interface PolicyContrastRow {
  contrast: string;
  metric: string;
  difference: number | null;
  ci_low: number | null;
  ci_high: number | null;
  resamples_used: number;
  resamples_left_out: number;
}

/** Validation and test values per system and label at the primary targets. */
export interface PolicyLabelRow {
  system: SystemId;
  label: Label;
  positives: number | null;
  action_enabled: boolean;
  recall_target: number | null;
  precision_target: number | null;
  validation_recall_at_low: number | null;
  test_recall_at_low: number | null;
  validation_precision_at_high: number | null;
  test_action_precision: number | null;
}

export interface ApRow {
  label: Label;
  positives: number;
  system: SystemId;
  ap: number;
  ap_lo: number;
  ap_hi: number;
}

export interface ApDifferenceRow {
  label: Label;
  contrast: string;
  difference: number | null;
  lo: number | null;
  hi: number | null;
}

export interface BudgetRow {
  budget_pct: number;
  metric: string;
  comparison: string;
  value: number | null;
  lo: number | null;
  hi: number | null;
  status: string;
}

/** How HIGH was fitted: the fewest validation comments it had to be supported by. */
export interface PolicyFit {
  support_floor: number;
}

export interface ResultsPayload {
  per_label_ap?: ApRow[];
  ap_differences?: ApDifferenceRow[];
  policy_summary?: PolicySummaryRow[];
  policy_contrasts?: PolicyContrastRow[];
  policy_per_label?: PolicyLabelRow[];
  review_budget?: BudgetRow[];
  policy_fit?: PolicyFit;
  missing_sections: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isSystemId(value: unknown): value is SystemId {
  return typeof value === "string" && (SYSTEM_IDS as readonly string[]).includes(value);
}

export function isLabel(value: unknown): value is Label {
  return typeof value === "string" && (LABELS as readonly string[]).includes(value);
}

function isDecision(value: unknown): value is Decision {
  return value === "ALLOW" || value === "HUMAN REVIEW" || value === "ACTION";
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function parseHealth(body: unknown): Health | null {
  if (!isRecord(body) || !isRecord(body["systems"])) return null;
  const systems = body["systems"];
  const result: Partial<Health> = {};
  for (const id of SYSTEM_IDS) {
    const entry = systems[id];
    if (!isRecord(entry)) continue;
    const status = entry["status"];
    if (status !== "available" && status !== "loading" && status !== "unavailable") return null;
    const reason = entry["reason"];
    const code = entry["reason_code"];
    result[id] = {
      status,
      reason: typeof reason === "string" ? reason : null,
      reason_code: typeof code === "string" ? code : null,
    };
  }
  const complete = SYSTEM_IDS.every((id) => result[id] !== undefined);
  return complete ? (result as Health) : null;
}

export function parseThresholds(body: unknown): ThresholdRow[] | null {
  if (!Array.isArray(body)) return null;
  const rows: ThresholdRow[] = [];
  for (const item of body as unknown[]) {
    if (!isRecord(item)) return null;
    const { system, label, low, high, action_enabled } = item;
    if (!isSystemId(system) || !isLabel(label) || typeof action_enabled !== "boolean") return null;
    rows.push({ system, label, low: numberOrNull(low), high: numberOrNull(high), action_enabled });
  }
  return rows;
}

function parseScore(item: unknown): SystemScore | null {
  if (!isRecord(item)) return null;
  const { system, probabilities, decision, triggering_labels, time_ms } = item;
  if (!isSystemId(system) || !isDecision(decision)) return null;
  if (!isRecord(probabilities) || !Array.isArray(triggering_labels)) return null;
  const parsed: Partial<Record<Label, number>> = {};
  for (const label of LABELS) {
    const value = probabilities[label];
    if (typeof value !== "number") return null;
    parsed[label] = value;
  }
  return {
    system,
    probabilities: parsed as Record<Label, number>,
    decision,
    triggering_labels: (triggering_labels as unknown[]).filter(isLabel),
    time_ms: numberOrNull(time_ms),
  };
}

export function parseScoreResponse(body: unknown): ScoreResponse | null {
  if (!isRecord(body) || !Array.isArray(body["results"]) || !Array.isArray(body["issues"])) {
    return null;
  }
  const results: SystemScore[] = [];
  for (const item of body["results"] as unknown[]) {
    const parsed = parseScore(item);
    if (parsed === null) return null;
    results.push(parsed);
  }
  const issues: SystemIssue[] = [];
  for (const item of body["issues"] as unknown[]) {
    if (!isRecord(item)) return null;
    const { system, status, reason, reason_code } = item;
    if (!isSystemId(system) || typeof reason !== "string") return null;
    if (status !== "unavailable" && status !== "loading" && status !== "error") return null;
    issues.push({ system, status, reason, reason_code: typeof reason_code === "string" ? reason_code : null });
  }
  return { results, issues };
}

/**
 * Parse a list section: undefined if the key is absent, null if any row is malformed. A row parser
 * returns undefined for a well-formed row the console does not use (for example the supervised stack).
 */
function parseSection<T>(
  body: Record<string, unknown>,
  key: string,
  row: (item: Record<string, unknown>) => T | null | undefined,
): T[] | undefined | null {
  const value = body[key];
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) return null;
  const rows: T[] = [];
  for (const item of value as unknown[]) {
    if (!isRecord(item)) return null;
    const parsed = row(item);
    if (parsed === null) return null;
    if (parsed !== undefined) rows.push(parsed);
  }
  return rows;
}

function parseApRow(item: Record<string, unknown>): ApRow | null | undefined {
  const { label, system, positives, ap, ap_lo, ap_hi } = item;
  if (typeof system === "string" && !isSystemId(system)) return undefined;
  if (!isLabel(label) || !isSystemId(system)) return null;
  const values = [numberOrNull(positives), numberOrNull(ap), numberOrNull(ap_lo), numberOrNull(ap_hi)];
  if (values.some((value) => value === null)) return null;
  return { label, system, positives: values[0] as number, ap: values[1] as number, ap_lo: values[2] as number, ap_hi: values[3] as number };
}

function parsePolicyLabelRow(item: Record<string, unknown>): PolicyLabelRow | null {
  const { system, label, action_enabled } = item;
  if (!isSystemId(system) || !isLabel(label) || typeof action_enabled !== "boolean") return null;
  return {
    system,
    label,
    action_enabled,
    positives: numberOrNull(item["positives"]),
    recall_target: numberOrNull(item["recall_target"]),
    precision_target: numberOrNull(item["precision_target"]),
    validation_recall_at_low: numberOrNull(item["validation_recall_at_low"]),
    test_recall_at_low: numberOrNull(item["test_recall_at_low"]),
    validation_precision_at_high: numberOrNull(item["validation_precision_at_high"]),
    test_action_precision: numberOrNull(item["test_action_precision"]),
  };
}

export function parseResults(body: unknown): ResultsPayload | null {
  if (!isRecord(body)) return null;
  const payload: ResultsPayload = { missing_sections: [] };
  const missing = body["missing_sections"];
  if (Array.isArray(missing)) {
    payload.missing_sections = (missing as unknown[]).filter(
      (item): item is string => typeof item === "string",
    );
  }

  const perLabelAp = parseSection(body, "per_label_ap", parseApRow);
  if (perLabelAp === null) return null;
  if (perLabelAp !== undefined) payload.per_label_ap = perLabelAp;

  const apDifferences = parseSection<ApDifferenceRow>(body, "ap_differences", (item) => {
    if (item["label"] === "mean") return undefined;
    if (!isLabel(item["label"]) || typeof item["contrast"] !== "string") return null;
    return {
      label: item["label"],
      contrast: item["contrast"],
      difference: numberOrNull(item["difference"]),
      lo: numberOrNull(item["lo"]),
      hi: numberOrNull(item["hi"]),
    };
  });
  if (apDifferences === null) return null;
  if (apDifferences !== undefined) payload.ap_differences = apDifferences;

  const summary = parseSection<PolicySummaryRow>(body, "policy_summary", (item) => {
    if (!isSystemId(item["system"])) return null;
    return {
      system: item["system"],
      n_rows: Number(item["n_rows"]),
      n_allow: Number(item["n_allow"]),
      n_review: Number(item["n_review"]),
      n_action: Number(item["n_action"]),
      m1_review_share: numberOrNull(item["m1_review_share"]),
      m2_action_precision: numberOrNull(item["m2_action_precision"]),
      m3_miss_rate: numberOrNull(item["m3_miss_rate"]),
      m4_rare_miss_rate: numberOrNull(item["m4_rare_miss_rate"]),
      action_coverage: numberOrNull(item["action_coverage"]),
    };
  });
  if (summary === null) return null;
  if (summary !== undefined) payload.policy_summary = summary;

  const contrasts = parseSection<PolicyContrastRow>(body, "policy_contrasts", (item) => {
    if (typeof item["contrast"] !== "string" || typeof item["metric"] !== "string") return null;
    return {
      contrast: item["contrast"],
      metric: item["metric"],
      difference: numberOrNull(item["difference"]),
      ci_low: numberOrNull(item["ci_low"]),
      ci_high: numberOrNull(item["ci_high"]),
      resamples_used: Number(item["resamples_used"]),
      resamples_left_out: Number(item["resamples_left_out"]),
    };
  });
  if (contrasts === null) return null;
  if (contrasts !== undefined) payload.policy_contrasts = contrasts;

  const perLabel = parseSection(body, "policy_per_label", parsePolicyLabelRow);
  if (perLabel === null) return null;
  if (perLabel !== undefined) payload.policy_per_label = perLabel;

  const budget = parseSection<BudgetRow>(body, "review_budget", (item) => {
    if (typeof item["metric"] !== "string" || typeof item["comparison"] !== "string") return null;
    const pct = numberOrNull(item["budget_pct"]);
    if (pct === null) return null;
    return {
      budget_pct: pct,
      metric: item["metric"],
      comparison: item["comparison"],
      value: numberOrNull(item["value"]),
      lo: numberOrNull(item["lo"]),
      hi: numberOrNull(item["hi"]),
      status: typeof item["status"] === "string" ? item["status"] : "",
    };
  });
  if (budget === null) return null;
  if (budget !== undefined) payload.review_budget = budget;

  const fit = body["policy_fit"];
  if (isRecord(fit)) {
    const floor = numberOrNull(fit["support_floor"]);
    if (floor !== null) payload.policy_fit = { support_floor: floor };
  }

  return payload;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return (await response.json()) as unknown;
  } catch {
    return null;
  }
}

/** How long each kind of request may take before the console gives up on it. */
export const SCORE_TIMEOUT_MS = 30_000;
export const HEALTH_TIMEOUT_MS = 8_000;
export const DATA_TIMEOUT_MS = 20_000;

type Exchange = { response: Response; body: unknown } | { failure: "timeout" | "aborted" | "network" };

/**
 * One request with a time limit that covers the whole exchange, headers and body. The caller's own
 * signal (a newer request replacing this one) is reported as "aborted", not as a failure.
 */
async function exchange(url: string, init: RequestInit, timeoutMs: number, outer?: AbortSignal): Promise<Exchange> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onOuterAbort = (): void => controller.abort();
  if (outer?.aborted) controller.abort();
  else outer?.addEventListener("abort", onOuterAbort);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const body = await readJson(response);
    if (timedOut) return { failure: "timeout" };
    return { response, body };
  } catch {
    if (timedOut) return { failure: "timeout" };
    if (outer?.aborted) return { failure: "aborted" };
    return { failure: "network" };
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener("abort", onOuterAbort);
  }
}

export async function getHealth(): Promise<HealthResult> {
  const result = await exchange("/api/health", {}, HEALTH_TIMEOUT_MS);
  if ("failure" in result) return { reachable: false };
  if (!result.response.ok) return { reachable: true, systems: null };
  return { reachable: true, systems: parseHealth(result.body) };
}

export async function getThresholds(): Promise<ThresholdRow[] | null> {
  const result = await exchange("/api/thresholds", {}, DATA_TIMEOUT_MS);
  if ("failure" in result || !result.response.ok) return null;
  return parseThresholds(result.body);
}

export async function getResults(): Promise<ResultsPayload | null> {
  const result = await exchange("/api/results", {}, DATA_TIMEOUT_MS);
  if ("failure" in result || !result.response.ok) return null;
  return parseResults(result.body);
}

/** Score one comment. The text goes only in the request body and is not kept anywhere here. */
export async function postScore(text: string, options: ScoreOptions = {}): Promise<ScoreOutcome> {
  const body: { text: string; systems?: readonly SystemId[] } = { text };
  if (options.systems !== undefined) body.systems = options.systems;
  const result = await exchange(
    "/api/score",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    SCORE_TIMEOUT_MS,
    options.signal,
  );
  if ("failure" in result) return { kind: result.failure };
  if (result.response.status === 422) return { kind: "validation" };
  if (!result.response.ok) return { kind: "server" };
  const data = parseScoreResponse(result.body);
  return data === null ? { kind: "server" } : { kind: "ok", data };
}
