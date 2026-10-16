import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import { LABELS, LOCAL_SYSTEM_IDS, NO_API_KEY, SYSTEM_IDS, SYSTEM_NAMES, postScore } from "../api";
import type { Health, Label, ResultsPayload, SystemId, SystemIssue, SystemScore, ThresholdRow } from "../api";
import { thresholdsOnTestSentence } from "../narrative";
import { DECISION_WORDS, ZONE_CLASS, decideComment, decisionWhy, systemThresholds } from "../decision";
import type { LabelThresholds, SystemThresholds } from "../decision";
import { capitalise, f4, words } from "../format";
import { useMediaQuery, useReducedMotion, useTweenedNumber } from "../hooks";
import { SYSTEM_SHORT } from "../theme";
import { Meter } from "./Meter";
import { Switch } from "./Switch";

export const NOTICE_KEY = "policyjev.notice.dismissed";
export const MAX_CHARACTERS = 2000;
export const LIVE_DEBOUNCE_MS = 320;
export const NARROW_QUERY = "(max-width: 760px)";

export const NOTICE_TEXT =
  "This tool scores text for toxicity, so you can type or paste offensive text. The app does not store or log what you type. JEV runs through the TypeSafe API, so when JEV is ready, pressing Score also sends the comment to that service.";

export const EXAMPLES: readonly { name: string; text: string }[] = [
  { name: "Friendly", text: "Thanks for the update, see you at the meeting tomorrow." },
  { name: "Blunt", text: "This is the worst idea I have ever heard." },
  { name: "Hostile", text: "You are an idiot and nobody wants you here." },
];

function readDismissed(): boolean {
  try {
    return window.sessionStorage.getItem(NOTICE_KEY) === "1";
  } catch {
    return false;
  }
}

function storeDismissed(): void {
  try {
    window.sessionStorage.setItem(NOTICE_KEY, "1");
  } catch {
    // Storage can be unavailable; the notice then simply shows again next time.
  }
}

/** What the last request returned for one system, and which version of the text it scored. */
export interface Entry {
  revision: number;
  score?: SystemScore;
  issue?: SystemIssue;
}
export type Entries = Partial<Record<SystemId, Entry>>;

export type Tile =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "pending" }
  | { kind: "unavailable"; reason: string | null; code: string | null }
  | { kind: "loading" }
  | { kind: "error" }
  | { kind: "scored"; score: SystemScore };

interface TileInputs {
  system: SystemId;
  entry: Entry | undefined;
  /** Version of the text now in the box; entries from older versions are stale for JEV. */
  revision: number;
  hasText: boolean;
  health: Health | null;
  fullRunning: boolean;
}

/**
 * What a system's tile shows. Logistic regression and DistilBERT show their latest result. JEV only
 * shows a result for the text now in the box: after an edit it asks for an explicit Score.
 */
export function tileFor({ system, entry, revision, hasText, health, fullRunning }: TileInputs): Tile {
  const remote = !LOCAL_SYSTEM_IDS.includes(system);
  const healthEntry = health === null ? null : health[system];
  const current = entry !== undefined && (!remote || entry.revision === revision);

  if (current && entry.score !== undefined) return { kind: "scored", score: entry.score };
  if (entry?.issue?.status === "unavailable") {
    return { kind: "unavailable", reason: entry.issue.reason, code: entry.issue.reason_code };
  }
  if (healthEntry?.status === "unavailable" && !current) {
    return { kind: "unavailable", reason: healthEntry.reason, code: healthEntry.reason_code };
  }
  if (current && entry.issue?.status === "loading") return { kind: "loading" };
  if (current && entry.issue?.status === "error") return { kind: "error" };
  if (!hasText) return { kind: "idle" };
  if (healthEntry?.status === "loading" && entry === undefined) return { kind: "loading" };
  if (remote && healthEntry !== null) return fullRunning ? { kind: "running" } : { kind: "pending" };
  return { kind: "idle" };
}

function unavailableWhy(system: SystemId, reason: string | null, code: string | null): string {
  if (system === "jev" && code === NO_API_KEY) {
    return "Scoring needs a TypeSafe API key, set in the environment of the API server or in its .env file.";
  }
  if (reason === null) return "This system is not available.";
  return `${capitalise(reason)}.`;
}

interface VerdictTileProps {
  system: SystemId;
  tile: Tile;
  thresholds: SystemThresholds | null;
  selected: boolean;
  narrow: boolean;
  onSelect: (system: SystemId) => void;
}

function VerdictTile({ system, tile, thresholds, selected, narrow, onSelect }: VerdictTileProps) {
  let className = "vt";
  let word = "";
  let why = "";
  let zone = "";
  let verdict: string | undefined;
  const score = tile.kind === "scored" ? tile.score : null;

  switch (tile.kind) {
    case "scored": {
      const decision =
        thresholds === null
          ? { decision: tile.score.decision, labels: tile.score.triggering_labels }
          : decideComment(tile.score.probabilities, thresholds);
      zone = ZONE_CLASS[decision.decision];
      verdict = decision.decision;
      className = `vt has ${zone}`;
      word = DECISION_WORDS[decision.decision];
      why = decisionWhy(decision);
      break;
    }
    case "unavailable":
      className = "vt na";
      word = "Unavailable";
      why = unavailableWhy(system, tile.reason, tile.code);
      break;
    case "loading":
      className = "vt wait";
      word = "Loading model";
      why = "The model is still loading. Score again in a few seconds.";
      break;
    case "error":
      className = "vt err";
      word = "Could not score";
      why = "This system failed to score the text. Score again.";
      break;
    case "pending":
      className = "vt pend";
      word = "Press Score to run JEV";
      why = "JEV sends the comment to the TypeSafe API, so it only runs when you press Score or Ctrl+Enter.";
      break;
    case "running":
      className = "vt wait";
      word = "Running JEV";
      why = "Waiting for the TypeSafe API.";
      break;
    case "idle":
      className = "vt idle";
      word = "Waiting for text";
      why = "Results appear here.";
      break;
  }

  const interactive = narrow
    ? {
        role: "button" as const,
        tabIndex: 0,
        "aria-pressed": selected,
        onClick: () => onSelect(system),
        onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onSelect(system);
          }
        },
      }
    : {};

  return (
    <div
      className={`${className}${selected ? " sel" : ""}`}
      data-sys={system}
      data-tile={tile.kind}
      data-decision={verdict}
      id={`vt-${system}`}
      {...interactive}
    >
      <div className="nm">
        <b title={SYSTEM_NAMES[system]}>
          <span className="fn">{SYSTEM_NAMES[system]}</span>
          <span className="sn">{SYSTEM_SHORT[system]}</span>
        </b>
        {/* the second row is always there, so tiles line up whether or not a time is known */}
        <span className="ms">{score !== null && score.time_ms !== null ? `${score.time_ms.toFixed(1)} ms` : "\u00a0"}</span>
      </div>
      <div className="word">
        <span key={word}>{word}</span>
      </div>
      <div className="why">{why}</div>
    </div>
  );
}

interface CellProps {
  system: SystemId;
  label: Label;
  tile: Tile;
  thresholds: LabelThresholds | null;
}

function Cell({ system, label, tile, thresholds }: CellProps) {
  const reduced = useReducedMotion();
  const probability = tile.kind === "scored" ? tile.score.probabilities[label] : null;
  const shown = useTweenedNumber(probability, reduced);
  const unavailable = tile.kind === "unavailable";
  const off = tile.kind !== "scored";
  const hot = probability !== null && probability >= 0.05;
  const className = `cell${off ? " off" : ""}${unavailable ? " na" : ""}${hot ? " hot" : ""}`;

  let readout = "–";
  if (unavailable) readout = "Not scored";
  else if (shown !== null) readout = f4(shown);

  const description =
    thresholds === null
      ? `${words(label)}, ${SYSTEM_NAMES[system]}: thresholds not loaded`
      : `${words(label)}, ${SYSTEM_NAMES[system]}: ${
          probability === null ? "not scored" : `probability ${f4(probability)}`
        }`;

  return (
    <div className={className} data-sys={system} data-l={label}>
      <span className="val">{readout}</span>
      {thresholds !== null && !unavailable && (
        <>
          <Meter thresholds={thresholds} probability={probability} label={description} />
          <div className="th">
            <span>LOW {f4(thresholds.low)}</span>
            <span>{thresholds.high === null ? "no HIGH threshold" : `HIGH ${f4(thresholds.high)}`}</span>
          </div>
        </>
      )}
    </div>
  );
}

type Failure = "network" | "timeout" | "validation" | "server";

function failureMessage(kind: Failure): string {
  switch (kind) {
    case "timeout":
      return "The API did not answer in time. Score again; if it keeps failing, check the output of the API server.";
    case "network":
      return "Could not reach the API. Start it with python -m policyjev.api, then score again.";
    case "validation":
      return "The text was rejected. It must be between 1 and 2000 characters. Shorten or fill it in, then score again.";
    case "server":
      return "The API returned an unexpected response. Score again; if it keeps failing, check the output of the API server.";
  }
}

interface ScoreViewProps {
  health: Health | null;
  /** The committed results, for the one sentence about how the thresholds held on the test set. */
  results?: ResultsPayload | null | undefined;
  thresholds: ThresholdRow[] | null | undefined;
  /** Called when a request could not reach the API, so the status chips can recheck. */
  onApiFailure?: () => void;
}

export function ScoreView({ health, thresholds, results, onApiFailure }: ScoreViewProps) {
  const [dismissed, setDismissed] = useState<boolean>(readDismissed);
  const [text, setText] = useState("");
  const [live, setLive] = useState(true);
  const [showThresholds, setShowThresholds] = useState(false);
  const [entries, setEntries] = useState<Entries>({});
  const [failure, setFailure] = useState<Failure | null>(null);
  const [fullRunning, setFullRunning] = useState(false);
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<SystemId>("logreg_word");
  const narrow = useMediaQuery(NARROW_QUERY);

  const revisionRef = useRef(0);
  const liveAbort = useRef<AbortController | null>(null);
  const skipLive = useRef<string | null>(null);
  const debounce = useRef<number | undefined>(undefined);

  const bySystem = useMemo(() => {
    const map = new Map<SystemId, SystemThresholds | null>();
    for (const id of SYSTEM_IDS) map.set(id, thresholds ? systemThresholds(thresholds, id) : null);
    return map;
  }, [thresholds]);

  const length = text.length;
  const tooLong = length > MAX_CHARACTERS;
  // A comment of only spaces and line breaks is treated as empty: nothing to score.
  const hasText = text.trim().length > 0;
  const canScore = hasText && !tooLong && !fullRunning;

  const run = useCallback(
    async (scope: "local" | "all", value: string, version: number): Promise<void> => {
      let signal: AbortSignal | undefined;
      if (scope === "local") {
        liveAbort.current?.abort();
        const controller = new AbortController();
        liveAbort.current = controller;
        signal = controller.signal;
      } else {
        setFullRunning(true);
      }
      const outcome = await postScore(value, {
        ...(scope === "local" ? { systems: LOCAL_SYSTEM_IDS } : {}),
        ...(signal === undefined ? {} : { signal }),
      });
      if (scope === "all") setFullRunning(false);
      if (outcome.kind === "aborted") return;
      if (outcome.kind !== "ok") {
        if (version === revisionRef.current) {
          setFailure(outcome.kind);
          setEntries({});
        }
        if (outcome.kind === "network" || outcome.kind === "timeout") onApiFailure?.();
        return;
      }
      if (version === revisionRef.current) setFailure(null);
      setEntries((previous) => {
        const next: Entries = { ...previous };
        const accept = (system: SystemId, entry: Entry): void => {
          const old = next[system];
          if (old === undefined || old.revision <= version) next[system] = entry;
        };
        for (const score of outcome.data.results) accept(score.system, { revision: version, score });
        for (const issue of outcome.data.issues) accept(issue.system, { revision: version, issue });
        return next;
      });
    },
    [onApiFailure],
  );

  const edit = (value: string): void => {
    window.clearTimeout(debounce.current);
    revisionRef.current += 1;
    setRevision(revisionRef.current);
    setText(value);
    if (value.trim().length === 0) {
      liveAbort.current?.abort();
      setEntries({});
      setFailure(null);
    }
  };

  // Score as I type: after a pause, and only the systems that run on this machine.
  useEffect(() => {
    if (!live || !hasText || tooLong) return;
    if (skipLive.current === text) {
      skipLive.current = null;
      return;
    }
    const version = revisionRef.current;
    debounce.current = window.setTimeout(() => void run("local", text, version), LIVE_DEBOUNCE_MS);
    return () => window.clearTimeout(debounce.current);
    // Only a change to the text or the switch restarts the pause.
  }, [text, live]);

  useEffect(
    () => () => {
      liveAbort.current?.abort();
      window.clearTimeout(debounce.current);
    },
    [],
  );

  /** An explicit Score: every available system, including JEV. */
  const scoreAll = (): void => {
    if (!canScore) return;
    window.clearTimeout(debounce.current);
    void run("all", text, revisionRef.current);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      scoreAll();
    }
  };

  const loadExample = (value: string): void => {
    edit(value);
    // Loading an example scores the local systems straight away when the switch is on; JEV still needs Score.
    if (live) {
      skipLive.current = value;
      void run("local", value, revisionRef.current);
    }
  };

  const tiles = SYSTEM_IDS.map((id) => ({
    id,
    tile: tileFor({ system: id, entry: entries[id], revision, hasText, health, fullRunning }),
  }));

  const loadingModels =
    (health !== null && SYSTEM_IDS.some((id) => health[id].status === "loading")) ||
    SYSTEM_IDS.some((id) => entries[id]?.issue?.status === "loading");

  return (
    <>
      <h1>Score a comment</h1>
      <p className="lede">
        Type or paste text and see what each system decides, label by label, and what the moderation policy
        would do with it.
      </p>
      <div className="score-grid">
        <div>
          <div className="panel input-panel">
            <label className="f" htmlFor="score-text">
              Comment
            </label>
            <textarea
              id="score-text"
              value={text}
              placeholder="Type or paste a comment…"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              aria-describedby="score-count"
              onChange={(event) => edit(event.target.value)}
              onKeyDown={onKeyDown}
            />
            <div className="in-foot">
              <span id="score-count">
                {length} / {MAX_CHARACTERS}
              </span>
              <Switch checked={live} onChange={setLive}>
                Score as I type
              </Switch>
            </div>
            {tooLong && (
              <p role="alert" className="limit">
                The text is longer than {MAX_CHARACTERS} characters. Shorten it, then score again.
              </p>
            )}
            <div className="try">
              <span>Try</span>
              {EXAMPLES.map((example) => (
                <button key={example.name} type="button" className="chip" onClick={() => loadExample(example.text)}>
                  {example.name}
                </button>
              ))}
            </div>
            <button type="button" className="primary" disabled={!canScore} aria-busy={fullRunning} onClick={scoreAll}>
              Score
            </button>
            <p className="hint">Ctrl+Enter also scores.</p>
            {!dismissed && (
              <p className="notice">
                {NOTICE_TEXT}
                <button
                  type="button"
                  onClick={() => {
                    storeDismissed();
                    setDismissed(true);
                  }}
                >
                  Dismiss
                </button>
              </p>
            )}
          </div>
        </div>

        <div>
          {failure !== null && (
            <div role="alert" className="alert" data-failure={failure}>
              <i aria-hidden="true" />
              <span>{failureMessage(failure)}</span>
            </div>
          )}
          {loadingModels && failure === null && (
            <p className="loading-note">Loading models. The first request can take several seconds.</p>
          )}
          <div className="matrix" data-sel={selected}>
            <div className="mrow head" aria-live="polite">
              <div className="spacer" />
              {tiles.map(({ id, tile }) => (
                <VerdictTile
                  key={id}
                  system={id}
                  tile={tile}
                  thresholds={bySystem.get(id) ?? null}
                  selected={narrow && selected === id}
                  narrow={narrow}
                  onSelect={setSelected}
                />
              ))}
            </div>
            <div className={showThresholds ? "show-th" : undefined}>
              {LABELS.map((label) => (
                <div className="mrow rowl" key={label}>
                  <div className="lbl">{words(label)}</div>
                  {tiles.map(({ id, tile }) => (
                    <Cell
                      key={id}
                      system={id}
                      label={label}
                      tile={tile}
                      thresholds={bySystem.get(id)?.[label] ?? null}
                    />
                  ))}
                </div>
              ))}
            </div>
            <div className="legend">
              <span className="k za">
                <i />
                Allow
              </span>
              <span className="k zr">
                <i />
                Human review
              </span>
              <span className="k zx">
                <i />
                Action
              </span>
              <span className="sp" />
              <Switch checked={showThresholds} onChange={setShowThresholds}>
                Show thresholds
              </Switch>
            </div>
            <p className="note">
              Each bar is a probability on a square-root scale from 0 to 1, so small values stay visible. Lit
              segments show the calibrated probability; the colour of a segment is the decision that probability
              would trigger for that label. {thresholdsOnTestSentence(results?.policy_per_label)} See Policy.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
