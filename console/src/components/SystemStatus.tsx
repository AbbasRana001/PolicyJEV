import { NO_API_KEY, SYSTEM_IDS, SYSTEM_NAMES } from "../api";
import type { Health, SystemId } from "../api";
import type { ApiState } from "../hooks";
import { systemStyle } from "../theme";

export interface Chip {
  /** Short state word shown beside the name when there is room. */
  word: string;
  /** Longer explanation for the tooltip and assistive technology. */
  detail: string;
  tone: "ready" | "wait" | "off";
}

/** True when the API's reason code for a system being unavailable is a missing API key. */
export function needsKey(reasonCode: string | null): boolean {
  return reasonCode === NO_API_KEY;
}

/** Chip text for one system, taken from its health entry (ready, loading, needs key or unavailable). */
export function systemChip(id: SystemId, health: Health): Chip {
  const entry = health[id];
  if (entry.status === "available") {
    return { word: "ready", detail: `${SYSTEM_NAMES[id]}: ready`, tone: "ready" };
  }
  if (entry.status === "loading") {
    return { word: "loading", detail: `${SYSTEM_NAMES[id]}: the model is loading`, tone: "wait" };
  }
  if (needsKey(entry.reason_code)) {
    return {
      word: "needs key",
      detail: `${SYSTEM_NAMES[id]}: needs a TypeSafe API key on the API server`,
      tone: "off",
    };
  }
  return {
    word: "unavailable",
    detail: `${SYSTEM_NAMES[id]}: unavailable${entry.reason === null ? "" : ` (${entry.reason})`}`,
    tone: "off",
  };
}

const API_CHIP: Record<ApiState, { className: string; word: string; title: string }> = {
  checking: { className: "sc api wait", word: "Checking API", title: "Checking whether the API is reachable" },
  online: { className: "sc api", word: "API online", title: "The API is reachable" },
  offline: {
    className: "sc down",
    word: "API unreachable",
    title: "The API did not answer. Start it with python -m policyjev.api.",
  },
};

/** Top-bar status: the API chip, then one chip per system once the API has answered. */
export function SystemStatus({ api, systems }: { api: ApiState; systems: Health | null }) {
  const chip = API_CHIP[api];
  const showSystems = api === "online" && systems !== null;
  return (
    <div className="status" role="status" aria-label="System status">
      <span className={chip.className} title={chip.title} data-api={api}>
        <i className="d" aria-hidden="true" />
        <span className="nm">{chip.word}</span>
        <span className="sr-only">{chip.title}</span>
      </span>
      {showSystems && <span className="sep" aria-hidden="true" />}
      {showSystems &&
        SYSTEM_IDS.map((id) => {
          const state = systemChip(id, systems);
          return (
            <span
              key={id}
              className={`sc${state.tone === "off" ? " off" : ""}${state.tone === "wait" ? " wait" : ""}`}
              style={systemStyle(id)}
              title={state.detail}
              data-system={id}
              data-state={state.word}
            >
              <i className="d" aria-hidden="true" />
              <span className="nm" aria-hidden="true">
                {SYSTEM_NAMES[id]}
              </span>
              <small aria-hidden="true">{state.word}</small>
              <span className="sr-only">{state.detail}</span>
            </span>
          );
        })}
    </div>
  );
}
