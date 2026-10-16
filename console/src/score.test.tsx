import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { parseHealth, parseThresholds } from "./api";
import type { Health } from "./api";
import { Meter, litCells } from "./components/Meter";
import {
  EXAMPLES,
  LIVE_DEBOUNCE_MS,
  MAX_CHARACTERS,
  NOTICE_KEY,
  NOTICE_TEXT,
  ScoreView,
  tileFor,
} from "./components/ScoreView";
import css from "./index.css?raw";
import { FIXTURES, mockApi, mockMedia } from "./test-utils";

const thresholds = parseThresholds(FIXTURES.thresholds) ?? [];
const baseHealth = parseHealth(FIXTURES.health) as Health;
const withJev: Health = { ...baseHealth, jev: { status: "available", reason: null, reason_code: null } };
const SENTENCE = EXAMPLES[0]?.text ?? "";

function view(health: Health | null = baseHealth, onApiFailure?: () => void) {
  return render(<ScoreView health={health} thresholds={thresholds} onApiFailure={onApiFailure} />);
}

function box(): HTMLTextAreaElement {
  return screen.getByLabelText("Comment") as HTMLTextAreaElement;
}

function type(value: string): void {
  fireEvent.change(box(), { target: { value } });
}

function tile(system: string): HTMLElement {
  const element = document.getElementById(`vt-${system}`);
  if (element === null) throw new Error(`no tile for ${system}`);
  return element;
}

async function tick(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  mockMedia({ narrow: false, reduced: true });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("the empty Score page", () => {
  it("shows a tile per system, waiting for text, and JEV as unavailable with the reason", () => {
    mockApi();
    view();
    expect(within(tile("logreg_word")).getByText("Waiting for text")).toBeTruthy();
    expect(within(tile("distilbert")).getByText("Waiting for text")).toBeTruthy();
    expect(within(tile("jev")).getByText("Unavailable")).toBeTruthy();
    expect(within(tile("jev")).getByText(/TypeSafe API key/)).toBeTruthy();
  });

  it("explains an unavailable JEV from the reason code, not from the wording of the reason", () => {
    mockApi();
    const reworded: Health = { ...baseHealth, jev: { status: "unavailable", reason: "some other words", reason_code: "no_api_key" } };
    const { unmount } = view(reworded);
    expect(within(tile("jev")).getByText(/TypeSafe API key/)).toBeTruthy();
    unmount();
    const other: Health = { ...baseHealth, jev: { status: "unavailable", reason: "API key not configured", reason_code: "manifest_failed" } };
    view(other);
    expect(within(tile("jev")).queryByText(/TypeSafe API key/)).toBeNull();
    expect(within(tile("jev")).getByText("API key not configured.")).toBeTruthy();
  });

  it("shows the legend, the threshold switch and a row of six labels with meters", () => {
    mockApi();
    view();
    expect(screen.getByText("Human review", { selector: ".legend .k" })).toBeTruthy();
    expect(screen.getByRole("switch", { name: "Show thresholds" }).getAttribute("aria-checked")).toBe("false");
    expect(document.querySelectorAll(".mrow.rowl")).toHaveLength(6);
    expect(document.querySelectorAll(".meter")).toHaveLength(6 * 3 - 6); // none for the unavailable system
  });

  it("disables Score until there is text", () => {
    mockApi();
    view();
    const button = screen.getByRole("button", { name: "Score" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    type("hello");
    expect(button.disabled).toBe(false);
  });

  it("treats text of only spaces and line breaks like an empty box", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    const button = screen.getByRole("button", { name: "Score" }) as HTMLButtonElement;
    for (const blank of ["   ", String.fromCharCode(10, 10), " " + String.fromCharCode(9) + " " + String.fromCharCode(10) + " "]) {
      type(blank);
      expect(button.disabled, JSON.stringify(blank)).toBe(true);
      fireEvent.keyDown(box(), { key: "Enter", ctrlKey: true });
      await tick(2000);
    }
    expect(api.scoreCalls()).toBe(0);
    expect(tile("logreg_word").getAttribute("data-tile")).toBe("idle");
    // the counter still counts every character
    type("    ");
    expect(screen.getByText(`4 / ${MAX_CHARACTERS}`)).toBeTruthy();
  });

  it("clears earlier results when the box is replaced by blank text", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    type(SENTENCE);
    await tick(LIVE_DEBOUNCE_MS + 10);
    expect(tile("logreg_word").getAttribute("data-tile")).toBe("scored");
    type("   ");
    await tick(1000);
    expect(tile("logreg_word").getAttribute("data-tile")).toBe("idle");
    expect(api.scoreCalls()).toBe(1);
  });

  it("still sends text that has spaces around real words, as typed", async () => {
    const api = mockApi();
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("  hello  ");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(api.scoreCalls()).toBe(1));
    expect(api.scoreBodies()[0]?.text).toBe("  hello  ");
  });

  it("counts characters", () => {
    mockApi();
    view();
    type("abcde");
    expect(screen.getByText(`5 / ${MAX_CHARACTERS}`)).toBeTruthy();
  });
});

describe("Score as I type", () => {
  it("waits for a pause of 320 ms and sends one request with only the local systems", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    expect(LIVE_DEBOUNCE_MS).toBe(320);

    type("Thanks");
    await tick(200);
    type("Thanks for");
    await tick(319);
    expect(api.scoreCalls()).toBe(0);

    await tick(2);
    expect(api.scoreCalls()).toBe(1);
    expect(api.scoreBodies()[0]).toEqual({ text: "Thanks for", systems: ["logreg_word", "distilbert"] });
  });

  it("never asks for JEV while typing, however long the text grows", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    for (const value of ["a", "ab", "abc"]) {
      type(value);
      await tick(LIVE_DEBOUNCE_MS + 10);
    }
    expect(api.scoreCalls()).toBe(3);
    for (const body of api.scoreBodies()) expect(body.systems).not.toContain("jev");
  });

  it("shows the local verdicts and tells the person to press Score for JEV", async () => {
    vi.useFakeTimers();
    mockApi();
    view(withJev);
    type(SENTENCE);
    await tick(LIVE_DEBOUNCE_MS + 10);

    expect(tile("logreg_word").getAttribute("data-decision")).toBe("ALLOW");
    expect(within(tile("logreg_word")).getByText("Allow")).toBeTruthy();
    expect(tile("distilbert").getAttribute("data-decision")).toBe("ALLOW");
    expect(within(tile("jev")).getByText("Press Score to run JEV")).toBeTruthy();
  });

  it("does not call the API when the switch is off", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    await tick(2000);
    expect(api.scoreCalls()).toBe(0);
  });

  it("clears the results when the text is emptied and sends nothing for empty text", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    type(SENTENCE);
    await tick(LIVE_DEBOUNCE_MS + 10);
    expect(tile("logreg_word").getAttribute("data-decision")).toBe("ALLOW");

    type("");
    await tick(1000);
    expect(api.scoreCalls()).toBe(1);
    expect(tile("logreg_word").getAttribute("data-tile")).toBe("idle");
  });

  it("does not score text over the limit", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    type("x".repeat(MAX_CHARACTERS + 1));
    await tick(2000);
    expect(api.scoreCalls()).toBe(0);
    expect(screen.getByRole("alert").textContent).toContain(`longer than ${MAX_CHARACTERS} characters`);
    expect((screen.getByRole("button", { name: "Score" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("scores the example straight away, locally only, without waiting for the pause", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    view(withJev);
    fireEvent.click(screen.getByRole("button", { name: EXAMPLES[0]?.name ?? "" }));
    await tick(1);
    expect(box().value).toBe(SENTENCE);
    expect(api.scoreCalls()).toBe(1);
    expect(api.scoreBodies()[0]?.systems).toEqual(["logreg_word", "distilbert"]);
    await tick(1000);
    expect(api.scoreCalls()).toBe(1);
  });
});

describe("an explicit Score", () => {
  it("asks for every system, including JEV, when Score is pressed", async () => {
    const api = mockApi({ score: () => FIXTURES.scoreAll });
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(api.scoreCalls()).toBe(1));
    expect(api.scoreBodies()[0]).toEqual({ text: SENTENCE });
  });

  it("does the same on Ctrl+Enter", async () => {
    const api = mockApi();
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    fireEvent.keyDown(box(), { key: "Enter", ctrlKey: true });
    await waitFor(() => expect(api.scoreCalls()).toBe(1));
    expect(api.scoreBodies()[0]?.systems).toBeUndefined();
  });

  it("shows a JEV verdict for the text that was scored and asks again once the text changes", async () => {
    const jevScore = {
      system: "jev",
      probabilities: { toxic: 0.9, severe_toxic: 0, obscene: 0, threat: 0, insult: 0, identity_hate: 0 },
      decision: "ACTION",
      triggering_labels: ["toxic"],
      time_ms: 412.3,
    };
    const answer = { results: [...FIXTURES.scoreAll.results, jevScore], issues: [] };
    mockApi({ score: (body) => (body.systems ? FIXTURES.scoreLocal : answer) });
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("a comment");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));

    await waitFor(() => expect(tile("jev").getAttribute("data-tile")).toBe("scored"));
    expect(tile("jev").getAttribute("data-decision")).toBe("ACTION");
    expect(within(tile("jev")).getByText("412.3 ms")).toBeTruthy();
    expect(within(tile("jev")).getByText(/triggered by toxic/i)).toBeTruthy();

    type("a comment, edited");
    expect(tile("jev").getAttribute("data-tile")).toBe("pending");
    expect(within(tile("jev")).getByText("Press Score to run JEV")).toBeTruthy();
  });

  it("keeps Score disabled while a request is running so JEV is not sent twice", async () => {
    let release: (value: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    vi.stubGlobal("fetch", vi.fn(() => pending));
    view(withJev);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("a comment");
    const button = screen.getByRole("button", { name: "Score" }) as HTMLButtonElement;
    fireEvent.click(button);
    await waitFor(() => expect(button.disabled).toBe(true));
    expect(within(tile("jev")).getByText("Running JEV")).toBeTruthy();
    release(new Response(JSON.stringify(FIXTURES.scoreAll), { status: 200 }));
    await waitFor(() => expect(button.disabled).toBe(false));
  });
});

describe("latency", () => {
  it("shows the time a system took when the response has it", async () => {
    mockApi();
    view();
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(tile("logreg_word").getAttribute("data-tile")).toBe("scored"));
    const expected = FIXTURES.scoreAll.results[0]?.time_ms ?? 0;
    expect(tile("logreg_word").querySelector(".ms")?.textContent).toMatch(/^\d+\.\d ms$/);
    expect(expected).toBeGreaterThan(0);
  });

  it("leaves it out when the response has none", async () => {
    const stripped = {
      ...FIXTURES.scoreAll,
      results: FIXTURES.scoreAll.results.map(({ time_ms: _omitted, ...rest }) => rest),
    };
    mockApi({ score: () => stripped });
    view();
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(tile("logreg_word").getAttribute("data-tile")).toBe("scored"));
    // the second header row is still there (so tiles line up) but holds no time
    expect(tile("logreg_word").querySelector(".ms")?.textContent?.trim()).toBe("");
    expect(document.body.textContent).not.toMatch(/\d ms\b/);
  });
});

describe("errors", () => {
  it("shows a visible alert when the API cannot be reached, and asks for a recheck", async () => {
    const failure = vi.fn();
    mockApi({ scoreNetworkError: true });
    view(baseHealth, failure);
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("hello");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Could not reach the API. Start it with python -m policyjev.api, then score again.");
    expect(alert.className).toContain("alert");
    expect(failure).toHaveBeenCalled();
  });

  it("explains a rejected text and an unexpected answer in plain words", async () => {
    mockApi({ scoreStatus: 422 });
    view();
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("hello");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    expect((await screen.findByRole("alert")).textContent).toContain("between 1 and 2000 characters");

    mockApi({ scoreStatus: 500 });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("unexpected response"));
  });

  it("clears the alert once a request works again", async () => {
    mockApi({ scoreNetworkError: true });
    view();
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("hello");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await screen.findByRole("alert");

    mockApi();
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });

  it("shows a loading tile and note while a model is still loading", async () => {
    const loading: Health = { ...baseHealth, distilbert: { status: "loading", reason: null, reason_code: null } };
    mockApi({
      score: () => ({
        results: FIXTURES.scoreLocal.results.slice(0, 1),
        issues: [{ system: "distilbert", status: "loading", reason: "model is loading" }],
      }),
    });
    view(loading);
    expect(screen.getByText(/Loading models/)).toBeTruthy();
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type("hello");
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(tile("distilbert").getAttribute("data-tile")).toBe("loading"));
    expect(within(tile("distilbert")).getByText("Loading model")).toBeTruthy();
  });
});

describe("the notice and privacy", () => {
  it("shows the notice text and remembers the dismissal for the session", () => {
    mockApi();
    const first = view();
    expect(screen.getByText(NOTICE_TEXT, { exact: false })).toBeTruthy();
    expect(NOTICE_TEXT).toBe(
      "This tool scores text for toxicity, so you can type or paste offensive text. The app does not store or log what you type. JEV runs through the TypeSafe API, so when JEV is ready, pressing Score also sends the comment to that service.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(NOTICE_TEXT, { exact: false })).toBeNull();
    expect(window.sessionStorage.getItem(NOTICE_KEY)).toBe("1");
    first.unmount();
    view();
    expect(screen.queryByText(NOTICE_TEXT, { exact: false })).toBeNull();
  });

  it("still works when session storage is unavailable", () => {
    mockApi();
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    view();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(NOTICE_TEXT, { exact: false })).toBeNull();
  });

  it("never writes what is typed to storage or the console", async () => {
    vi.useFakeTimers();
    const marker = "zq-private-marker-text";
    mockApi();
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const logs = [vi.spyOn(console, "log"), vi.spyOn(console, "info"), vi.spyOn(console, "warn"), vi.spyOn(console, "error")];
    view(withJev);
    type(marker);
    await tick(LIVE_DEBOUNCE_MS + 10);
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await tick(50);
    for (const call of setItem.mock.calls) expect(JSON.stringify(call)).not.toContain(marker);
    for (const log of logs) expect(JSON.stringify(log.mock.calls)).not.toContain(marker);
    expect(JSON.stringify({ local: { ...window.localStorage }, session: { ...window.sessionStorage } })).not.toContain(marker);
  });

  it("keeps the browser from remembering or sending the text elsewhere", () => {
    mockApi();
    view();
    expect(box().getAttribute("autocomplete")).toBe("off");
    expect(box().getAttribute("spellcheck")).toBe("false");
  });
});

describe("small screens", () => {
  beforeEach(() => mockMedia({ narrow: true, reduced: true }));

  it("makes the tiles pressable and shows the meters of the chosen system", () => {
    mockApi();
    view();
    const matrix = document.querySelector(".matrix") as HTMLElement;
    expect(matrix.dataset["sel"]).toBe("logreg_word");
    expect(tile("logreg_word").getAttribute("aria-pressed")).toBe("true");
    expect(tile("distilbert").getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(tile("distilbert"));
    expect(matrix.dataset["sel"]).toBe("distilbert");
    expect(tile("distilbert").getAttribute("aria-pressed")).toBe("true");

    fireEvent.keyDown(tile("jev"), { key: "Enter" });
    expect(matrix.dataset["sel"]).toBe("jev");
  });

  it("does not make the tiles buttons on wide screens", () => {
    mockMedia({ narrow: false, reduced: true });
    mockApi();
    view();
    expect(tile("logreg_word").getAttribute("role")).toBeNull();
  });
});

describe("tileFor", () => {
  const entry = { revision: 1, score: FIXTURES.scoreAll.results[0] as never };

  it("shows JEV only for the text it scored", () => {
    const jev = { revision: 1, score: { ...(FIXTURES.scoreAll.results[0] as object), system: "jev" } as never };
    const common = { system: "jev" as const, hasText: true, health: withJev, fullRunning: false };
    expect(tileFor({ ...common, entry: jev, revision: 1 }).kind).toBe("scored");
    expect(tileFor({ ...common, entry: jev, revision: 2 }).kind).toBe("pending");
    expect(tileFor({ ...common, entry: undefined, revision: 2, fullRunning: true }).kind).toBe("running");
  });

  it("keeps showing the latest local result while the person types", () => {
    const common = { system: "logreg_word" as const, hasText: true, health: withJev, fullRunning: false };
    expect(tileFor({ ...common, entry, revision: 5 }).kind).toBe("scored");
  });

  it("does not offer JEV before the API has said whether it is available", () => {
    expect(tileFor({ system: "jev", entry: undefined, revision: 1, hasText: true, health: null, fullRunning: false }).kind).toBe("idle");
  });
});

describe("the meter", () => {
  const rule = { low: 0.1, high: 0.6 };

  it("lights cells up to the square root of the probability", () => {
    expect(litCells(0)).toBe(0);
    expect(litCells(1)).toBe(40);
    expect(litCells(0.25)).toBe(20);
    const { container } = render(<Meter thresholds={rule} probability={0.25} label="x" />);
    expect(container.querySelectorAll(".c.lit")).toHaveLength(20);
    expect(container.querySelectorAll(".c.peak")).toHaveLength(1);
  });

  it("colours each cell by the decision that probability would trigger", () => {
    const { container } = render(<Meter thresholds={rule} probability={null} label="x" />);
    const cells = [...container.querySelectorAll(".c")];
    expect(cells).toHaveLength(40);
    const zones = cells.map((cell) => [...cell.classList].find((name) => /^z[arx]$/.test(name)));
    expect(zones[0]).toBe("za");
    expect(zones[39]).toBe("zx");
    expect(new Set(zones)).toEqual(new Set(["za", "zr", "zx"]));
    // the first cell of each zone is the first whose centre probability reaches the threshold
    const firstAtOrAbove = (threshold: number): number =>
      Array.from({ length: 40 }, (_, i) => ((i + 0.5) / 40) ** 2).findIndex((p) => p >= threshold);
    expect(zones.indexOf("zr")).toBe(firstAtOrAbove(rule.low));
    expect(zones.indexOf("zx")).toBe(firstAtOrAbove(rule.high));
  });

  it("has no action zone and one tick when automatic action is not enabled", () => {
    const { container } = render(<Meter thresholds={{ low: 0.1, high: null }} probability={0.5} label="x" />);
    expect(container.querySelectorAll(".c.zx")).toHaveLength(0);
    expect(container.querySelectorAll(".tk")).toHaveLength(1);
    expect(container.querySelector(".meter")?.getAttribute("title")).toContain("no HIGH threshold");
  });
});

describe("the tile header", () => {
  it("has two rows in every tile, the name first and the time directly beneath it, whatever the state", async () => {
    const jevScore = { system: "jev", probabilities: { toxic: 0.2, severe_toxic: 0, obscene: 0, threat: 0, insult: 0, identity_hate: 0 }, decision: "HUMAN REVIEW", triggering_labels: ["toxic"], time_ms: 412.3 };
    mockApi({ score: () => ({ results: [...FIXTURES.scoreAll.results, jevScore], issues: [] }) });
    view(withJev);
    const header = (system: string) => [...tile(system).querySelector(".nm")!.children].map((child) => child.tagName.toLowerCase());
    // before any result: two rows, the second empty
    for (const system of ["logreg_word", "distilbert", "jev"]) {
      expect(header(system)).toEqual(["b", "span"]);
      expect(tile(system).querySelector(".nm .ms")?.textContent?.trim()).toBe("");
    }
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    type(SENTENCE);
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await waitFor(() => expect(tile("jev").getAttribute("data-tile")).toBe("scored"));
    for (const system of ["logreg_word", "distilbert", "jev"]) {
      expect(header(system)).toEqual(["b", "span"]);
      expect(tile(system).querySelector(".nm .ms")?.textContent).toMatch(/^\d+\.\d ms$/);
    }
  });

  it("gives the same two rows when the system is unavailable, loading, in error or waiting for Score", () => {
    mockApi();
    view(baseHealth);
    expect([...tile("jev").querySelector(".nm")!.children].map((c) => c.tagName.toLowerCase())).toEqual(["b", "span"]);
    expect(tile("jev").getAttribute("data-tile")).toBe("unavailable");
  });

  it("keeps the layout independent of wrapping: the name cannot wrap, and the heading has room for two lines in every tile", () => {
    const rules = css.match(/\.vt \.nm b \{[^}]*\}/g)?.join(" ") ?? "";
    expect(rules).toMatch(/white-space:\s*nowrap/);
    expect(rules).toMatch(/text-overflow:\s*ellipsis/);
    expect(css).toMatch(/\.vt \.word \{[^}]*min-height:\s*calc\(2\.1 \* clamp\(26px, 2\.6vw, 34px\)\)/);
    expect(css).toMatch(/\.vt \.nm \.ms \{[^}]*min-height:\s*19px/);
  });
});
