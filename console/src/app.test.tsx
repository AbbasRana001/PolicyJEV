import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { parseHealth } from "./api";
import type { Health } from "./api";
import { App } from "./App";
import { SystemStatus, needsKey, systemChip } from "./components/SystemStatus";
import { LIVE_DEBOUNCE_MS } from "./components/ScoreView";
import { FIXTURES, mockApi, mockMedia } from "./test-utils";

const health = parseHealth(FIXTURES.health) as Health;

beforeEach(() => mockMedia({ narrow: false, reduced: true }));

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function chip(system: string): HTMLElement {
  const element = document.querySelector(`.sc[data-system="${system}"]`);
  if (element === null) throw new Error(`no chip for ${system}`);
  return element as HTMLElement;
}

function apiChip(): HTMLElement {
  return document.querySelector(".sc[data-api]") as HTMLElement;
}

describe("status chips are derived from the health response", () => {
  it("shows the API as online and each system as ready or needing a key", async () => {
    mockApi();
    render(<App />);
    await waitFor(() => expect(apiChip().dataset["api"]).toBe("online"));
    expect(within(apiChip()).getByText("API online")).toBeTruthy();
    await waitFor(() => expect(chip("logreg_word")).toBeTruthy());
    expect(chip("logreg_word").dataset["state"]).toBe("ready");
    expect(chip("distilbert").dataset["state"]).toBe("ready");
    expect(chip("jev").dataset["state"]).toBe("needs key");
    expect(chip("jev").className).toContain("off");
    expect(chip("logreg_word").className).not.toContain("off");
  });

  it("gives an unreachable API its own visible state and shows no system chips", async () => {
    mockApi({ offline: true });
    render(<App />);
    await waitFor(() => expect(apiChip().dataset["api"]).toBe("offline"));
    expect(within(apiChip()).getByText("API unreachable")).toBeTruthy();
    expect(apiChip().className).toContain("down");
    expect(document.querySelector(".sc[data-system]")).toBeNull();
  });

  it("says it is checking before the first answer, without claiming the API is online", () => {
    mockApi();
    render(<App />);
    expect(apiChip().dataset["api"]).toBe("checking");
    expect(screen.queryByText("API online")).toBeNull();
  });

  it("shows a system that is still loading as loading", async () => {
    mockApi({ health: { status: "ok", systems: { ...health, distilbert: { status: "loading", reason: null, reason_code: null } } } });
    render(<App />);
    await waitFor(() => expect(chip("distilbert").dataset["state"]).toBe("loading"));
    expect(chip("distilbert").className).toContain("wait");
  });

  it("shows unavailable (not needs key) when the reason is not about a key", () => {
    const unhealthy: Health = {
      ...health,
      logreg_word: { status: "unavailable", reason: "model file not found", reason_code: "model_missing" },
    };
    const state = systemChip("logreg_word", unhealthy);
    expect(state.word).toBe("unavailable");
    expect(state.detail).toContain("model file not found");
  });

  it("keys off the reason code, not the wording of the reason", () => {
    expect(needsKey("no_api_key")).toBe(true);
    expect(needsKey("model_missing")).toBe(false);
    expect(needsKey(null)).toBe(false);
    // the text mentions a key but the code says otherwise: not "needs key"
    const wording: Health = { ...health, jev: { status: "unavailable", reason: "API key not configured", reason_code: "model_missing" } };
    expect(systemChip("jev", wording).word).toBe("unavailable");
    // the text says nothing about a key but the code does: "needs key"
    const codeOnly: Health = { ...health, jev: { status: "unavailable", reason: "not set up here", reason_code: "no_api_key" } };
    expect(systemChip("jev", codeOnly).word).toBe("needs key");
    // an older server that sends no code at all is shown as unavailable, never guessed from the text
    const none: Health = { ...health, jev: { status: "unavailable", reason: "API key not configured", reason_code: null } };
    expect(systemChip("jev", none).word).toBe("unavailable");
  });

  it("only offers a state the health response provides", () => {
    for (const status of ["available", "loading", "unavailable"] as const) {
      const word = systemChip("jev", {
        ...health,
        jev: { status, reason: status === "unavailable" ? "x" : null, reason_code: status === "unavailable" ? "no_api_key" : null },
      }).word;
      expect(["ready", "loading", "unavailable", "needs key"]).toContain(word);
    }
  });

  it("is readable without the colours: every chip carries its state in text for assistive technology", async () => {
    mockApi();
    render(<App />);
    await waitFor(() => expect(chip("jev")).toBeTruthy());
    expect(chip("jev").textContent).toContain("needs a TypeSafe API key");
    expect(screen.getByRole("status", { name: "System status" })).toBeTruthy();
  });

  it("renders from props too", () => {
    render(<SystemStatus api="online" systems={health} />);
    expect(document.querySelectorAll(".sc[data-system]")).toHaveLength(3);
  });
});

describe("tabs", () => {
  it("shows one section at a time and marks the chosen tab", async () => {
    mockApi();
    render(<App />);
    const tabs = screen.getAllByRole("tab", { name: /^(Score|Policy|Results)$/ });
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Score", "Policy", "Results"]);
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");
    expect(document.getElementById("view-score")?.hidden).toBe(false);
    expect(document.getElementById("view-policy")?.hidden).toBe(true);

    fireEvent.click(tabs[1] as HTMLElement);
    expect(tabs[1]?.getAttribute("aria-selected")).toBe("true");
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("false");
    expect(document.getElementById("view-policy")?.hidden).toBe(false);
    expect(document.getElementById("view-score")?.hidden).toBe(true);
    expect(screen.getByRole("heading", { level: 1, name: "Policy" })).toBeTruthy();

    fireEvent.click(tabs[2] as HTMLElement);
    expect(screen.getByRole("heading", { level: 1, name: "Results" })).toBeTruthy();
  });

  it("slides the highlight to the chosen tab", () => {
    mockApi();
    render(<App />);
    const ink = document.querySelector(".tab-ink") as HTMLElement;
    expect(ink.style.transform).toMatch(/translateX\(/);
    expect(ink.className).toBe("tab-ink");
  });

  it("keeps what was typed when the person visits another section and comes back", async () => {
    mockApi();
    render(<App />);
    const box = screen.getByLabelText("Comment") as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "a draft" } });
    fireEvent.click(screen.getByRole("tab", { name: "Policy" }));
    fireEvent.click(screen.getByRole("tab", { name: "Score" }));
    expect((screen.getByLabelText("Comment") as HTMLTextAreaElement).value).toBe("a draft");
  });
});

describe("the whole app with the real captured API answers", () => {
  it("scores the benign sentence through the typed path, local systems only", async () => {
    vi.useFakeTimers();
    const api = mockApi();
    render(<App />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10);
    });
    fireEvent.change(screen.getByLabelText("Comment"), { target: { value: "Thanks for the update, see you at the meeting tomorrow." } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_DEBOUNCE_MS + 10);
    });
    expect(api.scoreBodies()).toHaveLength(1);
    expect(api.scoreBodies()[0]?.systems).toEqual(["logreg_word", "distilbert"]);
    expect(document.getElementById("vt-logreg_word")?.dataset["decision"]).toBe("ALLOW");
  });

  it("rechecks the API when a request fails and shows it as unreachable", async () => {
    mockApi({ scoreNetworkError: true });
    render(<App />);
    await waitFor(() => expect(apiChip().dataset["api"]).toBe("online"));
    fireEvent.click(screen.getByRole("switch", { name: "Score as I type" }));
    fireEvent.change(screen.getByLabelText("Comment"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Score" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert").textContent).toContain("Could not reach the API");
  });
});
