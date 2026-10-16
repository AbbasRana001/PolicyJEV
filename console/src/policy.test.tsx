import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { LABELS, SYSTEM_IDS, SYSTEM_NAMES, parseResults, parseThresholds } from "./api";
import type { ResultsPayload, ThresholdRow } from "./api";
import { PolicyView, trustText } from "./components/PolicyView";
import { FLOW_LABEL_MIN_PERCENT } from "./components/SummaryCharts";
import { agreementRows } from "./components/Agreement";
import { DEFAULT_SLIDER } from "./components/Explorer";
import { decideLabel, decisionWhy, probabilityFromSlider } from "./decision";
import { f4, formatCount } from "./format";
import { COMPARISON_MEASURES, FIVE_MEASURES, bestIndex, measureValues } from "./measures";
import { FIXTURES, mockMedia } from "./test-utils";

const thresholds = parseThresholds(FIXTURES.thresholds) as ThresholdRow[];
const results = parseResults(FIXTURES.results) as ResultsPayload;

function view(overrides: Partial<{ thresholds: ThresholdRow[] | null | undefined; results: ResultsPayload | null | undefined }> = {}) {
  return render(
    <PolicyView
      thresholds={"thresholds" in overrides ? overrides.thresholds : thresholds}
      results={"results" in overrides ? overrides.results : results}
    />,
  );
}

function slider(): HTMLInputElement {
  return screen.getByLabelText("Probability every label scores") as HTMLInputElement;
}

function move(position: number): void {
  fireEvent.change(slider(), { target: { value: String(position) } });
}

function row(label: string): HTMLElement {
  const element = document.querySelector(`.explore .xr[data-l="${label}"]`);
  if (element === null) throw new Error(`no row for ${label}`);
  return element as HTMLElement;
}

function outcome(): HTMLElement {
  return document.querySelector(".outcome") as HTMLElement;
}

beforeEach(() => mockMedia({ narrow: false, reduced: true }));

describe("the threshold explorer", () => {
  it("starts at the design's default probability and shows a decision per label", () => {
    view();
    expect(slider().value).toBe(String(DEFAULT_SLIDER));
    expect(document.querySelector("[data-probability]")?.textContent).toBe(f4(probabilityFromSlider(DEFAULT_SLIDER)));
    expect(document.querySelectorAll(".explore .xr[data-l]")).toHaveLength(6);
  });

  it("has a slider that a keyboard can drive and a screen reader can read", () => {
    view();
    expect(slider().type).toBe("range");
    expect(slider().min).toBe("0");
    expect(slider().max).toBe("1000");
    expect(slider().step).toBe("1");
    move(500);
    expect(slider().getAttribute("aria-valuetext")).toBe(`probability ${f4(0.25)}`);
  });

  it("announces the outcome politely", () => {
    view();
    expect(outcome().getAttribute("aria-live")).toBe("polite");
  });

  it("allows everything at zero and acts on the labels that can act at one", () => {
    view();
    move(0);
    expect(outcome().dataset["outcome"]).toBe("ALLOW");
    expect(outcome().textContent).toBe("AllowNo label reached its review threshold.");
    move(1000);
    expect(outcome().dataset["outcome"]).toBe("ACTION");
    const enabled = thresholds.filter((item) => item.system === "logreg_word" && item.action_enabled).map((item) => item.label);
    for (const label of LABELS) {
      const expected = enabled.includes(label) ? "ACTION" : "HUMAN REVIEW";
      expect(row(label).dataset["decision"]).toBe(expected);
      expect(within(row(label)).getByText(expected === "ACTION" ? "Action" : "Human review", { selector: ".dchip" })).toBeTruthy();
    }
    expect(outcome().textContent).toMatch(/^ActionAction triggered by .+\.$/);
    // the sentence "The comment gets the most severe of six outcomes." is not repeated here: the intro states the rule
    expect(outcome().textContent).not.toContain("most severe");
    expect(document.body.textContent).not.toContain("The comment gets the most severe of six outcomes");
  });

  it("uses the same sentence as the Score tiles for the same outcome, from one function", () => {
    view();
    move(1000);
    const labels = thresholds.filter((item) => item.system === "logreg_word" && item.action_enabled).map((item) => item.label);
    const expected = decisionWhy({ decision: "ACTION", labels });
    expect(outcome().querySelector(".ot")?.textContent).toBe(expected);
    move(300);
    const review = outcome().querySelector(".ot")?.textContent ?? "";
    expect(review).toMatch(/^(Review|Action) triggered by /);
    move(0);
    expect(outcome().querySelector(".ot")?.textContent).toBe(decisionWhy({ decision: "ALLOW", labels: [] }));
  });

  it("is headed 'Try a probability' and says the thresholds are fixed and the value goes to all six labels", () => {
    view();
    expect(screen.getByRole("heading", { level: 2, name: "Try a probability" })).toBeTruthy();
    expect(screen.queryByText("Try the thresholds")).toBeNull();
    const caption = (screen.getByRole("heading", { level: 2, name: "Try a probability" }).nextElementSibling as HTMLElement).textContent ?? "";
    expect(caption).toContain("The thresholds are fixed.");
    expect(caption).toContain("the same value is applied to all six labels");
    expect(caption).toContain("Each system has its own thresholds, fitted on validation data only");
    expect(document.querySelector(".xscrub .lbl")?.textContent).toBe("Probabilityapplied to every label");
  });

  it("says which labels have no HIGH threshold for the selected system, from the thresholds, and why", () => {
    view();
    const action = () => (document.querySelector(".zlegend p.zx") as HTMLElement).textContent ?? "";
    for (const [system, name] of [["logreg_word", "Logistic regression"], ["distilbert", "DistilBERT"], ["jev", "JEV"]] as const) {
      fireEvent.click(screen.getByRole("tab", { name }));
      const none = thresholds.filter((item) => item.system === system && !item.action_enabled).map((item) => item.label.replace(/_/g, " "));
      const text = action();
      if (none.length === 0) expect(text).toContain("Every label has a HIGH threshold for this system.");
      else {
        const list = none.length === 1 ? none[0] : `${none.slice(0, -1).join(", ")} and ${none[none.length - 1]}`;
        expect(text.toLowerCase()).toContain(`${list} ${none.length === 1 ? "has" : "have"} no high threshold for this system: no threshold met that.`);
      }
      expect(text).not.toContain("Some labels never reach it");
    }
  });

  it("never shows Action for a label without automatic action, at any position", () => {
    view();
    const disabled = thresholds.filter((item) => item.system === "logreg_word" && !item.action_enabled).map((item) => item.label);
    expect(disabled.length).toBeGreaterThan(0);
    for (const position of [0, 250, 500, 750, 999, 1000]) {
      move(position);
      for (const label of disabled) expect(row(label).dataset["decision"]).not.toBe("ACTION");
    }
  });

  it("decides each row with the shared rule at every position", () => {
    view();
    for (const position of [0, 100, 224, 300, 500, 640, 800, 1000]) {
      move(position);
      const p = probabilityFromSlider(position);
      for (const item of thresholds.filter((entry) => entry.system === "logreg_word")) {
        const expected = decideLabel(p, item.low ?? 0, item.action_enabled ? item.high : null);
        expect(row(item.label).dataset["decision"], `${item.label} at ${position}`).toBe(expected);
      }
    }
  });

  it("marks only the hit zone of each row and puts the marker on the square-root scale", () => {
    view();
    move(500);
    const marker = row("toxic").querySelector(".mk") as HTMLElement;
    expect(marker.style.left).toBe("50%");
    const hits = row("toxic").querySelectorAll(".zs .hit");
    expect(hits).toHaveLength(1);
  });

  it("switches system with the segmented control and with the arrow keys", () => {
    view();
    const group = screen.getByRole("tablist", { name: "System" });
    const tabs = within(group).getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual(SYSTEM_IDS.map((id) => SYSTEM_NAMES[id]));
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");

    move(1000);
    expect(row("identity_hate").dataset["decision"]).toBe("HUMAN REVIEW");
    fireEvent.click(tabs[1] as HTMLElement);
    expect(tabs[1]?.getAttribute("aria-selected")).toBe("true");
    const distilbertIdentity = thresholds.find((item) => item.system === "distilbert" && item.label === "identity_hate");
    expect(row("identity_hate").dataset["decision"]).toBe(distilbertIdentity?.action_enabled ? "ACTION" : "HUMAN REVIEW");

    fireEvent.keyDown(group, { key: "ArrowRight" });
    expect(within(group).getAllByRole("tab")[2]?.getAttribute("aria-selected")).toBe("true");
    fireEvent.keyDown(group, { key: "ArrowLeft" });
    expect(within(group).getAllByRole("tab")[1]?.getAttribute("aria-selected")).toBe("true");
  });

  it("lets a drag across a bar set the probability", () => {
    view();
    const bar = row("toxic").querySelector(".xbar") as HTMLElement;
    bar.getBoundingClientRect = () => ({ left: 100, width: 200, right: 300, top: 0, bottom: 10, height: 10, x: 100, y: 0, toJSON: () => ({}) });
    fireEvent.pointerDown(bar, { clientX: 200, pointerId: 1 });
    expect(slider().value).toBe("500");
    fireEvent.pointerMove(bar, { clientX: 300, pointerId: 1 });
    expect(slider().value).toBe("1000");
    fireEvent.pointerUp(bar, { pointerId: 1 });
    fireEvent.pointerMove(bar, { clientX: 100, pointerId: 1 });
    expect(slider().value).toBe("1000");
  });

  it("prints LOW and HIGH from the API, and under each label says Can trigger action or Cannot trigger action", () => {
    view();
    for (const item of thresholds.filter((entry) => entry.system === "logreg_word")) {
      const scale = row(item.label).querySelector(".xsc") as HTMLElement;
      expect(scale.textContent).toContain(`LOW ${f4(item.low ?? 0)}`);
      if (item.action_enabled && item.high !== null) expect(scale.textContent).toContain(`HIGH ${f4(item.high)}`);
      else expect(scale.textContent).toContain("no HIGH threshold");
      expect(row(item.label).textContent).toContain(item.action_enabled ? "Can trigger action" : "Cannot trigger action");
    }
  });

  it("states the targets from the results rather than from the page", () => {
    view();
    const target = results.policy_per_label?.[0]?.recall_target ?? 0;
    expect(screen.getByText(new RegExp(`at least ${Math.round(target * 100)}% of true positives`))).toBeTruthy();
  });

  it("states the support floor in the Action definition, as the API gives it, not as a number in the page", () => {
    view({ results: { ...results, policy_fit: { support_floor: 77 } } });
    const action = document.querySelector(".zlegend p.zx") as HTMLElement;
    expect(action.textContent).toContain("was at least 90% with at least 77 comments");
    expect(action.textContent).not.toContain("30");
  });

  it("falls back to wording without a number when the API does not give the support floor", () => {
    const { policy_fit: _fit, ...rest } = results;
    view({ results: rest as ResultsPayload });
    const action = document.querySelector(".zlegend p.zx") as HTMLElement;
    expect(action.textContent).toContain("with enough comments");
    expect(action.textContent).not.toMatch(/at least d+ comments/);
  });

  it("says the data is not available when the thresholds could not be loaded", () => {
    view({ thresholds: null });
    expect(document.querySelector(".explore .na-box")?.textContent).toContain("Data not available");
    expect(outcome().textContent).toContain("Thresholds not loaded");
  });
});

describe("where the comments went", () => {
  it("draws one bar per system with the share each decision got", () => {
    view();
    const summary = results.policy_summary ?? [];
    for (const id of SYSTEM_IDS) {
      const data = summary.find((item) => item.system === id);
      const flow = document.querySelector(`.fl[data-system="${id}"]`) as HTMLElement;
      const total = (data?.n_allow ?? 0) + (data?.n_review ?? 0) + (data?.n_action ?? 0);
      const parts = [...flow.querySelectorAll<HTMLElement>(".fl-bar > div")];
      expect(parts.map((part) => part.dataset["flow"])).toEqual(["ALLOW", "HUMAN REVIEW", "ACTION"]);
      const widths = parts.map((part) => Number.parseFloat(part.style.width));
      expect(widths.reduce((sum, value) => sum + value, 0)).toBeCloseTo(100, 6);
      expect(widths[0]).toBeCloseTo(((data?.n_allow ?? 0) / total) * 100, 6);
      expect(flow.textContent).toContain(`Human review ${formatCount(data?.n_review ?? 0)}`);
      expect(flow.textContent).toContain(`Action ${formatCount(data?.n_action ?? 0)}`);
      // a printed percentage only where the segment is wide enough to hold it
      parts.forEach((part, index) => {
        const printed = part.textContent !== "";
        expect(printed).toBe((widths[index] ?? 0) >= FLOW_LABEL_MIN_PERCENT);
      });
    }
  });

  it("states the number of test comments in the lead from the data", () => {
    view();
    const total = results.policy_summary?.[0]?.n_rows ?? 0;
    expect(screen.getByText(new RegExp(`All ${formatCount(total)} labelled test comments`))).toBeTruthy();
  });
});

describe("the five measures", () => {
  it("shows five rows with the value of each system and the best one marked", () => {
    view();
    const rows = [...document.querySelectorAll<HTMLElement>("#metrics .mr")];
    expect(rows).toHaveLength(5);
    for (const measure of FIVE_MEASURES) {
      const element = rows.find((item) => item.dataset["measure"] === measure.key) as HTMLElement;
      const values = measureValues(results.policy_summary ?? [], measure.key, SYSTEM_IDS);
      const best = bestIndex(values, measure.better);
      const texts = [...element.querySelectorAll(".vc")].map((cell) => cell.textContent);
      expect(texts).toEqual(values.map((value) => (value ?? 0).toFixed(3)));
      const flagged = [...element.querySelectorAll<HTMLElement>(".vc")].findIndex((cell) => cell.dataset["best"] === "true");
      expect(flagged).toBe(best);
      expect(element.querySelectorAll(".pd.best")).toHaveLength(1);
      expect(element.querySelectorAll(".pd")).toHaveLength(3);
    }
  });

  it("puts each dot at its value as a fraction of the axis", () => {
    view();
    const element = document.querySelector('#metrics .mr[data-measure="m1_review_share"]') as HTMLElement;
    const dots = [...element.querySelectorAll<HTMLElement>(".pd")];
    const fractions = dots.map((dot) => Number.parseFloat(dot.style.getPropertyValue("--f")));
    const values = measureValues(results.policy_summary ?? [], "m1_review_share", SYSTEM_IDS);
    // every value is on the axis, and the order of the dots follows the order of the values
    expect(Math.max(...fractions)).toBeLessThanOrEqual(1);
    expect(fractions.map((value, index) => [value, values[index] ?? 0])).toSatisfy((pairs: number[][]) =>
      pairs.every(([a = 0, b = 0], i) => pairs.every(([c = 0, d = 0], j) => i === j || (a < c) === (b < d))),
    );
  });
});

describe("how JEV compares", () => {
  it("shows a row per measure with a plotted line for each baseline, flipped so left means JEV is worse", () => {
    view();
    const rows = [...document.querySelectorAll<HTMLElement>("#cmp .cr")];
    expect(rows.map((item) => item.dataset["measure"])).toEqual(COMPARISON_MEASURES.map((measure) => measure.metric));
    for (const element of rows) {
      expect(element.querySelectorAll(".lane")).toHaveLength(2);
      const measure = COMPARISON_MEASURES.find((item) => item.metric === element.dataset["measure"]);
      const first = element.querySelector(".nn p b")?.textContent ?? "";
      // JEV is worse on the review share and both miss rates, which flip to negative; it is more precise, which stays positive
      expect(first.startsWith(measure?.better === "lower" ? "−" : "+")).toBe(true);
    }
  });

  it("places the dot left of the middle line exactly when the printed value is negative", () => {
    view();
    for (const element of document.querySelectorAll<HTMLElement>("#cmp .cr")) {
      const values = [...element.querySelectorAll(".nn p b")].map((b) => b.textContent ?? "");
      const dots = [...element.querySelectorAll<HTMLElement>(".dt")].map((dot) => Number.parseFloat(dot.style.left));
      values.forEach((text, index) => expect((dots[index] ?? 0) < 50).toBe(text.startsWith("−")));
    }
  });

  it("says the red side is JEV worse and counts the intervals from the data", () => {
    view();
    expect(screen.getByText("JEV worse")).toBeTruthy();
    expect(screen.getByText("JEV better")).toBeTruthy();
    expect(screen.getByText(/No interval touches the middle line\./)).toBeTruthy();
  });
});

describe("what validation promised and what the test set delivered", () => {
  it("shows the readings and the precision plot first", () => {
    view();
    expect(document.querySelectorAll(".reads p").length).toBeGreaterThanOrEqual(2);
    expect(document.getElementById("fp-precision")).toBeTruthy();
    expect(document.getElementById("fp-recall")).toBeNull();
    const precisionRows = agreementRows(results.policy_per_label ?? [], "precision");
    expect(document.querySelectorAll("#fp-precision .fp:not(.gh)")).toHaveLength(precisionRows.length);
  });

  it("switches to recall with every label for every system", () => {
    view();
    fireEvent.click(screen.getByRole("tab", { name: "Recall at the review threshold" }));
    expect(document.getElementById("fp-precision")).toBeNull();
    expect(document.querySelectorAll("#fp-recall .fp:not(.gh)")).toHaveLength(SYSTEM_IDS.length * LABELS.length);
    expect(document.querySelectorAll("#fp-recall .fp.gh")).toHaveLength(3);
  });

  it("only plots precision where automatic action is on", () => {
    const rows = agreementRows(results.policy_per_label ?? [], "precision");
    const enabled = (results.policy_per_label ?? []).filter((item) => item.action_enabled && item.test_action_precision !== null);
    expect(rows).toHaveLength(enabled.length);
  });

  it("draws the change from validation to test with its sign", () => {
    view();
    for (const element of document.querySelectorAll<HTMLElement>("#fp-precision .fp:not(.gh)")) {
      const key = element.dataset["row"] ?? "";
      const [system, label] = key.split("|");
      const data = results.policy_per_label?.find((item) => item.system === system && item.label === label);
      const change = (data?.test_action_precision ?? 0) - (data?.validation_precision_at_high ?? 0);
      const text = element.querySelector(".dl")?.textContent ?? "";
      expect(text.startsWith(change >= 0 ? "+" : "−")).toBe(true);
      expect(element.querySelector(".dl")?.classList.contains("up")).toBe(change >= 0);
    }
  });
});

describe("the trust paragraph", () => {
  it("names the targets from the data", () => {
    expect(trustText({ recall: 0.9, precision: 0.9 })).toContain("The 0.90 recall and precision targets are assumptions");
    expect(trustText({ recall: 0.9, precision: 0.95 })).toContain("0.90 recall and 0.95 precision targets");
    expect(trustText(null)).toContain("The recall and precision targets are assumptions");
  });

  it("is on the page, impersonal, and says what was fitted on what and evaluated on what", () => {
    view();
    expect(screen.getByText(/re-estimated on recent data/)).toBeTruthy();
    const text = document.querySelector(".finding .sub")?.textContent ?? "";
    expect(text).not.toMatch(/\bI\b|\bwe\b|\bour\b/i);
    expect(text).toContain("The cause of the difference is not established.");
    expect(text).toContain("fitted on the validation split of train.csv and applied once to the official test set.");
    expect(text).not.toContain("one data source");
    expect(text).not.toContain("did not hold");
    // only what the cards show, with the counts taken from the rows
    const rows = results.policy_per_label ?? [];
    const below = rows.filter((row) => (row.test_recall_at_low ?? 1) < (row.recall_target ?? 0)).length;
    expect(text).toContain("precision at the action threshold fell on toxic and obscene for all three systems");
    expect(text).toContain(`recall at the review threshold was below the 0.90 target on ${below} of ${rows.length} system-label pairs`);
  });

  it("makes no claim about precision or recall when there are no rows to show it", () => {
    expect(trustText({ recall: 0.9, precision: 0.9 }, [])).not.toContain("On the test set");
  });
});

describe("when data is missing", () => {
  it("shows Data not available for a chart whose section the API did not send", () => {
    const { policy_per_label: _dropped, ...rest } = results;
    view({ results: { ...rest, missing_sections: ["policy_per_label"] } });
    const section = screen.getByText("Validation against test results at the fitted thresholds").closest(".sec") as HTMLElement;
    expect(within(section).getByText(/Data not available/)).toBeTruthy();
    // the other charts are unaffected
    expect(document.querySelectorAll("#metrics .mr")).toHaveLength(5);
  });

  it("shows Data not available for every chart, with how to fix it, when the results could not be loaded", () => {
    view({ results: null });
    const boxes = [...document.querySelectorAll(".na-box")].map((box) => box.textContent ?? "");
    expect(boxes.filter((text) => text.includes("Data not available")).length).toBeGreaterThanOrEqual(4);
    expect(boxes.some((text) => text.includes("python -m policyjev.api"))).toBe(true);
  });

  it("shows a loading state while the results are on their way", () => {
    view({ results: undefined });
    expect(document.querySelectorAll('.na-box[data-state="loading"]').length).toBeGreaterThanOrEqual(1);
  });
});
