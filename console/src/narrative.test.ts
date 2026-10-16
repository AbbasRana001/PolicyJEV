import { describe, expect, it } from "vitest";

import { LABELS, SYSTEM_IDS, SYSTEM_NAMES, parseResults } from "./api";
import type { ApDifferenceRow, BudgetRow, PolicyLabelRow, PolicySummaryRow, ResultsPayload } from "./api";
import resultsFixture from "./fixtures/results.json";
import { formatCount, words } from "./format";
import {
  agreementReadings,
  apFinding,
  budgetFinding,
  budgetHeadline,
  budgetStatusNote,
  comparisonNote,
  automationFootnote,
  policyTargets,
  recallReading,
  thresholdsOnTestSentence,
  relation,
  summaryStatement,
} from "./narrative";

const payload = parseResults(resultsFixture) as ResultsPayload;
const summary = payload.policy_summary ?? [];
const perLabel = payload.policy_per_label ?? [];

function summaryRow(system: PolicySummaryRow["system"], values: Partial<PolicySummaryRow>): PolicySummaryRow {
  return {
    system,
    n_rows: 1000,
    n_allow: 0,
    n_review: 0,
    n_action: 0,
    m1_review_share: 0.1,
    m2_action_precision: 0.7,
    m3_miss_rate: 0.1,
    m4_rare_miss_rate: 0.1,
    action_coverage: 0.5,
    ...values,
  };
}

describe("summaryStatement", () => {
  it("names the system with the lowest review share and miss rate, taken from the data", () => {
    const lowest = (key: "m1_review_share" | "m3_miss_rate") =>
      [...summary].sort((a, b) => (a[key] ?? 1) - (b[key] ?? 1))[0]?.system;
    const result = summaryStatement(summary);
    expect(lowest("m1_review_share")).toBe(lowest("m3_miss_rate"));
    expect(result.headline).toContain(SYSTEM_NAMES[lowest("m1_review_share") ?? "jev"]);
    expect(result.headline).toContain("had the lowest review share and the lowest share of harmful comments allowed");
  });

  it("states the number of test comments from the summary rows", () => {
    expect(summaryStatement(summary).detail).toContain(`All ${formatCount(summary[0]?.n_rows ?? 0)} labelled test comments`);
  });

  it("says who needed the most review, who missed the most and who was most precise", () => {
    const rows = [
      summaryRow("logreg_word", { m1_review_share: 0.1, m3_miss_rate: 0.05, m2_action_precision: 0.6 }),
      summaryRow("distilbert", { m1_review_share: 0.05, m3_miss_rate: 0.04, m2_action_precision: 0.5 }),
      summaryRow("jev", { m1_review_share: 0.2, m3_miss_rate: 0.12, m2_action_precision: 0.8 }),
    ];
    const result = summaryStatement(rows);
    expect(result.headline).toBe("On the test set, DistilBERT had the lowest review share and the lowest share of harmful comments allowed");
    expect(result.detail).toContain("JEV needed the most review and missed the most. Its automatic actions had the highest precision.");
  });

  it("names each system separately when the extremes fall on different systems", () => {
    const rows = [
      summaryRow("logreg_word", { m1_review_share: 0.05, m3_miss_rate: 0.2, m2_action_precision: 0.9 }),
      summaryRow("distilbert", { m1_review_share: 0.2, m3_miss_rate: 0.04, m2_action_precision: 0.5 }),
      summaryRow("jev", { m1_review_share: 0.1, m3_miss_rate: 0.1, m2_action_precision: 0.6 }),
    ];
    const result = summaryStatement(rows);
    expect(result.headline).toBe(
      "On the test set, Logistic regression had the lowest review share and DistilBERT the lowest share of harmful comments allowed",
    );
    expect(result.detail).toContain("DistilBERT needed the most review and Logistic regression missed the most.");
    expect(result.detail).toContain("Logistic regression's automatic actions had the highest precision.");
  });

  it("falls back to a plain heading when the measures are missing", () => {
    expect(summaryStatement([]).headline).toBe("Policy results on the test set");
  });
});

describe("comparisonNote", () => {
  it("says no interval touches the middle line when none does", () => {
    expect(comparisonNote([{ difference: -0.1, low: -0.2, high: -0.05 }, { difference: 0.1, low: 0.05, high: 0.2 }])).toBe(
      "No interval touches the middle line.",
    );
  });

  it("counts the intervals that reach zero", () => {
    expect(comparisonNote([{ difference: -0.1, low: -0.2, high: 0 }, { difference: 0.1, low: 0.05, high: 0.2 }])).toBe(
      "1 of 2 intervals touch the middle line.",
    );
  });
});

describe("agreementReadings on the real data", () => {
  const readings = agreementReadings(perLabel);
  const enabled = perLabel.filter(
    (row) => row.action_enabled && row.validation_precision_at_high !== null && row.test_action_precision !== null,
  );
  const fellLabels = [...new Set(enabled.map((row) => row.label))].filter((label) =>
    enabled.filter((row) => row.label === label).every((row) => (row.test_action_precision ?? 0) < (row.validation_precision_at_high ?? 0)),
  );

  it("names exactly the labels whose precision fell for every system that has automatic action", () => {
    const first = readings[0];
    expect(first?.tone).toBe("bad");
    for (const label of LABELS) {
      expect(first?.text.includes(words(label))).toBe(fellLabels.includes(label));
    }
  });

  it("prints the range of validation and of test precision it describes, with no rounding to 'about'", () => {
    const pick = (field: "validation_precision_at_high" | "test_action_precision") =>
      enabled.filter((row) => fellLabels.includes(row.label)).map((row) => row[field] ?? 0);
    const range = (values: number[]) => `${Math.min(...values).toFixed(2)} to ${Math.max(...values).toFixed(2)}`;
    expect(readings[0]?.text).toContain(`${range(pick("validation_precision_at_high"))} on validation, ${range(pick("test_action_precision"))} on test`);
    expect(readings[0]?.text).not.toContain("about");
  });
});

describe("recallReading", () => {
  const reading = recallReading(perLabel);

  it("gives the exact sentence for the committed results (the Recall at LOW table in docs/results_policy.md)", () => {
    expect(reading?.text).toBe(
      "Recall at the review threshold was below the 0.90 target on 1 of 6 labels for logistic regression (lowest 0.895, obscene), 2 of 6 for DistilBERT (lowest 0.868, obscene) and 5 of 6 for JEV (lowest 0.768, insult).",
    );
    expect(reading?.tone).toBe("bad");
  });

  it("counts, for each system, the labels whose test recall is strictly below the recall target", () => {
    for (const system of SYSTEM_IDS) {
      const rows = perLabel.filter((row) => row.system === system);
      const below = rows.filter((row) => (row.test_recall_at_low ?? 1) < (row.recall_target ?? 0)).length;
      const lowest = [...rows].sort((a, b) => (a.test_recall_at_low ?? 1) - (b.test_recall_at_low ?? 1))[0];
      expect(reading?.text).toContain(`${below} of ${rows.length}`);
      expect(reading?.text).toContain(`(lowest ${(lowest?.test_recall_at_low ?? 0).toFixed(3)}, ${words(lowest?.label ?? "")})`);
    }
  });

  function row(values: Partial<PolicyLabelRow>): PolicyLabelRow {
    return {
      system: "jev",
      label: "toxic",
      positives: 10,
      action_enabled: true,
      recall_target: 0.9,
      precision_target: 0.9,
      validation_recall_at_low: 0.9,
      test_recall_at_low: 0.9,
      validation_precision_at_high: 0.9,
      test_action_precision: 0.9,
      ...values,
    };
  }

  it("uses no margin: a recall equal to the target is not below it, and one a hair under is", () => {
    const equal = recallReading([row({ test_recall_at_low: 0.9 })]);
    expect(equal?.text).toContain("below the 0.90 target on 0 of 1 labels for JEV (lowest 0.900, toxic)");
    expect(equal?.tone).toBe("good");
    const under = recallReading([row({ test_recall_at_low: 0.9 - 1e-9 })]);
    expect(under?.text).toContain("on 1 of 1 labels for JEV");
    expect(under?.tone).toBe("bad");
  });

  it("gives the lowest value of a system even when none is below the target", () => {
    const text = recallReading([row({ label: "toxic", test_recall_at_low: 0.95 }), row({ label: "insult", test_recall_at_low: 0.92 })])?.text;
    expect(text).toContain("0 of 2 labels for JEV (lowest 0.920, insult)");
  });

  it("reads the target from the rows, and says 'its target' when the rows disagree", () => {
    expect(recallReading([row({ recall_target: 0.95, test_recall_at_low: 0.93 })])?.text).toContain("below the 0.95 target on 1 of 1");
    const mixed = recallReading([row({ recall_target: 0.95 }), row({ label: "insult", recall_target: 0.9 })]);
    expect(mixed?.text).toContain("below its target on");
  });

  it("is null when there is no test recall", () => {
    expect(recallReading([])).toBeNull();
    expect(recallReading([row({ test_recall_at_low: null })])).toBeNull();
  });
});

describe("agreementReadings on other data", () => {
  function row(values: Partial<PolicyLabelRow>): PolicyLabelRow {
    return {
      system: "jev",
      label: "toxic",
      positives: 10,
      action_enabled: true,
      recall_target: 0.9,
      precision_target: 0.9,
      validation_recall_at_low: 0.9,
      test_recall_at_low: 0.9,
      validation_precision_at_high: 0.9,
      test_action_precision: 0.9,
      ...values,
    };
  }

  it("says everything held when nothing moved", () => {
    const readings = agreementReadings([row({}), row({ system: "distilbert" })]);
    expect(readings.every((reading) => reading.tone === "good")).toBe(true);
    expect(readings[0]?.text).toContain("Toxic held where automatic action is on");
  });

  it("calls any drop a fall and no drop a hold, with no tolerance", () => {
    const equal = agreementReadings([row({ test_action_precision: 0.9 })]);
    expect(equal[0]?.tone).toBe("good");
    const hairUnder = agreementReadings([row({ test_action_precision: 0.9 - 1e-9 })]);
    expect(hairUnder[0]?.tone).toBe("bad");
    expect(hairUnder[0]?.text).toContain("fell on toxic");
  });

  it("reports a label where systems disagree instead of hiding it", () => {
    const readings = agreementReadings([
      row({ system: "jev", test_action_precision: 0.5 }),
      row({ system: "distilbert", test_action_precision: 0.95 }),
    ]);
    expect(readings.some((reading) => reading.text.includes("fell for some systems and held for others"))).toBe(true);
  });
});

describe("relation", () => {
  it("is ahead when the whole interval is above zero, behind when below, level when it reaches zero", () => {
    expect(relation(0.01, 0.1)).toBe("ahead");
    expect(relation(-0.1, -0.01)).toBe("behind");
    expect(relation(-0.1, 0.1)).toBe("level");
    expect(relation(0, 0.1)).toBe("level");
    expect(relation(-0.1, 0)).toBe("level");
    expect(relation(null, 0.1)).toBeNull();
  });
});

function diff(label: ApDifferenceRow["label"], contrast: string, lo: number, hi: number): ApDifferenceRow {
  return { label, contrast, difference: (lo + hi) / 2, lo, hi };
}

describe("apFinding", () => {
  it("puts every label in exactly one sentence on the real data", () => {
    const text = apFinding(payload.ap_differences ?? [], LABELS);
    for (const label of LABELS) {
      const pattern = new RegExp(`\\b${words(label)}\\b`, "g");
      // severe toxic contains the word toxic, so count whole phrases for the label that is a suffix of another
      const matches = text.match(pattern)?.length ?? 0;
      const inLonger = label === "toxic" ? (text.match(/severe toxic/g)?.length ?? 0) : 0;
      expect(matches - inLonger).toBe(1);
    }
  });

  it("groups labels by how JEV compares with both baselines", () => {
    const rows: ApDifferenceRow[] = [
      diff("toxic", "jev - logreg_word", -0.2, -0.1),
      diff("toxic", "jev - distilbert", -0.3, -0.2),
      diff("obscene", "jev - logreg_word", 0.1, 0.2),
      diff("obscene", "jev - distilbert", 0.05, 0.1),
      diff("threat", "jev - logreg_word", -0.1, 0.1),
      diff("threat", "jev - distilbert", -0.05, 0.2),
      diff("insult", "jev - logreg_word", -0.1, 0.1),
      diff("insult", "jev - distilbert", -0.3, -0.1),
    ];
    const text = apFinding(rows, ["toxic", "obscene", "threat", "insult"]);
    expect(text).toContain("JEV trails both baselines on toxic.");
    expect(text).toContain("JEV leads both baselines on obscene.");
    expect(text).toContain("On threat the test set cannot separate it from either baseline.");
    expect(text).toContain("On insult it is level with logistic regression and behind DistilBERT.");
  });
});

describe("budgetFinding and budgetHeadline", () => {
  function budget(level: number, comparison: string, lo: number, hi: number): BudgetRow {
    return { budget_pct: level, metric: "recall_rare", comparison, value: (lo + hi) / 2, lo, hi, status: "descriptive" };
  }
  const rows: BudgetRow[] = [
    budget(1, "jev - logreg_word", 0.05, 0.1),
    budget(1, "jev - distilbert", 0.1, 0.2),
    budget(2, "jev - logreg_word", 0.02, 0.1),
    budget(2, "jev - distilbert", 0.1, 0.2),
    budget(5, "jev - logreg_word", -0.04, 0.02),
    budget(5, "jev - distilbert", 0.04, 0.1),
    budget(10, "jev - logreg_word", -0.06, -0.01),
    budget(10, "jev - distilbert", -0.07, -0.02),
  ];

  it("groups neighbouring budgets that share the same result", () => {
    expect(budgetFinding(rows, "recall_rare")).toBe(
      "JEV leads both baselines at 1% and 2%. At 5% it is level with logistic regression and ahead of DistilBERT. It trails both baselines at 10%.",
    );
  });

  it("states the lead in the heading only when JEV is ahead of both at the smallest budget", () => {
    expect(budgetHeadline(rows, "recall_rare")).toBe("JEV has higher rare-label recall only at the smallest review budgets");
    const behind = rows.map((row) => (row.budget_pct === 1 ? { ...row, lo: -0.2, hi: -0.1 } : row));
    expect(budgetHeadline(behind, "recall_rare")).toBe("Recall of rare-label comments as the review budget grows");
  });

  it("states from the status column which budgets were not pre-specified, and nothing when the statuses differ", () => {
    const real = payload.review_budget ?? [];
    expect(budgetStatusNote(real, "recall_rare")).toBe(
      "The 1% and 2% comparisons were not pre-specified: against logistic regression the difference is descriptive, and against DistilBERT it is a hypothesis generated on validation. Only the 5% and 10% contrasts were pre-specified.",
    );
    expect(budgetStatusNote(rows, "recall_rare")).toBe("");
  });

  it("mentions every budget of the real data once", () => {
    const real = payload.review_budget ?? [];
    const levels = [...new Set(real.filter((row) => row.metric === "recall_rare").map((row) => row.budget_pct))];
    const text = budgetFinding(real, "recall_rare");
    for (const level of levels) expect(text.match(new RegExp(`\\b${level}%`, "g"))?.length).toBe(1);
  });
});

describe("policyTargets", () => {
  it("reads the targets from the rows", () => {
    expect(policyTargets(perLabel)).toEqual({ recall: 0.9, precision: 0.9 });
  });

  it("is null without rows", () => {
    expect(policyTargets(undefined)).toBeNull();
    expect(policyTargets([])).toBeNull();
  });
});

describe("wording that must stay true for everything it covers", () => {
  const sets = (system: PolicyLabelRow["system"], enabled: string[]): PolicyLabelRow[] =>
    ["toxic", "obscene", "insult"].map((label) => ({
      system, label: label as PolicyLabelRow["label"], positives: 1, action_enabled: enabled.includes(label),
      recall_target: 0.9, precision_target: 0.9, validation_recall_at_low: 0.9, test_recall_at_low: 0.9,
      validation_precision_at_high: enabled.includes(label) ? 0.9 : null, test_action_precision: enabled.includes(label) ? 0.8 : null,
    }));

  it("says the extremes are point estimates and that the two baselines were never compared with each other", () => {
    const detail = summaryStatement(summary).detail;
    expect(detail).toContain("Only JEV was compared with each baseline; the difference between logistic regression and DistilBERT was not tested.");
    expect(summaryStatement(summary).headline).toMatch(/had the lowest review share/);
  });

  it("adds that systems automate different labels only when they do, from the data", () => {
    expect(summaryStatement(summary, perLabel).detail).toContain("but each system automates a different set of labels");
    const same = [...sets("logreg_word", ["toxic"]), ...sets("distilbert", ["toxic"]), ...sets("jev", ["toxic"])];
    expect(summaryStatement(summary, same).detail).not.toContain("different set of labels");
  });

  it("names each system's automatic-action labels, derived from the rows", () => {
    const text = automationFootnote(perLabel) ?? "";
    for (const system of SYSTEM_IDS) {
      const labels = perLabel.filter((row) => row.system === system && row.action_enabled).map((row) => words(row.label));
      expect(text).toContain(labels.length === 1 ? labels[0] : `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`);
    }
    expect(text).toContain("not like-for-like");
    const same = [...sets("logreg_word", ["toxic"]), ...sets("distilbert", ["toxic"])];
    expect(automationFootnote(same)).toBeNull();
  });

  it("states how the thresholds held on the test set from the precision rows (one sentence for the Score page)", () => {
    const sentence = thresholdsOnTestSentence(perLabel);
    expect(sentence).toBe("Thresholds were fitted on validation data, and its validation targets were missed on the test set: precision of automatic action fell on toxic and obscene for all three systems.");
    expect(thresholdsOnTestSentence(undefined)).toContain("shown on the Policy page");
    expect(thresholdsOnTestSentence([])).not.toContain("missed");
  });

  const budgetRows = (leads: boolean[]): BudgetRow[] =>
    leads.flatMap((ahead, index) =>
      ["jev - logreg_word", "jev - distilbert"].map((comparison) => ({
        budget_pct: [1, 2, 5][index] ?? 10, metric: "recall_rare", comparison, value: 0, lo: ahead ? 0.01 : -0.05, hi: ahead ? 0.1 : 0.05, status: "",
      })),
    );

  it("claims a lead at every budget only if there is one at every budget, and only at the smallest if it ends", () => {
    expect(budgetHeadline(budgetRows([true, true, true]), "recall_rare")).toBe("JEV has higher rare-label recall at every review budget");
    expect(budgetHeadline(budgetRows([true, true, false]), "recall_rare")).toBe("JEV has higher rare-label recall only at the smallest review budgets");
    expect(budgetHeadline(budgetRows([false, true, true]), "recall_rare")).toBe("Recall of rare-label comments as the review budget grows");
    expect(budgetHeadline(budgetRows([true, false, true]), "recall_rare")).toBe("Recall of rare-label comments as the review budget grows");
  });

  it("gives the exact unrounded recall behind '1 of 6' for logistic regression: insult is above the target, though it is the closest", () => {
    const insult = perLabel.find((row) => row.system === "logreg_word" && row.label === "insult");
    expect(insult?.validation_recall_at_low).toBeGreaterThan(insult?.test_recall_at_low ?? 1);
    expect(insult?.test_recall_at_low).toBeGreaterThanOrEqual(insult?.recall_target ?? 0);
    expect(recallReading(perLabel)?.text).toContain("1 of 6 labels for logistic regression");
  });
});
