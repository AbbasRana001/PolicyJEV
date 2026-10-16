// S06 Policy numbers, S07 the explorer, S08 Results, S09 tabs keep state.
// The browser records what it shows; tests/e2e_check.py compares it with the committed CSVs and the engine.
import { getJson, sleep } from "../lib/harness.mjs";
import { LABELS, SENTENCE, SYSTEMS, box, tab, waitForScoreResponses } from "../lib/kit.mjs";
import { readExplorer, readPolicyCharts, readResultsPage } from "../lib/extract.mjs";

const SYSTEM_TABS = { logreg_word: "Logistic regression", distilbert: "DistilBERT", jev: "JEV" };

async function selectSystem(page, system) {
  await page.locator(`.explore [role="tab"]`, { hasText: SYSTEM_TABS[system] }).click();
  await page.waitForFunction((name) => document.querySelector('.explore [role="tab"][aria-selected="true"]')?.textContent.trim() === name, SYSTEM_TABS[system]);
}

export const s06 = {
  id: "S06",
  title: "Policy page numbers equal their sources (explorer thresholds, flow, five measures, comparison, dumbbells, sentences)",
  async run(s, ctx) {
    const { kit, servers, apis, obs } = ctx;
    obs.api = {
      results: (await getJson(`${apis.a.url}/api/results`)).body,
      thresholds: (await getJson(`${apis.a.url}/api/thresholds`)).body,
    };
    const { page, close } = await kit.open(servers.a, { tag: "s06" });
    await tab(page, "policy");
    await page.waitForSelector("#metrics .mr");
    await page.waitForSelector("#fp-precision .fp");
    await sleep(1300);
    obs.policy = { explorer: {} };
    for (const system of SYSTEMS) {
      await selectSystem(page, system);
      obs.policy.explorer[system] = await page.evaluate(readExplorer);
    }
    obs.policy.charts = await page.evaluate(readPolicyCharts);
    await page.getByRole("tab", { name: "Recall at the review threshold" }).click();
    await page.waitForSelector("#fp-recall .fp");
    obs.policy.chartsRecall = await page.evaluate(readPolicyCharts);

    const rows = SYSTEMS.reduce((sum, system) => sum + obs.policy.explorer[system].rows.length, 0);
    s.check(rows === 18, "the explorer shows six labels for each of the three systems", `${rows} system-label rows`);
    const c = obs.policy.charts;
    s.check(c.flows.length === 3 && c.measures.length === 5 && c.comparison.length === 4, "three flow bars, five measures and four comparison rows are drawn", `${c.flows.length} flows, ${c.measures.length} measures, ${c.comparison.length} comparison rows`);
    s.check(c.dumbbells.precision.length > 0 && obs.policy.chartsRecall.dumbbells.recall.length === 18, "both dumbbell plots are drawn (recall for all 18 system-label pairs)", `${c.dumbbells.precision.length} precision rows, ${obs.policy.chartsRecall.dumbbells.recall.length} recall rows`);
    s.check(c.reads.length >= 2 && c.sections.slice(0, 4).every((section) => section.h2), "the readings and the section headings are present", `${c.reads.length} readings`);
    s.check(c.headings.includes("Validation against test results at the fitted thresholds") && c.headings.includes("Try a probability"), "the explorer is headed Try a probability and the validation section is headed descriptively", c.headings.join(" | "));
    s.check(c.sliderLabel.startsWith("Probability") && c.sliderLabel.endsWith("applied to every label"), "the slider label says it is applied to every label", c.sliderLabel);
    s.check(c.pillNotes.every((n) => ["Can trigger action", "Cannot trigger action"].includes(n)) && c.pillNotes.length === 2, "the pill notes are Can trigger action and Cannot trigger action", c.pillNotes.join(" | "));
    s.check(!/\b[Ii]\b/.test(c.trust), "the paragraph on trust has no first person", c.trust.slice(0, 80));
    s.note("each displayed number is compared with policy_thresholds.csv, policy_test_summary.csv, policy_test_contrasts.csv and policy_test_per_label.csv by tests/e2e_check.py (S06 checks)");
    await close();
  },
};

/** The smallest slider position whose probability is at or above the threshold. */
export function positionAtOrAbove(threshold) {
  let v = Math.max(0, Math.floor(Math.sqrt(threshold) * 1000) - 2);
  while ((v / 1000) ** 2 < threshold) v += 1;
  return v;
}

export const s07 = {
  id: "S07",
  title: "Explorer: every system and label just below and at-or-above LOW and HIGH, by drag and by keyboard",
  async run(s, ctx) {
    const { kit, servers, obs } = ctx;
    const thresholds = obs.api?.thresholds ?? (await getJson(`${ctx.apis.a.url}/api/thresholds`)).body;
    const { page, close } = await kit.open(servers.a, { tag: "s07" });
    await tab(page, "policy");
    await page.waitForSelector(".explore .xr[data-l]");
    await sleep(900);

    // how far PageUp moves the slider here
    await page.focus("#probe");
    await page.keyboard.press("Home");
    await page.keyboard.press("PageUp");
    const pageStep = await page.evaluate(() => Number(document.querySelector("#probe").value));
    await page.keyboard.press("Home");
    s.note(`PageUp moves the slider by ${pageStep} positions; positions are whole numbers 0 to 1000 and the probability is (position / 1000) squared, so the probability can come within about 2e-3 of a threshold but a threshold is never hit exactly`);

    obs.explorerCases = [];
    const methods = { drag: 0, "drag+keys": 0, keys: 0 };
    let switchChecks = 0;
    // the explorer starts on logistic regression, so visit the others first and come back: every visit is a change
    for (const system of ["distilbert", "jev", "logreg_word"]) {
      // switching system rebuilds all six rows
      await page.evaluate(() => document.querySelectorAll(".explore .xr[data-l]").forEach((row) => { row.dataset.old = "1"; }));
      await selectSystem(page, system);
      const rebuilt = await page.evaluate(() => {
        const rows = [...document.querySelectorAll(".explore .xr[data-l]")];
        return { count: rows.length, stale: rows.filter((row) => row.dataset.old === "1").length };
      });
      switchChecks += 1;
      s.check(rebuilt.count === 6 && rebuilt.stale === 0, `switching to ${system} re-renders all six rows`, `${rebuilt.count} rows, ${rebuilt.stale} left over from before`);

      const rules = thresholds.filter((row) => row.system === system);
      let n = 0;
      for (const rule of rules) {
        const edges = [["low", rule.low]];
        if (rule.action_enabled && rule.high !== null) edges.push(["high", rule.high]);
        for (const [edge, threshold] of edges) {
          const atOrAbove = positionAtOrAbove(threshold);
          for (const [side, v] of [["below", atOrAbove - 1], ["at", atOrAbove]]) {
            if (v < 0) continue;
            // alternate the method: pointer drag on a bar (then keys to land on the exact position), or keys alone
            const useKeys = n % 3 === 2;
            let method;
            if (useKeys) {
              await page.focus("#probe");
              await page.keyboard.press("Home");
              for (let i = 0; i < Math.floor(v / pageStep); i += 1) await page.keyboard.press("PageUp");
              const rest = v - Math.floor(v / pageStep) * pageStep;
              for (let i = 0; i < rest; i += 1) await page.keyboard.press("ArrowRight");
              method = "keys";
            } else {
              const bar = page.locator(`.explore .xr[data-l="${rule.label}"] .xbar`);
              await bar.scrollIntoViewIfNeeded();
              const area = await bar.boundingBox();
              const y = area.y + area.height / 2;
              const x = area.x + (v / 1000) * area.width;
              await page.mouse.move(x, y);
              await page.mouse.down();
              await page.mouse.move(x + 0.01, y);
              await page.mouse.up();
              let now = await page.evaluate(() => Number(document.querySelector("#probe").value));
              method = "drag";
              if (now !== v) {
                await page.focus("#probe");
                for (let guard = 0; guard < 30 && now !== v; guard += 1) {
                  await page.keyboard.press(now < v ? "ArrowRight" : "ArrowLeft");
                  now = await page.evaluate(() => Number(document.querySelector("#probe").value));
                }
                method = "drag+keys";
              }
            }
            methods[method] += 1;
            n += 1;
            const ui = await page.evaluate(readExplorer);
            obs.explorerCases.push({ system, label: rule.label, edge, side, v, p: (v / 1000) ** 2, method, ui });
          }
        }
      }
    }
    const reached = obs.explorerCases.filter((c) => c.ui.slider === c.v).length;
    s.check(reached === obs.explorerCases.length, "every probe position was reached exactly", `${reached} of ${obs.explorerCases.length} cases`);
    s.check(obs.explorerCases.length >= 50, "all systems, labels and both thresholds were probed on both sides", `${obs.explorerCases.length} cases: ${methods.drag} by drag, ${methods["drag+keys"]} by drag then keys, ${methods.keys} by keys alone`);
    const reviewOnly = obs.explorerCases.flatMap((c) => c.ui.rows.filter((row) => /Cannot trigger action/.test(row.note ?? "")));
    s.check(reviewOnly.length > 0 && reviewOnly.every((row) => row.decision !== "ACTION" && row.chip !== "Action"), "review-only labels never show Action, at any probe", `${reviewOnly.length} review-only rows seen`);
    s.check(switchChecks === 3, "each system tab was switched to and checked");
    s.note("the expected chip of every row and the outcome capsule for every case come from src/policyjev/policy/engine.py in tests/e2e_check.py (S07 checks)");
    await close();
  },
};

export const s08 = {
  id: "S08",
  title: "Results page: AP bars, intervals and budget points equal the CSVs; sentences are true",
  async run(s, ctx) {
    const { kit, servers, obs } = ctx;
    const { page, close } = await kit.open(servers.a, { tag: "s08" });
    await tab(page, "results");
    await page.waitForSelector("#apc .apb");
    await page.waitForSelector("#lc circle");
    await sleep(1800);
    obs.results = await page.evaluate(readResultsPage);
    const r = obs.results;
    s.check(r.bars.length === 18 && r.groups.length === 6, "six labels with three bars each", `${r.bars.length} bars, ${r.groups.length} groups`);
    s.check(r.points.length === 15 && r.svg !== null, "five budget points for each of the three systems", `${r.points.length} points; chart ${r.svg?.width}x${r.svg?.height}`);
    const legend = ["Logistic regression", "DistilBERT", "JEV"];
    s.check(legend.every((name) => r.directLabels.some((l) => l.text.startsWith(`${name} `))), "the direct labels use the same names as the legend (full names)", r.directLabels.map((l) => l.text).join("; "));
    s.check(r.leaders.every((l) => r.points.some((p) => p.level === Math.min(...r.points.map((q) => q.level)) && p.system === l.system && Math.abs(p.cx - l.x1) < 0.01 && Math.abs(p.cy - l.y1) < 0.01)), "every leader line starts at its own system's first point", `${r.leaders.length} leader lines`);
    s.check(r.directLabels.length === 3, "all three lines are labelled at the smallest budget", r.directLabels.map((l) => l.text).join("; "));
    s.check(r.sections.length === 2 && r.sections.every((section) => section.h2 && section.sub), "both sections have a heading and a sentence", `${r.sections.length} sections`);
    s.note("bars, intervals, points and sentences are compared with test_comparison_with_stack.csv and review_budget_test_with_stack.csv by tests/e2e_check.py (S08 checks)");
    await close();
  },
};

export const s09 = {
  id: "S09",
  title: "Tabs keep typed text, scores, slider position and selections",
  async run(s, ctx) {
    const { kit, servers } = ctx;
    const { page, close } = await kit.open(servers.a, { tag: "s09" });
    await box(page).fill(SENTENCE);
    await waitForScoreResponses(page, 1);
    await tab(page, "policy");
    await page.waitForSelector(".explore .xr[data-l]");
    await page.locator("#probe").fill("777");
    await selectSystem(page, "distilbert");
    await page.getByRole("tab", { name: "Recall at the review threshold" }).click();
    await page.waitForSelector("#fp-recall .fp");
    await tab(page, "results");
    await sleep(200);
    await tab(page, "score");
    s.check((await box(page).inputValue()) === SENTENCE, "the typed text is still in the box after visiting Policy and Results");
    const tiles = await page.evaluate(() => [...document.querySelectorAll("#view-score .vt")].map((t) => `${t.dataset.sys}:${t.dataset.tile}`));
    s.check(tiles.includes("logreg_word:scored") && tiles.includes("distilbert:scored"), "the verdicts are still shown", tiles.join(", "));
    await tab(page, "policy");
    const state = await page.evaluate(() => ({
      slider: document.querySelector("#probe").value,
      valuetext: document.querySelector("#probe").getAttribute("aria-valuetext"),
      system: document.querySelector('.explore [role="tab"][aria-selected="true"]').textContent.trim(),
      recallShown: Boolean(document.querySelector("#fp-recall")),
    }));
    const expectedText = `probability ${((777 / 1000) ** 2).toFixed(4)}`;
    s.check(state.slider === "777" && state.valuetext === expectedText, `the slider is still at 777 (${expectedText})`, `${state.slider}, ${state.valuetext}`);
    s.check(state.system === "DistilBERT", "the chosen system is still DistilBERT", state.system);
    s.check(state.recallShown, "the Recall tab of the dumbbell plot is still the one shown");
    await close();
  },
};
