// S10 failure states, S11 responsive layout.
import { sleep } from "../lib/harness.mjs";
import { SENTENCE, box, scoreButton, settleValues, tab, waitForScoreResponses } from "../lib/kit.mjs";
import { findLayoutProblems, readExplorer, readScoreUi } from "../lib/extract.mjs";

const waitForAlert = (page, pattern, timeout = 10_000) =>
  page.waitForFunction((source) => new RegExp(source).test(document.querySelector("#view-score .alert")?.textContent ?? ""), pattern.source, { timeout });

export const s10 = {
  id: "S10",
  title: "Failure states: 500, timeout, invalid JSON, results unavailable",
  async run(s, ctx) {
    const { kit, servers } = ctx;

    // /api/score fails in three ways, and the next Score works
    {
      const { page, close } = await kit.open(servers.a, { tag: "s10-score", expected: true, clock: true });
      await page.getByRole("switch", { name: "Score as I type" }).click();
      await box(page).fill(SENTENCE);

      // 500
      await page.route("**/api/score", (route) => route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ detail: "internal error" }) }));
      await scoreButton(page).click();
      await waitForAlert(page, /unexpected response/);
      let ui = await page.evaluate(readScoreUi);
      s.check(/unexpected response/.test(ui.alert ?? "") && ui.scoreDisabled === false, "a 500 shows the error box and leaves Score usable", `"${ui.alert}"`);
      s.check(Object.values(ui.tiles).every((t) => t.kind !== "scored"), "no verdict is shown for a failed request", Object.values(ui.tiles).map((t) => t.kind).join(", "));
      await page.unroute("**/api/score");
      const seenBefore = page.scores.responses.length;
      await scoreButton(page).click();
      await waitForScoreResponses(page, seenBefore + 1);
      await settleValues(page);
      ui = await page.evaluate(readScoreUi);
      s.check(ui.alert === null && ui.tiles.logreg_word.kind === "scored", "the next Score recovers: error box gone, verdicts shown", `alert=${ui.alert}, tile ${ui.tiles.logreg_word.kind}`);

      // invalid JSON
      await page.route("**/api/score", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "{this is not json" }));
      await scoreButton(page).click();
      await waitForAlert(page, /unexpected response/);
      ui = await page.evaluate(readScoreUi);
      s.check(/unexpected response/.test(ui.alert ?? ""), "invalid JSON shows the error box", `"${ui.alert}"`);
      await page.unroute("**/api/score");
      await scoreButton(page).click();
      await page.waitForFunction(() => !document.querySelector("#view-score .alert"));
      ui = await page.evaluate(readScoreUi);
      s.check(ui.alert === null, "and the next Score recovers", `alert=${ui.alert}`);

      // a response of the wrong shape
      await page.route("**/api/score", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ results: "nope", issues: [] }) }));
      await scoreButton(page).click();
      await waitForAlert(page, /unexpected response/);
      ui = await page.evaluate(readScoreUi);
      s.check(/unexpected response/.test(ui.alert ?? ""), "a JSON answer of the wrong shape shows the error box", `"${ui.alert}"`);
      await page.unroute("**/api/score");

      // timeout: the request never answers; the clock is moved past the limit
      const hung = [];
      await page.route("**/api/score", (route) => {
        hung.push(route);
      });
      await scoreButton(page).click();
      await page.waitForFunction(() => document.querySelector("#view-score button.primary")?.getAttribute("aria-busy") === "true");
      ui = await page.evaluate(readScoreUi);
      s.check(ui.scoreDisabled === true && ui.tiles.jev.kind === "unavailable", "while the request hangs Score is busy and disabled", `busy=${ui.scoreBusy}`);
      await page.clock.fastForward(31_000);
      await waitForAlert(page, /did not answer in time/);
      ui = await page.evaluate(readScoreUi);
      s.check(/did not answer in time/.test(ui.alert ?? "") && ui.scoreDisabled === false, "after the time limit the error box says the API did not answer in time and Score works again", `"${ui.alert}"`);
      await page.unroute("**/api/score");
      for (const route of hung) await route.abort().catch(() => {});
      await scoreButton(page).click();
      await page.waitForFunction(() => !document.querySelector("#view-score .alert"));
      await settleValues(page);
      ui = await page.evaluate(readScoreUi);
      s.check(ui.alert === null && ui.tiles.logreg_word.kind === "scored", "the next Score after the timeout recovers", `alert=${ui.alert}, tile ${ui.tiles.logreg_word.kind}`);

      // a failed local request while typing shows the box too
      await page.getByRole("switch", { name: "Score as I type" }).click();
      await page.route("**/api/score", (route) => route.abort());
      await box(page).fill(`${SENTENCE} More.`);
      await page.clock.fastForward(400);
      await waitForAlert(page, /Could not reach the API/);
      ui = await page.evaluate(readScoreUi);
      s.check(/Could not reach the API/.test(ui.alert ?? ""), "a request that fails while typing shows the Could not reach the API message", `"${ui.alert}"`);
      await close();
    }

    // /api/results fails: sections say Data not available
    for (const mode of ["aborted", "500"]) {
      const { page, close } = await kit.open(servers.a, {
        tag: `s10-results-${mode}`,
        expected: true,
        before: async (p) => {
          await p.route("**/api/results", (route) => (mode === "aborted" ? route.abort() : route.fulfill({ status: 500, contentType: "application/json", body: "{}" })));
        },
      });
      await tab(page, "policy");
      await sleep(600);
      const policy = await page.evaluate(() =>
        [...document.querySelectorAll("#view-policy .sec")].map((sec) => ({
          h2: sec.querySelector("h2")?.textContent.trim(),
          boxes: [...sec.querySelectorAll(".na-box")].map((b) => b.textContent.replace(/\s+/g, " ").trim()),
          text: sec.textContent.replace(/\s+/g, " ").trim().length,
        })),
      );
      const na = policy.filter((sec) => sec.boxes.length > 0);
      const boxes = na.reduce((sum, sec) => sum + sec.boxes.length, 0);
      s.check(na.length >= 3 && boxes >= 4 && na.every((sec) => sec.boxes.every((b) => b.startsWith("Data not available"))), `results ${mode}: the Policy charts say Data not available`, `${na.length} sections, ${boxes} boxes: ${na.map((sec) => sec.h2).join(" | ").slice(0, 160)}`);
      s.check(policy.every((sec) => sec.text > 20), `results ${mode}: no section is blank`, `shortest section has ${Math.min(...policy.map((sec) => sec.text))} characters`);
      const explorer = await page.evaluate(readExplorer);
      s.check(explorer.rows.length === 6, `results ${mode}: the explorer (which uses thresholds) still works`, `${explorer.rows.length} rows`);
      await tab(page, "results");
      await sleep(300);
      const results = await page.evaluate(() => [...document.querySelectorAll("#view-results .sec")].map((sec) => ({ h2: sec.querySelector("h2")?.textContent.trim(), na: sec.querySelector(".na-box")?.textContent.replace(/\s+/g, " ").trim() ?? null })));
      s.check(results.length === 2 && results.every((sec) => sec.na?.startsWith("Data not available")), `results ${mode}: both Results charts say Data not available`, results.map((sec) => sec.na).join(" | ").slice(0, 160));
      s.check(results.every((sec) => /python -m policyjev\.api/.test(sec.na ?? "")), `results ${mode}: and say how to start the API`);
      await close();
    }
  },
};

const WIDTHS = [360, 375, 768, 1024, 1440];
// widths where the tile heading used to wrap in one tile and not in the others
const TILE_WIDTHS = [1180, 1060, 900, 760];

/** Geometry of the three verdict tiles: heading boxes, and where the time sits under the name. */
const tileGeometry = () => {
  const box = (el) => {
    const r = el.getBoundingClientRect();
    return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height, width: r.width };
  };
  return [...document.querySelectorAll("#view-score .vt")].map((tile) => {
    const name = tile.querySelector(".nm b");
    const ms = tile.querySelector(".nm .ms");
    const word = tile.querySelector(".word");
    const span = word.querySelector("span");
    const lineHeight = parseFloat(getComputedStyle(span).lineHeight) || parseFloat(getComputedStyle(span).fontSize) * 1.05;
    return {
      system: tile.dataset.sys,
      tile: box(tile),
      name: box(name),
      ms: box(ms),
      msText: ms.textContent.trim(),
      word: box(word),
      wordLines: Math.round(span.getBoundingClientRect().height / lineHeight),
      wordText: span.textContent,
      wordFont: parseFloat(getComputedStyle(span).fontSize),
      nameScroll: name.scrollWidth <= name.clientWidth + 1,
    };
  });
};

export const s11tiles = {
  async run(s, ctx) {
    const { kit, servers } = ctx;
    for (const width of TILE_WIDTHS) {
      const { page, close } = await kit.open(servers.b, { width, height: 900, reducedMotion: "reduce", tag: `s11-tiles-${width}` });
      await box(page).fill("A calm sentence about the schedule.");
      await scoreButton(page).click();
      await page.waitForFunction(() => document.querySelector("#vt-jev")?.dataset.tile === "scored");
      await settleValues(page);
      const tiles = await page.evaluate(tileGeometry);
      const words = tiles.map((t) => `${t.system}: "${t.wordText}" on ${t.wordLines} line(s)`).join("; ");
      const heights = new Set(tiles.map((t) => Math.round(t.word.height)));
      s.check(heights.size === 1, `${width}px: the three tiles reserve the same heading height`, `heights ${[...heights].join(",")}; ${words}`);
      s.check(tiles.every((t) => Math.abs(t.ms.top - t.name.bottom) <= 2 && t.msText.endsWith("ms")), `${width}px: in every tile the time sits directly under the name`, tiles.map((t) => `${t.system} ${t.msText} ${Math.round(t.ms.top - t.name.bottom)}px below`).join("; "));
      const tops = new Set(tiles.map((t) => Math.round(t.word.top)));
      s.check(tops.size === 1, `${width}px: the headings start at the same height in all three tiles`, [...tops].join(","));
      const review = tiles.find((t) => t.wordText === "Human review");
      s.check(review !== undefined && review.word.height >= review.wordLines * review.wordFont, `${width}px: the Human review heading is not clipped`, review ? `${review.wordLines} line(s) at ${review.wordFont}px in a box ${Math.round(review.word.height)}px high` : "no tile shows Human review");
      // a long system name changes nothing about the layout
      const before = tiles.map((t) => [Math.round(t.tile.width), Math.round(t.word.top), Math.round(t.ms.top)]);
      await page.evaluate(() => { for (const el of document.querySelectorAll("#view-score .vt .fn, #view-score .vt .sn")) el.textContent = "A system with a very long name that cannot fit in this tile"; });
      const longer = await page.evaluate(tileGeometry);
      const after = longer.map((t) => [Math.round(t.tile.width), Math.round(t.word.top), Math.round(t.ms.top)]);
      s.check(JSON.stringify(before) === JSON.stringify(after) && longer.every((t) => t.name.right <= t.tile.right + 1), `${width}px: a very long system name does not move the heading or the time, or leave the tile`, `tile widths, heading and time tops unchanged: ${JSON.stringify(before) === JSON.stringify(after)}`);
      // the "Different labels per system" marker is on its own line, between the title and the description
      await tab(page, "policy");
      await page.waitForSelector("#metrics .mr");
      const markers = await page.evaluate(() =>
        [...document.querySelectorAll("#metrics .mr")].filter((mr) => mr.querySelector("[data-flag]")).map((mr) => {
          const r = (el) => el.getBoundingClientRect();
          const title = r(mr.querySelector(".mn b"));
          const flag = r(mr.querySelector(".mn .flag"));
          const description = r(mr.querySelector(".mn > span"));
          return { measure: mr.dataset.measure, flagBelowTitle: flag.top >= title.bottom - 1, descriptionBelowFlag: description.top >= flag.bottom - 1, sameLeft: Math.abs(flag.left - title.left) < 1 };
        }),
      );
      s.check(markers.length === 2 && markers.every((m) => m.flagBelowTitle && m.descriptionBelowFlag && m.sameLeft), `${width}px: the marker is on its own line under the measure title and above the description`, JSON.stringify(markers));
      await close();
    }
  },
};

export const s11 = {
  id: "S11",
  title: "Responsive: no sideways scroll and no clipped text at 360, 375, 768, 1024 and 1440 px",
  async run(s, ctx) {
    const { kit, servers, obs } = ctx;
    await s11tiles.run(s, ctx);
    obs.layout = [];
    for (const width of WIDTHS) {
      const touch = width <= 760;
      const { page, close } = await kit.open(servers.a, { width, height: 900, reducedMotion: "reduce", touch, tag: `s11-${width}` });
      const measure = async (scene) => {
        await sleep(250);
        const result = await page.evaluate(findLayoutProblems);
        obs.layout.push({ width, scene, ...result });
        s.check(
          result.scrollWidth <= result.innerWidth && result.bodyScrollWidth <= result.innerWidth && result.problems.length === 0,
          `${width}px, ${scene}: no sideways scroll and no clipped text`,
          `scrollWidth ${result.scrollWidth} of ${result.innerWidth}${result.problems.length ? `; ${result.problems.length} problems: ${result.problems.slice(0, 3).map((p) => `${p.element} "${p.text}" ${p.why}`).join(" | ")}` : ""}`,
        );
      };
      await measure("Score empty");
      await box(page).fill(SENTENCE);
      await scoreButton(page).click();
      await page.waitForFunction(() => document.querySelector("#vt-distilbert")?.dataset.tile === "scored");
      await settleValues(page);
      await measure("Score scored, JEV unavailable");
      s.check((await page.evaluate(readScoreUi)).tiles.jev.kind === "unavailable", `${width}px: JEV shows as unavailable`);

      await tab(page, "policy");
      await page.waitForSelector("#fp-precision .fp");
      await measure("Policy, Precision tab");
      await page.getByRole("tab", { name: "Recall at the review threshold" }).click();
      await page.waitForSelector("#fp-recall .fp");
      await measure("Policy, Recall tab");
      await tab(page, "results");
      await page.waitForSelector("#lc circle");
      await measure("Results");

      if (touch) {
        // tapping a verdict tile switches the meter column
        await tab(page, "score");
        for (const system of ["distilbert", "jev", "logreg_word"]) {
          await page.locator(`#vt-${system}`).tap();
          const shown = await page.evaluate(() => {
            const cells = [...document.querySelectorAll("#view-score .cell")].filter((c) => getComputedStyle(c).display !== "none");
            return { sel: document.querySelector("#view-score .matrix").dataset.sel, systems: [...new Set(cells.map((c) => c.dataset.sys))], count: cells.length };
          });
          s.check(shown.sel === system && shown.systems.length === 1 && shown.systems[0] === system && shown.count === 6, `${width}px: tapping the ${system} tile shows only its meter column`, `selected ${shown.sel}; columns shown: ${shown.systems.join(",")} (${shown.count} cells)`);
        }
        // the segmented control switches system in the explorer
        await tab(page, "policy");
        const lows = {};
        for (const [system, name] of [["distilbert", "DistilBERT"], ["jev", "JEV"], ["logreg_word", "Logistic regression"]]) {
          await page.locator('.explore [role="tab"]', { hasText: name }).tap();
          await page.waitForFunction((n) => document.querySelector('.explore [role="tab"][aria-selected="true"]')?.textContent.trim() === n, name);
          lows[system] = (await page.evaluate(readExplorer)).rows.map((row) => row.scale).join("|");
        }
        s.check(new Set(Object.values(lows)).size === 3, `${width}px: the segmented control changes the thresholds shown for each system`, `${Object.keys(lows).length} systems, ${new Set(Object.values(lows)).size} distinct threshold sets`);
        await measure("Policy after switching system");
      }
      await close();
    }
  },
};
