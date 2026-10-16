// S12 keyboard and focus, S13 reduced motion.
import { sleep } from "../lib/harness.mjs";
import { SENTENCE, box, scoreButton, settleValues, tab, waitForScoreResponses } from "../lib/kit.mjs";
import { describeFocus, listFocusable, readScoreUi } from "../lib/extract.mjs";

const ring = (focus) =>
  focus.outlineStyle !== "none" && parseFloat(focus.outlineWidth) >= 2 && !/rgba?\(0, 0, 0, 0\)/.test(focus.outlineColor);

/** Press Tab until the focus comes back to where it started (or 80 presses); return what was focused. */
async function tabAround(page) {
  const visited = [];
  // a click on empty space at the top left puts the Tab starting point before the first control
  await page.mouse.click(1, 1);
  for (let i = 0; i < 80; i += 1) {
    await page.keyboard.press("Tab");
    const focus = await page.evaluate(describeFocus);
    if (focus.body) {
      visited.push({ ...focus, leftThePage: true });
      break;
    }
    const key = `${focus.tag}#${focus.id}|${focus.role}|${focus.label}`;
    if (visited.length > 2 && visited.some((v) => v.key === key)) {
      visited.push({ ...focus, key, wrapped: true });
      break;
    }
    visited.push({ ...focus, key });
  }
  return visited;
}

export const s12 = {
  id: "S12",
  title: "Keyboard: Tab reaches every control, focus is visible, the slider works, no trap",
  async run(s, ctx) {
    const { kit, servers } = ctx;
    const { page, close } = await kit.open(servers.b, { tag: "s12", reducedMotion: "reduce" });
    // text in the box so that Score is enabled, and a scored result so everything is on screen
    await box(page).fill(SENTENCE);
    await waitForScoreResponses(page, 1);
    await settleValues(page);

    for (const view of ["score", "policy", "results"]) {
      await tab(page, view);
      if (view === "policy") await page.waitForSelector("#fp-precision .fp");
      if (view === "results") await page.waitForSelector("#lc circle");
      await sleep(200);
      const list = await page.evaluate(listFocusable);
      const reachableByTab = list.filter((el) => el.tabIndex >= 0 && !el.disabled);
      const visited = await tabAround(page);
      const wrapped = visited[visited.length - 1];
      s.check(wrapped.wrapped === true || wrapped.leftThePage === true, `${view}: Tab goes around and does not get stuck`, `${visited.length} stops before ${wrapped.wrapped ? "wrapping to the start" : "leaving the page"}`);

      const names = new Set(visited.map((v) => `${v.tag}|${v.role ?? ""}|${v.id ?? ""}|${v.label}`));
      const missed = reachableByTab.filter((el) => !names.has(`${el.tag}|${el.role ?? ""}|${el.id ?? ""}|${el.label}`));
      s.check(missed.length === 0, `${view}: Tab reaches every focusable control that is on screen`, `${reachableByTab.length} controls, ${missed.length} missed${missed.length ? `: ${missed.slice(0, 3).map((m) => m.label).join(", ")}` : ""}`);

      const noRing = visited.filter((v) => !v.body && !v.wrapped && !ring(v));
      s.check(noRing.length === 0, `${view}: every control shows a focus ring (outline of 2px or more) when it has focus`, `${visited.length} stops, ${noRing.length} without${noRing.length ? `: ${noRing.slice(0, 3).map((v) => `${v.tag} "${v.label}" ${v.outlineStyle} ${v.outlineWidth}`).join("; ")}` : ""}`);

      // the controls that are not in the Tab order (the unselected tabs of a tab list) are reached by arrow keys
      const lists = await page.evaluate(() => [...document.querySelectorAll('[role="tablist"]')].filter((tl) => !tl.closest("[hidden]") && tl.getBoundingClientRect().width > 0).map((tl) => ({ label: tl.getAttribute("aria-label"), tabs: [...tl.querySelectorAll('[role="tab"]')].map((t) => t.textContent.trim()) })));
      for (const group of lists.filter((g) => g.label !== "Sections")) {
        const seen = [];
        await page.locator(`[role="tablist"][aria-label="${group.label}"] [role="tab"][aria-selected="true"]`).focus();
        for (let i = 0; i < group.tabs.length; i += 1) {
          seen.push(await page.evaluate(() => document.activeElement.textContent.trim()));
          await page.keyboard.press("ArrowRight");
        }
        s.check(new Set(seen).size === group.tabs.length, `${view}: arrow keys reach every option of the "${group.label}" control`, seen.join(" > "));
      }
    }

    // the slider
    await tab(page, "policy");
    await page.waitForSelector("#probe");
    await page.focus("#probe");
    const read = () => page.evaluate(() => ({ value: Number(document.querySelector("#probe").value), p: document.querySelector("[data-probability]").textContent, valuetext: document.querySelector("#probe").getAttribute("aria-valuetext"), outcome: document.querySelector(".explore .outcome")?.dataset.outcome }));
    await page.keyboard.press("Home");
    const home = await read();
    await page.keyboard.press("End");
    const end = await read();
    await page.keyboard.press("Home");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    const three = await read();
    await page.keyboard.press("ArrowLeft");
    const two = await read();
    await page.keyboard.press("PageUp");
    const paged = await read();
    s.check(home.value === 0 && home.p === "0.0000" && home.outcome === "ALLOW", "Home sets the slider to 0 (probability 0.0000, outcome Allow)", `${home.value}, ${home.p}, ${home.outcome}`);
    s.check(end.value === 1000 && end.p === "1.0000" && end.outcome === "ACTION", "End sets it to 1000 (probability 1.0000, outcome Action)", `${end.value}, ${end.p}, ${end.outcome}`);
    s.check(three.value === 3 && two.value === 2, "ArrowRight moves up by one and ArrowLeft down by one", `${three.value} then ${two.value}`);
    s.check(paged.value > two.value + 1, "PageUp moves by more than one step", `${two.value} to ${paged.value}`);
    s.check(paged.valuetext === `probability ${paged.p}`, "the slider's spoken value follows the probability shown", `"${paged.valuetext}" / ${paged.p}`);
    await close();

    // a page that was opened and used with the keyboard only
    {
      const keyboard = await kit.open(servers.a, { tag: "s12-keys-only", reducedMotion: "reduce" });
      const p = keyboard.page;
      await p.evaluate(() => document.activeElement?.blur?.());
      // tab to the Policy tab and open it with Enter
      for (let i = 0; i < 6; i += 1) {
        await p.keyboard.press("Tab");
        const f = await p.evaluate(describeFocus);
        if (f.id === "tab-policy") break;
      }
      await p.keyboard.press("Enter");
      await p.waitForSelector("#view-policy:not([hidden])");
      const opened = await p.evaluate(() => ({ hidden: document.querySelector("#view-policy").hidden, selected: document.querySelector("#tab-policy").getAttribute("aria-selected") }));
      s.check(opened.hidden === false && opened.selected === "true", "the Policy page opens from the keyboard alone (Tab to the tab, Enter)", `hidden=${opened.hidden}, selected=${opened.selected}`);
      await keyboard.close();
    }
  },
};

export const s13 = {
  id: "S13",
  title: "Reduced motion: no animations or transitions, numbers jump to the final value",
  async run(s, ctx) {
    const { kit, servers } = ctx;
    const { page, close } = await kit.open(servers.a, { tag: "s13", reducedMotion: "reduce" });
    s.check(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches), "the page sees prefers-reduced-motion: reduce");

    await box(page).fill(SENTENCE);
    await waitForScoreResponses(page, 1);
    // read the numbers at once and again later: they must already be final
    await page.waitForFunction(() => document.querySelector("#vt-logreg_word")?.dataset.tile === "scored");
    const first = await page.evaluate(readScoreUi);
    await sleep(700);
    const later = await page.evaluate(readScoreUi);
    const response = page.scores.responses[0].body;
    const expected = {};
    for (const result of response.results) for (const [label, p] of Object.entries(result.probabilities)) expected[`${result.system}|${label}`] = p.toFixed(4);
    const atOnce = Object.entries(first.cells).filter(([key]) => key in expected);
    const wrongAtOnce = atOnce.filter(([key, cell]) => cell.val !== expected[key]);
    s.check(atOnce.length === 12 && wrongAtOnce.length === 0, "the numbers show their final values the moment the verdict appears (no count-up)", `${atOnce.length - wrongAtOnce.length} of ${atOnce.length} already final`);
    const changed = Object.keys(first.cells).filter((key) => first.cells[key].val !== later.cells[key].val);
    s.check(changed.length === 0, "and they do not change afterwards", `${changed.length} changed after 700 ms`);

    // every animated element
    await page.getByRole("tab", { name: "Policy" }).click();
    await page.waitForSelector("#fp-precision .fp");
    await page.getByRole("tab", { name: "Results" }).click();
    await page.waitForSelector("#lc circle");
    const report = await page.evaluate(() => {
      const names = new Set();
      const bad = [];
      let checked = 0;
      for (const el of document.querySelectorAll("*")) {
        for (const pseudo of [null, "::before", "::after"]) {
          const style = getComputedStyle(el, pseudo);
          checked += 1;
          const animates = style.animationName !== "none" && parseFloat(style.animationDuration) > 0;
          const transitions = style.transitionProperty !== "none" && style.transitionDuration.split(",").some((d) => parseFloat(d) > 0);
          if (animates || transitions) {
            bad.push(`${el.tagName.toLowerCase()}${el.className && typeof el.className === "string" ? "." + el.className.split(" ")[0] : ""}${pseudo ?? ""} ${animates ? `animation ${style.animationName}` : `transition ${style.transitionDuration}`}`);
          }
        }
      }
      return { checked, bad: [...new Set(bad)], running: document.getAnimations().filter((a) => a.playState === "running").map((a) => a.animationName ?? a.transitionProperty ?? "?") };
    });
    s.check(report.bad.length === 0, "no element or pseudo-element has an animation or a transition longer than 0", `${report.checked} computed styles checked; ${report.bad.length} animated${report.bad.length ? `: ${report.bad.slice(0, 4).join("; ")}` : ""}`);
    s.check(report.running.length === 0, "document.getAnimations() reports nothing running", `${report.running.length} running`);
    const chart = await page.evaluate(() => {
      const path = document.querySelector("#lc path");
      return { offset: path.style.strokeDashoffset, animation: path.style.animation };
    });
    s.check(chart.offset === "0" || chart.offset === "0px", "the chart lines are drawn in full (not waiting for an animation)", `dashoffset ${chart.offset}, animation "${chart.animation}"`);

    // for comparison, with motion allowed the same elements do animate (so the check above can fail)
    await close();
    const lively = await kit.open(servers.a, { tag: "s13-motion-on", reducedMotion: "no-preference" });
    const animated = await lively.page.evaluate(() => {
      let count = 0;
      for (const el of document.querySelectorAll("*")) {
        const style = getComputedStyle(el);
        if ((style.animationName !== "none" && parseFloat(style.animationDuration) > 0) || (style.transitionProperty !== "none" && style.transitionDuration.split(",").some((d) => parseFloat(d) > 0))) count += 1;
      }
      return count;
    });
    s.check(animated > 10, "control: with motion allowed, many elements do animate or transition", `${animated} elements`);
    await lively.close();
  },
};
