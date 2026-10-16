// S14 console cleanliness, S15 model warm-up, S16 contrast on the rendered page.
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { CONSOLE, freePort, localReady, sleep, startApi, startConsoleServer, waitForHealth } from "../lib/harness.mjs";
import { SENTENCE, box, scoreButton, tab, waitForScoreResponses } from "../lib/kit.mjs";
import { readScoreUi } from "../lib/extract.mjs";

export const s14 = {
  id: "S14",
  title: "Console cleanliness: no errors or warnings, no failed requests, no outside host",
  async run(s, ctx) {
    const { kit } = ctx;
    const tags = (list) => [...new Set(list.map((entry) => entry.tag))];
    const unexpectedMessages = kit.messages.filter((m) => !m.expected);
    const unexpectedFailures = kit.failures.filter((f) => !f.expected);
    const expectedMessages = kit.messages.length - unexpectedMessages.length;
    const expectedFailures = kit.failures.length - unexpectedFailures.length;
    s.check(unexpectedMessages.length === 0, "zero console errors or warnings in every page that was not made to fail", `${kit.pages} pages opened; ${unexpectedMessages.length} unexpected${unexpectedMessages.length ? `: ${unexpectedMessages.slice(0, 3).map((m) => `[${m.tag}] ${m.type} ${m.text}`).join(" | ")}` : ""}`);
    s.check(unexpectedFailures.length === 0, "zero failed or rejected requests and zero page errors in those pages", `${unexpectedFailures.length} unexpected${unexpectedFailures.length ? `: ${unexpectedFailures.slice(0, 3).map((f) => `[${f.tag}] ${f.kind} ${f.detail}`).join(" | ")}` : ""}`);
    const hosts = [...new Set(kit.requests.map((r) => r.host))];
    const outside = hosts.filter((host) => !ctx.allowedHosts.has(host));
    s.check(outside.length === 0, "every request of the whole run went to a test server on this machine", `${kit.requests.length} requests, ${hosts.length} hosts, ${outside.length} outside${outside.length ? `: ${outside.join(", ")}` : ""}`);
    const external = kit.requests.filter((r) => !/^http:\/\/127\.0\.0\.1:/.test(r.url) && !r.url.startsWith("data:") && !r.url.startsWith("blob:"));
    s.check(external.length === 0, "no request used an address other than 127.0.0.1", `${external.length} requests`);
    s.note(`expected failures, caused on purpose by scenarios ${tags([...kit.messages, ...kit.failures].filter((e) => e.expected)).join(", ")}: ${expectedMessages} console messages and ${expectedFailures} failed requests, left out of the counts above`);
  },
};

export const s15 = {
  id: "S15",
  title: "Model warm-up: the page shows a loading state, not a blank; first-request latency",
  async run(s, ctx) {
    const { kit } = ctx;
    const port = await freePort();
    const api = startApi({ name: "warm-up", port, mode: "real", jev: "off", warmDelay: 4 });
    const server = await startConsoleServer({ upstreamPort: port, label: "warm-up" });
    ctx.allowedHosts.add(server.host);
    try {
      await waitForHealth(api, () => true, 120_000);
      const started = Date.now();
      const { page, close } = await kit.open(server, { tag: "s15" });
      const chips = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll(".sc[data-system]")].map((c) => [c.dataset.system, c.dataset.state])));
      s.check(chips.logreg_word === "loading" && chips.distilbert === "loading", "while the models load, both local chips say loading", JSON.stringify(chips));

      // score while loading
      await box(page).fill(SENTENCE);
      await waitForScoreResponses(page, 1);
      await page.waitForFunction(() => document.querySelector("#vt-distilbert")?.dataset.tile === "loading");
      const ui = await page.evaluate(readScoreUi);
      const response = page.scores.responses[0].body;
      s.check(response.issues.filter((i) => i.status === "loading").length >= 1, "the API answers a request made during warm-up with loading issues, not an error", JSON.stringify(response.issues.map((i) => `${i.system}:${i.status}`)));
      s.check(ui.tiles.distilbert.kind === "loading" && ui.tiles.distilbert.word === "Loading model" && /still loading/.test(ui.tiles.distilbert.why ?? ""), "the DistilBERT tile shows Loading model and says why, not a blank", `"${ui.tiles.distilbert.word}": ${ui.tiles.distilbert.why}`);
      s.check(/Loading models/.test(ui.loadingNote ?? ""), "a note says the models are loading", ui.loadingNote ?? "none");
      s.check(ui.alert === null, "loading is not shown as an error", `alert=${ui.alert}`);
      const blankTiles = Object.entries(ui.tiles).filter(([, tile]) => !tile.word);
      s.check(blankTiles.length === 0, "no verdict tile is blank", `${Object.keys(ui.tiles).length} tiles, ${blankTiles.length} blank`);

      // the chips follow the server as the models finish loading
      await page.waitForFunction(() => document.querySelector('.sc[data-system="distilbert"]')?.dataset.state === "ready", null, { timeout: 90_000 });
      const readyAfter = (Date.now() - started) / 1000;
      s.note(`the chips went from loading to ready by themselves, ${readyAfter.toFixed(1)} s after the page opened (the server waits ${4} s before loading each model)`);

      // first requests after warm-up
      const latency = [];
      for (let n = 1; n <= 2; n += 1) {
        const seen = page.scores.responses.length;
        const t0 = Date.now();
        await scoreButton(page).click();
        await waitForScoreResponses(page, seen + 1);
        const wall = Date.now() - t0;
        const body = page.scores.responses[seen].body;
        latency.push({ n, wall, systems: Object.fromEntries(body.results.map((r) => [r.system, r.time_ms])) });
      }
      for (const { n, wall, systems } of latency) {
        s.note(`request ${n} after warm-up: whole request ${wall} ms; logistic regression ${systems.logreg_word?.toFixed(1)} ms, DistilBERT ${systems.distilbert?.toFixed(1)} ms (time_ms from the API)`);
      }
      const after = await page.evaluate(readScoreUi);
      s.check(after.tiles.logreg_word.kind === "scored" && after.tiles.distilbert.kind === "scored" && after.loadingNote === null, "once ready, Score gives both verdicts and the loading note is gone", `${after.tiles.logreg_word.kind}, ${after.tiles.distilbert.kind}, note=${after.loadingNote}`);
      ctx.obs.firstRequestLatency = latency;
      await close();
    } finally {
      api.stop();
      await server.close();
    }
  },
};

const OUT = join(CONSOLE, "e2e", "out");

export const s16 = {
  id: "S16",
  title: "Contrast: small text is at least 4.5:1 on the background it is actually drawn on",
  async run(s, ctx) {
    const { kit, servers, obs } = ctx;
    mkdirSync(OUT, { recursive: true });
    obs.contrast = [];
    const scenes = [
      { width: 1440, name: "score-empty" },
      { width: 1440, name: "score-scored" },
      { width: 1440, name: "policy" },
      { width: 1440, name: "results" },
      { width: 375, name: "score-scored" },
      { width: 375, name: "policy" },
      { width: 375, name: "results" },
    ];
    for (const scene of scenes) {
      const { page, close } = await kit.open(servers.a, { width: scene.width, reducedMotion: "reduce", tag: `s16-${scene.name}-${scene.width}` });
      if (scene.name === "score-scored") {
        await box(page).fill(SENTENCE);
        await scoreButton(page).click();
        await page.waitForFunction(() => document.querySelector("#vt-distilbert")?.dataset.tile === "scored");
      }
      if (scene.name === "policy") {
        await tab(page, "policy");
        await page.waitForSelector("#fp-precision .fp");
      }
      if (scene.name === "results") {
        await tab(page, "results");
        await page.waitForSelector("#lc circle");
      }
      await sleep(400);
      const items = await page.evaluate(() => {
        const out = [];
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const parse = (value) => {
          const m = value.match(/rgba?\(([^)]+)\)/);
          if (!m) return null;
          const [r, g, b, a = "1"] = m[1].split(/[ ,/]+/).filter(Boolean);
          return [Number(r), Number(g), Number(b), Number(a)];
        };
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          if (!node.nodeValue || node.nodeValue.trim() === "") continue;
          const el = node.parentElement;
          if (!el || ["SCRIPT", "STYLE", "TITLE"].includes(el.tagName)) continue;
          let hidden = false;
          let opacity = 1;
          for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
            const st = getComputedStyle(n);
            if (st.display === "none" || st.visibility === "hidden" || n.classList?.contains("sr-only") || n.hasAttribute?.("hidden")) hidden = true;
            opacity *= Number(st.opacity);
          }
          if (hidden) continue;
          const style = getComputedStyle(el);
          const svg = el.namespaceURI === "http://www.w3.org/2000/svg";
          const color = parse(svg ? style.fill : style.color);
          if (!color) continue;
          const range = document.createRange();
          range.selectNodeContents(node);
          const box = svg ? el.getBoundingClientRect() : range.getBoundingClientRect();
          if (box.width < 2 || box.height < 2) continue;
          out.push({
            text: node.nodeValue.replace(/\s+/g, " ").trim().slice(0, 30),
            color,
            opacity,
            size: parseFloat(style.fontSize),
            weight: Number(style.fontWeight),
            rect: [box.left + scrollX, box.top + scrollY, box.width, box.height],
            where: `${el.tagName.toLowerCase()}.${String(el.className?.baseVal ?? el.className).split(" ")[0]}`,
          });
        }
        for (const area of document.querySelectorAll("textarea")) {
          if (area.value === "" && area.placeholder && !area.closest("[hidden]")) {
            const style = getComputedStyle(area, "::placeholder");
            const box = area.getBoundingClientRect();
            out.push({ text: area.placeholder, color: parse(style.color), opacity: 1, size: 16, weight: 400, rect: [box.left + scrollX + 16, box.top + scrollY + 14, 160, 22], where: "textarea::placeholder" });
          }
        }
        return out;
      });
      await page.addStyleTag({
        content: "* { color: transparent !important; -webkit-text-fill-color: transparent !important; text-shadow: none !important; } svg text { fill: transparent !important; stroke: none !important; } textarea::placeholder { color: transparent !important; }",
      });
      await sleep(100);
      const file = join(OUT, `contrast-${scene.name}-${scene.width}.png`);
      await page.screenshot({ path: file, fullPage: true });
      obs.contrast.push({ scene: scene.name, width: scene.width, png: file, items });
      await close();
    }
    const total = obs.contrast.reduce((sum, scene) => sum + scene.items.length, 0);
    s.check(total > 400, "text was collected from all three pages at two widths", `${total} pieces of text on ${obs.contrast.length} screens`);
    s.note("each piece of text is checked against the pixels behind it, with the text hidden, by tests/e2e_check.py (S16 checks)");
  },
};
