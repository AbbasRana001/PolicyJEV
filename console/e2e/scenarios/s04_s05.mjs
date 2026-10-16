// S04 verdict consistency (the browser records what it shows; tests/e2e_check.py compares it with the
// policy engine), S05 privacy.
import { randomBytes } from "node:crypto";
import { changedFiles, getJson, needlesIn, sleep, snapshotRepo } from "../lib/harness.mjs";
import { SYSTEMS, box, scoreButton, settleValues, textOfLength, waitForScoreResponses } from "../lib/kit.mjs";
import { readScoreUi } from "../lib/extract.mjs";

export const VERDICT_CASES = [
  { id: "benign (Friendly chip)", chip: "Friendly" },
  { id: "blunt (Blunt chip)", chip: "Blunt" },
  { id: "hostile (Hostile chip)", chip: "Hostile" },
  { id: "empty-ish (one full stop)", text: "." },
  { id: "2000 characters", text: textOfLength(2000) },
  { id: "unicode and emoji", text: "Grüße 👍 ありがとう 😀 مرحبا שלום" },
  { id: "one character repeated 600 times", text: "a".repeat(600) },
  { id: "mixed case", text: "tHiS Is a MiXeD cAsE sEnTeNcE WiTh WoRdS" },
  { id: "digits and punctuation", text: "12345 !!! ??? ... ### $$$" },
  { id: "several lines", text: "First line.\nSecond line.\n\nThird line after a gap." },
];

export const s04 = {
  id: "S04",
  title: "Verdict consistency with the policy engine, probabilities in range, numbers match the response",
  async run(s, ctx) {
    const { kit, servers, obs } = ctx;
    const { page, close } = await kit.open(servers.b, { tag: "s04" });
    // explicit requests only, so each case produces exactly one response
    await page.getByRole("switch", { name: "Score as I type" }).click();
    obs.verdictCases = [];
    for (const item of VERDICT_CASES) {
      if (item.chip) await page.getByRole("button", { name: item.chip, exact: true }).click();
      else await box(page).fill(item.text);
      const seen = page.scores.responses.length;
      await scoreButton(page).click();
      await waitForScoreResponses(page, seen + 1);
      await settleValues(page, 300);
      const ui = await page.evaluate(readScoreUi);
      const response = page.scores.responses[seen];
      const request = page.scores.requests[seen];
      obs.verdictCases.push({
        id: item.id,
        textLength: request.body.text.length,
        status: response.status,
        api: response.body,
        ui,
      });
      const scored = response.body.results.map((r) => r.system).join();
      s.check(
        response.status === 200 && scored === SYSTEMS.join() && SYSTEMS.every((id) => ui.tiles[id].kind === "scored"),
        `${item.id}: three systems scored and shown`,
        `${request.body.text.length} characters; decisions ${response.body.results.map((r) => `${r.system}=${r.decision}`).join(", ")}`,
      );
    }
    s.note(`${obs.verdictCases.length} inputs sent through the page; each response and the screen were recorded for tests/e2e_check.py`);
    await close();
  },
};

const hex = () => randomBytes(10).toString("hex");

/** Every cookie, storage and cache the page could be keeping, as text. */
async function browserState(context, page) {
  const cookies = await context.cookies();
  const state = await page.evaluate(async () => {
    const dump = (storage) => Object.fromEntries(Object.keys(storage).map((k) => [k, storage.getItem(k)]));
    const databases = [];
    if (indexedDB.databases) {
      for (const info of await indexedDB.databases()) {
        await new Promise((done) => {
          const open = indexedDB.open(info.name);
          open.onerror = () => done();
          open.onsuccess = () => {
            const db = open.result;
            const names = [...db.objectStoreNames];
            if (names.length === 0) {
              db.close();
              done();
              return;
            }
            const tx = db.transaction(names, "readonly");
            let pending = names.length;
            const finish = () => {
              pending -= 1;
              if (pending === 0) {
                db.close();
                done();
              }
            };
            for (const name of names) {
              const all = tx.objectStore(name).getAll();
              all.onsuccess = () => {
                databases.push({ db: info.name, store: name, data: JSON.stringify(all.result) });
                finish();
              };
              all.onerror = finish;
            }
          };
        });
      }
    }
    const cacheNames = "caches" in window ? await caches.keys() : [];
    const workers = navigator.serviceWorker ? (await navigator.serviceWorker.getRegistrations()).length : 0;
    return {
      cookie: document.cookie,
      local: dump(localStorage),
      session: dump(sessionStorage),
      databases,
      cacheNames,
      workers,
      windowName: window.name,
      title: document.title,
      url: location.href,
      historyState: JSON.stringify(history.state),
      resources: performance.getEntriesByType("resource").map((entry) => entry.name),
    };
  });
  return { cookies, ...state };
}

export const s05 = {
  id: "S05",
  title: "Privacy: a unique sentence appears nowhere it should not",
  async run(s, ctx) {
    const { kit, servers, apis } = ctx;
    const tag = `sn${hex()}`;
    const sentence = `Please keep ${tag} private, thanks.`;
    const needles = [tag, sentence, encodeURIComponent(sentence), Buffer.from(sentence).toString("base64"), Buffer.from(tag).toString("base64")];
    ctx.sentinelNeedles = needles;
    const has = (haystack) => needles.some((needle) => String(haystack).includes(needle));
    // anything printed in the report is cleaned of the sentence first, so a leak cannot spread into the report
    const show = (value) => needles.reduce((text, needle) => text.split(needle).join("[sentence]"), String(value));

    const before = snapshotRepo();
    const apiLogLength = { a: apis.a.text().length, b: apis.b.text().length };
    const { page, context, close } = await kit.open(servers.b, { tag: "s05" });
    const requestsBefore = kit.requests.length;

    await box(page).fill(sentence);
    await waitForScoreResponses(page, 1);
    await scoreButton(page).click();
    await waitForScoreResponses(page, 2);
    await settleValues(page);
    const ui = await page.evaluate(readScoreUi);
    s.check(ui.tiles.logreg_word.kind === "scored" && ui.tiles.jev.kind === "scored", "the sentence was scored by the local systems and by the JEV stub", `${ui.tiles.logreg_word.kind}, ${ui.tiles.distilbert.kind}, ${ui.tiles.jev.kind}`);
    await sleep(500);

    // what the page sent
    const mine = kit.requests.slice(requestsBefore).filter((r) => r.tag === "s05");
    const hosts = [...new Set(mine.map((r) => r.host))];
    s.check(hosts.length === 1 && hosts[0] === servers.b.host, "every request went to the test server and to no other host", `${mine.length} requests; hosts ${hosts.join(", ")}`);
    const withText = mine.filter((r) => has(r.body ?? ""));
    s.check(withText.length >= 2 && withText.every((r) => r.method === "POST" && new URL(r.url).pathname === "/api/score"), "the sentence is in the body of the score requests and in no other request", `${withText.length} of ${mine.length} requests carry it, all POST /api/score`);
    s.check(mine.every((r) => !has(r.url)), "it is in no request URL", `${mine.length} URLs checked`);

    // what the page kept
    const state = await browserState(context, page);
    const kept = { cookies: JSON.stringify(state.cookies), cookie: state.cookie, local: JSON.stringify(state.local), session: JSON.stringify(state.session), idb: JSON.stringify(state.databases), name: state.windowName, title: state.title, url: state.url, history: state.historyState, resources: state.resources.join("\n"), caches: JSON.stringify(state.cacheNames) };
    for (const [where, value] of Object.entries(kept)) {
      const shown = show(value);
      s.check(!has(value), `it is not in the page's ${where}`, shown.length > 80 ? `${shown.slice(0, 80)}...` : shown || "(empty)");
    }
    s.check(state.workers === 0 && state.cacheNames.length === 0, "no service worker and no cache storage", `${state.workers} workers, ${state.cacheNames.length} caches`);
    s.note(`localStorage keys: ${JSON.stringify(Object.keys(state.local))}; sessionStorage keys: ${JSON.stringify(Object.keys(state.session))}; IndexedDB stores: ${state.databases.length}`);

    // what the servers logged
    const logsA = apis.a.text();
    const logsB = apis.b.text();
    s.check(!has(logsA) && !has(logsB), "it is in no output of either API process (stdout and stderr)", `${logsA.length + logsB.length} characters of output, ${logsB.length - apiLogLength.b} written during this scenario`);
    const accessLines = logsB.split("\n").filter((line) => line.includes("/api/score"));
    s.check(accessLines.length >= 2 && accessLines.every((line) => !has(line)), "the access log lists the score requests without their content", `${accessLines.length} lines, e.g. ${show(accessLines[0]?.trim().slice(0, 80) ?? "")}`);

    // what reached the disk during this scenario
    const changed = changedFiles(before, snapshotRepo());
    const leaking = changed.filter((path) => needlesIn(path, needles).length > 0);
    s.check(leaking.length === 0, "it is in no file created or changed under the repository during this scenario", `${changed.length} files changed, ${leaking.length} contain it`);
    await close();

    // invalid requests never echo the input
    const probes = [
      { name: "text over the limit", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: sentence + "x".repeat(2100) }) } },
      { name: "empty text, unknown field", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: "", extra: sentence }) } },
      { name: "unknown system", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: sentence, systems: ["nope"] }) } },
      { name: "empty systems list", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: sentence, systems: [] }) } },
      { name: "text of the wrong type", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: [sentence] }) } },
      { name: "broken JSON", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: `{"text": "${sentence}` } },
      { name: "plain text body", init: { method: "POST", headers: { "Content-Type": "text/plain" }, body: sentence } },
    ];
    for (const probe of probes) {
      const { status, text } = await getJson(`${apis.b.url}/api/score`, probe.init);
      s.check(status === 422 && !has(text), `422 for ${probe.name} and the body does not echo the input`, `status ${status}, ${text.length} characters`);
    }
    s.check(!has(apis.b.text()), "and the invalid requests did not put it in the process output either");

    // for the end-of-run sweep (whole run, including the files written by the harness)
    ctx.privacy = { needles };
  },
};
