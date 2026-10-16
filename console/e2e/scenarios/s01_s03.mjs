// S01 status chips, S02 empty input and limits, S03 live typing and explicit scoring.
import { freePort, getJson, localReady, sleep, startApi, startConsoleServer, waitForHealth } from "../lib/harness.mjs";
import { LIMIT, LOCAL, SENTENCE, SYSTEMS, box, scoreButton, settleValues, textOfLength, waitForScoreRequests, waitForScoreResponses } from "../lib/kit.mjs";
import { readScoreUi } from "../lib/extract.mjs";

const chipState = (page, system) =>
  page.evaluate((id) => {
    const el = document.querySelector(`.sc[data-system="${id}"]`);
    return el ? { state: el.dataset.state, className: el.className, title: el.getAttribute("title"), text: el.textContent.replace(/\s+/g, " ").trim() } : null;
  }, system);

const apiChip = (page) =>
  page.evaluate(() => {
    const el = document.querySelector(".sc[data-api]");
    return el ? { api: el.dataset.api, className: el.className, text: el.textContent.replace(/\s+/g, " ").trim(), title: el.getAttribute("title") } : null;
  });

const systemChipCount = (page) => page.evaluate(() => document.querySelectorAll(".sc[data-system]").length);

export const s01 = {
  id: "S01",
  title: "Status chips: reachable, ready, needs key, unreachable and recovery",
  async run(s, ctx) {
    const { kit, servers, apis } = ctx;

    // JEV off: the real server with no key
    {
      const { page, close } = await kit.open(servers.a, { tag: "s01-jev-off" });
      const api = await apiChip(page);
      s.check(api.api === "online" && api.text.includes("API online"), "API chip says the API is reachable", `data-api=${api.api}, text "${api.text}"`);
      const lr = await chipState(page, "logreg_word");
      const db = await chipState(page, "distilbert");
      const jev = await chipState(page, "jev");
      s.check(lr.state === "ready" && !lr.className.includes("off"), "logistic regression chip is ready", `state=${lr.state}, text "${lr.text}"`);
      s.check(db.state === "ready" && !db.className.includes("off"), "DistilBERT chip is ready", `state=${db.state}, text "${db.text}"`);
      s.check(jev.state === "needs key" && jev.className.includes("off"), "JEV chip shows needs key when no key is configured", `state=${jev.state}, text "${jev.text}"`);
      s.check(/TypeSafe API key/.test(jev.title ?? ""), "JEV chip explains the key in its tooltip", jev.title ?? "");
      const health = (await getJson(`${apis.a.url}/api/health`)).body;
      s.check(health.systems.jev.reason_code === "no_api_key" && health.systems.jev.status === "unavailable", "the chip follows /api/health (reason_code no_api_key)", JSON.stringify(health.systems.jev));
      await close();
    }

    // JEV on: the stub scorer
    {
      const { page, close } = await kit.open(servers.b, { tag: "s01-jev-stub" });
      const jev = await chipState(page, "jev");
      s.check(jev.state === "ready" && !jev.className.includes("off"), "JEV chip is ready when a JEV scorer is available", `state=${jev.state}, text "${jev.text}"`);
      const health = (await getJson(`${apis.b.url}/api/health`)).body;
      s.check(health.systems.jev.status === "available" && health.systems.jev.reason_code === null, "/api/health agrees (available, no reason code)", JSON.stringify(health.systems.jev));
      await close();
    }

    // The request is aborted (as if the server were unreachable) and then allowed again
    {
      const { page, close } = await kit.open(servers.a, { tag: "s01-abort", expected: true, clock: true });
      s.check((await apiChip(page)).api === "online", "starts online (clock controlled)");
      await page.route("**/api/health", (route) => route.abort());
      await page.clock.fastForward(16_000);
      await page.waitForSelector('.sc[data-api="offline"]', { timeout: 10_000 });
      const down = await apiChip(page);
      s.check(down.api === "offline" && down.text.includes("API unreachable") && down.className.includes("down"), "API chip changes to the unreachable state when the health request is aborted", `text "${down.text}", class "${down.className}"`);
      s.check((await systemChipCount(page)) === 0, "system chips are not shown while the API is unreachable", `${await systemChipCount(page)} chips`);
      await page.unroute("**/api/health");
      await page.clock.fastForward(5_000);
      await page.waitForSelector('.sc[data-api="online"]', { timeout: 10_000 });
      await page.waitForSelector('.sc[data-system="jev"]', { timeout: 10_000 });
      const back = await apiChip(page);
      s.check(back.api === "online" && (await systemChipCount(page)) === 3, "the chip recovers and the three system chips return", `text "${back.text}", ${await systemChipCount(page)} chips`);
      await close();
    }

    // The server process is killed and started again (a light server, so it comes back in seconds)
    {
      const port = await freePort();
      let light = startApi({ name: "light", port, mode: "light", jev: "off" });
      const server = await startConsoleServer({ upstreamPort: port, label: "light" });
      ctx.allowedHosts.add(server.host);
      try {
        await waitForHealth(light, localReady, 60_000);
        const { page, close } = await kit.open(server, { tag: "s01-kill", expected: true, clock: true });
        s.check((await apiChip(page)).api === "online", "light server is online before it is killed");
        light.stop();
        await sleep(1500);
        await page.clock.fastForward(16_000);
        await page.waitForSelector('.sc[data-api="offline"]', { timeout: 15_000 });
        const down = await apiChip(page);
        s.check(down.api === "offline" && down.text.includes("API unreachable"), "API chip changes to unreachable when the server process is killed", `text "${down.text}"`);
        light = startApi({ name: "light-again", port, mode: "light", jev: "off" });
        await waitForHealth(light, localReady, 60_000);
        await page.clock.fastForward(5_000);
        await page.waitForSelector('.sc[data-api="online"]', { timeout: 15_000 });
        await page.waitForSelector('.sc[data-system="logreg_word"]', { timeout: 15_000 });
        const back = await apiChip(page);
        const lr = await chipState(page, "logreg_word");
        s.check(back.api === "online" && lr.state === "ready", "the chip recovers when the server comes back", `text "${back.text}", logistic regression ${lr.state}`);
        await close();
      } finally {
        light.stop();
        await server.close();
      }
    }
  },
};

export const s02 = {
  id: "S02",
  title: "Score: empty input, the 2000-character limit, Ctrl+Enter, whitespace-only",
  async run(s, ctx) {
    const { kit, servers } = ctx;

    // empty
    {
      const { page, close } = await kit.open(servers.a, { tag: "s02-empty" });
      s.check(await scoreButton(page).isDisabled(), "Score is disabled while the box is empty");
      await box(page).fill("hello");
      s.check(!(await scoreButton(page).isDisabled()), "Score is enabled once there is text");
      await box(page).fill("");
      s.check(await scoreButton(page).isDisabled(), "Score is disabled again after clearing the box");
      await close();
    }

    // exactly the limit is accepted
    {
      const { page, close } = await kit.open(servers.a, { tag: "s02-limit" });
      await box(page).fill(textOfLength(LIMIT));
      await waitForScoreResponses(page, 1);
      await settleValues(page);
      const ui = await page.evaluate(readScoreUi);
      const request = page.scores.requests[0];
      const response = page.scores.responses[0];
      s.check(request.body.text.length === LIMIT, "a text of exactly 2000 characters is sent", `${request.body.text.length} characters`);
      s.check(response.status === 200 && response.body.results.length === 2, "the API accepts it and scores the two local systems", `status ${response.status}, ${response.body.results.length} results`);
      s.check(ui.alert === null && ui.limit === null, "no error or limit message is shown at 2000", `alert=${ui.alert}, limit=${ui.limit}`);
      s.check(ui.count === `${LIMIT} / ${LIMIT}`, "the counter reads 2000 / 2000", ui.count ?? "");
      s.check(ui.tiles.logreg_word.kind === "scored" && !ui.scoreDisabled, "the verdicts show and Score stays enabled", `tile ${ui.tiles.logreg_word.kind}`);

      // one more character is refused
      const before = page.scores.requests.length;
      await box(page).fill(textOfLength(LIMIT + 1));
      await sleep(900);
      await box(page).press("Control+Enter");
      await sleep(400);
      const over = await page.evaluate(readScoreUi);
      s.check(page.scores.requests.length === before, "2001 characters: no request is sent, by typing or by Ctrl+Enter", `${page.scores.requests.length - before} requests`);
      s.check(/longer than 2000 characters/.test(over.limit ?? ""), "2001 characters: the limit message is visible", over.limit ?? "none");
      s.check(over.scoreDisabled === true, "2001 characters: Score is disabled");
      s.check(over.count === `${LIMIT + 1} / ${LIMIT}`, "2001 characters: the counter reads 2001 / 2000", over.count ?? "");

      // and back to the limit
      await box(page).fill(textOfLength(LIMIT));
      await waitForScoreRequests(page, before + 1);
      const ok = await page.evaluate(readScoreUi);
      s.check(ok.limit === null && ok.scoreDisabled === false, "removing the extra character clears the message and scoring resumes", `limit=${ok.limit}`);
      await close();
    }

    // Ctrl+Enter
    {
      const { page, close } = await kit.open(servers.a, { tag: "s02-ctrl" });
      await box(page).fill("hello there");
      await box(page).press("Control+Enter");
      await waitForScoreResponses(page, 1);
      await sleep(800);
      s.check(page.scores.requests.length === 1, "Ctrl+Enter sends one request straight away and the typing pause adds none", `${page.scores.requests.length} requests`);
      s.check(page.scores.requests[0].body.systems === undefined, "Ctrl+Enter asks for every system (no systems list)", JSON.stringify(Object.keys(page.scores.requests[0].body)));
      const ui = await page.evaluate(readScoreUi);
      s.check(ui.tiles.logreg_word.kind === "scored" && ui.tiles.distilbert.kind === "scored", "both local verdicts appear", `${ui.tiles.logreg_word.kind}, ${ui.tiles.distilbert.kind}`);
      s.check(ui.tiles.jev.kind === "unavailable", "JEV shows as unavailable (no key), not as an error", ui.tiles.jev.word);
      await close();
    }

    // whitespace-only behaves like empty
    {
      const { page, close } = await kit.open(servers.a, { tag: "s02-blank" });
      for (const blank of ["   ", "\n\n", " \t \n "]) {
        await box(page).fill(blank);
        s.check(await scoreButton(page).isDisabled(), `whitespace-only text (${JSON.stringify(blank)}) leaves Score disabled`);
        await box(page).press("Control+Enter");
      }
      await sleep(900);
      const ui = await page.evaluate(readScoreUi);
      s.check(page.scores.requests.length === 0, "whitespace-only text sends no request, by typing or by Ctrl+Enter", `${page.scores.requests.length} requests`);
      s.check(ui.tiles.logreg_word.kind === "idle" && ui.tiles.logreg_word.word === "Waiting for text", "the tiles still say Waiting for text", ui.tiles.logreg_word.word ?? "");
      await box(page).fill("real words");
      await waitForScoreRequests(page, 1);
      s.check(page.scores.requests.length === 1 && page.scores.requests[0].body.text === "real words", "real text typed after the whitespace is scored", `${page.scores.requests.length} request`);
      await close();
    }
  },
};

export const s03 = {
  id: "S03",
  title: "Live typing: 320 ms pause, local systems only, explicit Score for JEV",
  async run(s, ctx) {
    const { kit, servers, apis } = ctx;
    const { page, close } = await kit.open(servers.b, { tag: "s03-b" });
    const baseJev = apis.b.jevCalls();
    const text = "Hello there friend";

    await box(page).click();
    // the time of the last input event, taken in the page (the same clock as this process)
    await page.evaluate(() => {
      window.__lastInput = 0;
      document.addEventListener("input", () => { window.__lastInput = Date.now(); }, true);
    });
    await page.keyboard.type(text, { delay: 120 });
    const during = page.scores.requests.length;
    s.check(during === 0, "typing with 120 ms between keys sends no request", `${during} requests while typing ${text.length} characters`);
    await waitForScoreRequests(page, 1);
    const lastKey = await page.evaluate(() => window.__lastInput);
    const gap = page.scores.requests[0].at - lastKey;
    await sleep(900);
    s.check(page.scores.requests.length === 1, "after the pause exactly one request is sent", `${page.scores.requests.length} requests`);
    s.check(gap >= 300 && gap <= 1500, "the request follows the last keystroke by the 320 ms pause (not sooner)", `${gap} ms`);
    const body = page.scores.requests[0].body;
    const api = (await getJson(`${apis.b.url}/api/health`)).body;
    const localIds = Object.keys(api.systems).filter((id) => id !== "jev");
    s.check(JSON.stringify(body.systems) === JSON.stringify(localIds), "the body names the two local systems, using the ids the API reports", JSON.stringify(body.systems));
    s.check(!body.systems.includes("jev"), "JEV is never in a live request");
    s.check(body.text === text, "the text is the text typed");
    await waitForScoreResponses(page, 1);
    s.check(page.scores.responses[0].body.results.map((r) => r.system).join() === LOCAL.join(), "the API answers for the two local systems only", page.scores.responses[0].body.results.map((r) => r.system).join());
    s.check(apis.b.jevCalls() === baseJev, "the JEV scorer was not called while typing", `${apis.b.jevCalls() - baseJev} calls`);
    let ui = await page.evaluate(readScoreUi);
    s.check(ui.tiles.jev.kind === "pending" && ui.tiles.jev.word === "Press Score to run JEV", "the JEV tile says Press Score to run JEV", `${ui.tiles.jev.kind}: "${ui.tiles.jev.word}"`);

    // explicit Score click
    await scoreButton(page).click();
    await waitForScoreResponses(page, 2);
    s.check(page.scores.requests[1].body.systems === undefined, "a Score click asks for every system", JSON.stringify(Object.keys(page.scores.requests[1].body)));
    s.check(page.scores.responses[1].body.results.map((r) => r.system).join() === SYSTEMS.join(), "and the API answers for all three", page.scores.responses[1].body.results.map((r) => r.system).join());
    await settleValues(page);
    ui = await page.evaluate(readScoreUi);
    s.check(ui.tiles.jev.kind === "scored", "the JEV verdict appears", `${ui.tiles.jev.kind}: "${ui.tiles.jev.word}"`);
    s.check(apis.b.jevCalls() === baseJev + 1, "the JEV scorer was called once for the click", `${apis.b.jevCalls() - baseJev} calls`);

    // Ctrl+Enter
    await box(page).click();
    await box(page).press("Control+Enter");
    await waitForScoreResponses(page, 3);
    s.check(page.scores.requests[2].body.systems === undefined && page.scores.responses[2].body.results.length === 3, "Ctrl+Enter also asks for and gets all three", `${page.scores.responses[2].body.results.length} results`);
    s.check(apis.b.jevCalls() === baseJev + 2, "and calls JEV once more", `${apis.b.jevCalls() - baseJev} calls`);

    // after an edit
    await box(page).press("End");
    await page.keyboard.type("!");
    ui = await page.evaluate(readScoreUi);
    s.check(ui.tiles.jev.kind === "pending" && ui.tiles.jev.word === "Press Score to run JEV", "after an edit the JEV tile returns to Press Score to run JEV", `${ui.tiles.jev.kind}: "${ui.tiles.jev.word}"`);
    await sleep(900);
    s.check(page.scores.requests.length === 4 && JSON.stringify(page.scores.requests[3].body.systems) === JSON.stringify(localIds), "the edit triggers a local-only live request", `${page.scores.requests.length} requests, systems ${JSON.stringify(page.scores.requests[3]?.body.systems)}`);
    s.check(apis.b.jevCalls() === baseJev + 2, "live typing after the edit did not call JEV", `${apis.b.jevCalls() - baseJev} calls in total`);
    await close();

    // JEV unavailable: the tile says so instead of offering to run it
    {
      const unavailable = await kit.open(servers.a, { tag: "s03-a" });
      await box(unavailable.page).fill(SENTENCE);
      await waitForScoreResponses(unavailable.page, 1);
      await settleValues(unavailable.page);
      const view = await unavailable.page.evaluate(readScoreUi);
      s.check(view.tiles.jev.kind === "unavailable" && view.tiles.jev.word === "Unavailable", "with no key the JEV tile says Unavailable, not Press Score", `${view.tiles.jev.kind}: "${view.tiles.jev.word}"`);
      await scoreButton(unavailable.page).click();
      await waitForScoreResponses(unavailable.page, 2);
      const issue = unavailable.page.scores.responses[1].body.issues[0];
      s.check(issue.system === "jev" && issue.status === "unavailable" && issue.reason_code === "no_api_key", "a Score click returns JEV as an unavailable issue with its reason code", JSON.stringify(issue));
      await unavailable.close();
    }
  },
};
