// Browser helpers: opening the console with collectors for console output, failed requests and
// every score request, and a few waits that the scenarios share.
import { sleep } from "./harness.mjs";

export const SENTENCE = "Thanks for the update, see you at the meeting tomorrow.";
export const SYSTEMS = ["logreg_word", "distilbert", "jev"];
export const LOCAL = ["logreg_word", "distilbert"];
export const LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];
export const LIMIT = 2000;

export class Kit {
  constructor(browser) {
    this.browser = browser;
    /** console errors and warnings, page errors, failed and rejected requests, across the run */
    this.messages = [];
    this.failures = [];
    this.requests = [];
    this.pages = 0;
  }

  /**
   * Open the console served by `server`. Everything the page logs or fails to load is recorded under
   * `tag`; set `expected` for pages where the scenario causes failures on purpose.
   */
  async open(server, { width = 1440, height = 900, reducedMotion = "no-preference", tag = "page", expected = false, touch = false, clock = false, ready = true, before } = {}) {
    const context = await this.browser.newContext({
      viewport: { width, height },
      reducedMotion,
      hasTouch: touch,
      isMobile: touch,
      deviceScaleFactor: touch ? 2 : 1,
    });
    const page = await context.newPage();
    this.pages += 1;
    const scores = { requests: [], responses: [] };
    page.scores = scores;
    page.tag = tag;
    page.on("console", (message) => {
      if (message.type() === "error" || message.type() === "warning") {
        this.messages.push({ tag, expected, type: message.type(), text: message.text().slice(0, 300) });
      }
    });
    page.on("pageerror", (error) => this.failures.push({ tag, expected, kind: "pageerror", detail: String(error.message).slice(0, 300) }));
    page.on("requestfailed", (request) => this.failures.push({ tag, expected, kind: "requestfailed", detail: `${request.method()} ${new URL(request.url()).pathname} ${request.failure()?.errorText ?? ""}` }));
    page.on("response", (response) => {
      if (response.status() >= 400) this.failures.push({ tag, expected, kind: "http", detail: `${response.request().method()} ${new URL(response.url()).pathname} ${response.status()}` });
    });
    page.on("request", (request) => {
      const url = new URL(request.url());
      this.requests.push({ tag, method: request.method(), url: request.url(), host: url.host, body: request.postData() ?? null });
      if (url.pathname === "/api/score" && request.method() === "POST") {
        let body = null;
        try {
          body = JSON.parse(request.postData() ?? "null");
        } catch {
          body = null;
        }
        scores.requests.push({ at: Date.now(), body });
      }
    });
    page.on("response", async (response) => {
      const url = new URL(response.url());
      if (url.pathname === "/api/score" && response.request().method() === "POST") {
        try {
          scores.responses.push({ at: Date.now(), status: response.status(), body: await response.json() });
        } catch {
          scores.responses.push({ at: Date.now(), status: response.status(), body: null });
        }
      }
    });
    if (clock) await page.clock.install();
    if (before) await before(page, context);
    await page.goto(server.origin);
    await page.waitForSelector("h1");
    await page.evaluate(() => document.fonts.ready);
    if (ready) await waitReady(page);
    return { page, context, close: () => context.close() };
  }
}

/** The chips and meters appear once the API has answered health and thresholds. */
export async function waitReady(page) {
  await page.waitForSelector(".sc[data-system]", { timeout: 30_000 });
  await page.waitForSelector("#view-score .meter", { timeout: 30_000 });
}

export function box(page) {
  return page.locator("#score-text");
}

export function scoreButton(page) {
  return page.locator("#view-score button.primary");
}

export async function tab(page, name) {
  await page.click(`#tab-${name}`);
  await page.waitForSelector(`#view-${name}:not([hidden])`);
}

/** Wait until at least n score requests have been seen on the page. */
export async function waitForScoreRequests(page, n, timeout = 10_000) {
  const started = Date.now();
  while (page.scores.requests.length < n) {
    if (Date.now() - started > timeout) throw new Error(`only ${page.scores.requests.length} of ${n} score requests after ${timeout} ms`);
    await sleep(20);
  }
}

/** Wait until at least n score responses have been seen (and parsed) on the page. */
export async function waitForScoreResponses(page, n, timeout = 20_000) {
  const started = Date.now();
  while (page.scores.responses.length < n) {
    if (Date.now() - started > timeout) throw new Error(`only ${page.scores.responses.length} of ${n} score responses after ${timeout} ms`);
    await sleep(20);
  }
}

/** Wait for the value readouts to stop changing (the number count-up is 520 ms). */
export async function settleValues(page, quietMs = 250, timeout = 5_000) {
  const started = Date.now();
  let last = "";
  let since = Date.now();
  while (Date.now() - started < timeout) {
    const now = await page.evaluate(() => [...document.querySelectorAll("#view-score .cell .val")].map((v) => v.textContent).join("|"));
    if (now !== last) {
      last = now;
      since = Date.now();
    } else if (Date.now() - since >= quietMs) return;
    await sleep(40);
  }
}

/** Text of exactly n characters from a benign sentence repeated. */
export function textOfLength(n) {
  const unit = "This is a calm note about the schedule for next week. ";
  let text = "";
  while (text.length < n) text += unit;
  return text.slice(0, n);
}
