// Serves the built app and proxies /api to a running local API, then saves screenshots to
// console/.shots/ at four widths. Also checks, at each width and at 360 px, that the page does
// not scroll sideways, and records console errors and any request that leaves this machine.
//
// Run `npm run build` first, start the API (python -m policyjev.api), then:
//   node scripts/shots.mjs            (API on http://127.0.0.1:8000)
//   API_URL=http://127.0.0.1:8010 node scripts/shots.mjs
//
// Only benign text is typed.
import { createReadStream, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const DIST = join(ROOT, "dist");
const OUT = join(ROOT, ".shots");
const API = process.env.API_URL ?? "http://127.0.0.1:8000";
const PORT = 4173;
const WIDTHS = [1440, 1024, 768, 375];
const SENTENCE = "Thanks for the update, see you at the meeting tomorrow.";
const HIGH_PROBABILITY_POSITION = 950;

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".svg": "image/svg+xml",
  ".json": "application/json",
};

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", "http://localhost");
  if (url.pathname.startsWith("/api/")) {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    try {
      const upstream = await fetch(API + url.pathname + url.search, {
        method: request.method,
        headers: { "content-type": request.headers["content-type"] ?? "application/json" },
        body: hasBody ? Buffer.concat(chunks) : undefined,
      });
      response.writeHead(upstream.status, { "content-type": upstream.headers.get("content-type") ?? "application/json" });
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch {
      response.writeHead(502);
      response.end();
    }
    return;
  }
  let file = normalize(join(DIST, url.pathname === "/" ? "index.html" : url.pathname));
  if (!file.startsWith(DIST) || !existsSync(file) || statSync(file).isDirectory()) file = join(DIST, "index.html");
  response.setHeader("Content-Type", TYPES[extname(file)] ?? "application/octet-stream");
  createReadStream(file).pipe(response);
});
await new Promise((resolve) => server.listen(PORT, "127.0.0.1", resolve));

mkdirSync(OUT, { recursive: true });
for (const name of readdirSync(OUT)) if (name.endsWith(".png")) rmSync(join(OUT, name));

const browser = await chromium.launch();
const problems = [];
const external = new Set();
const consoleErrors = [];

async function open(width) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  const page = await context.newPage();
  page.on("request", (request) => {
    const host = new URL(request.url()).host;
    if (host !== `127.0.0.1:${PORT}` && !request.url().startsWith("data:")) external.add(request.url());
  });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(`${width}px: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleErrors.push(`${width}px: ${error.message}`));
  await page.goto(`http://127.0.0.1:${PORT}/`);
  await page.waitForSelector("h1");
  await page.evaluate(() => document.fonts.ready);
  // the chips and the meters need the health and threshold answers
  await page.waitForSelector(".sc[data-system]");
  await page.waitForSelector(".meter");
  return { context, page };
}

async function check(page, name, width) {
  const report = await page.evaluate(() => {
    const doc = document.documentElement;
    const overflowX = doc.scrollWidth - doc.clientWidth;
    const wide = [];
    for (const element of document.querySelectorAll("body *")) {
      if (element.closest("[hidden]")) continue;
      const box = element.getBoundingClientRect();
      if (box.width > 0 && box.right > doc.clientWidth + 1) wide.push(`${element.tagName}.${String(element.className).slice(0, 40)}`);
    }
    return { overflowX, wide: wide.slice(0, 6) };
  });
  if (report.overflowX > 0 || report.wide.length) problems.push({ name, width, ...report });
}

async function settle(page, ms = 1800) {
  await page.waitForTimeout(ms);
}

async function shot(page, name, width) {
  await check(page, name, width);
  await page.screenshot({ path: join(OUT, `${name}-${width}.png`), fullPage: true });
}

for (const width of WIDTHS) {
  const { context, page } = await open(width);

  await settle(page, 600);
  await shot(page, "score-empty", width);

  await page.fill("#score-text", SENTENCE);
  await page.click("button.primary");
  await page.waitForSelector("#vt-logreg_word[data-decision]");
  await page.waitForSelector("#vt-distilbert[data-decision]");
  await settle(page, 900);
  await shot(page, "score-scored", width);

  await page.click("#tab-policy");
  await page.waitForSelector("#metrics .mr");
  await settle(page);
  await shot(page, "policy", width);

  await page.locator("#probe").fill(String(HIGH_PROBABILITY_POSITION));
  await settle(page, 600);
  await shot(page, "policy-explorer-high", width);

  await page.click("#tab-results");
  await page.waitForSelector("#apc .apb");
  await page.waitForSelector("#lc path");
  await settle(page, 2000);
  await shot(page, "results", width);

  await context.close();
}

// No sideways scrolling from 360 px up, in every section.
{
  const { context, page } = await open(360);
  await page.fill("#score-text", SENTENCE);
  await page.click("button.primary");
  await page.waitForSelector("#vt-logreg_word[data-decision]");
  await settle(page, 600);
  await check(page, "score-scored", 360);
  await page.click("#tab-policy");
  await page.waitForSelector("#metrics .mr");
  await settle(page, 1200);
  await check(page, "policy", 360);
  await page.click("#tab-results");
  await page.waitForSelector("#apc .apb");
  await settle(page, 1500);
  await check(page, "results", 360);
  await context.close();
}

await browser.close();
server.close();

console.log(JSON.stringify({ problems, consoleErrors, requestsLeavingThisMachine: [...external] }, null, 1));
console.log(`saved screenshots to ${OUT}`);
process.exit(problems.length || consoleErrors.length || external.size ? 1 : 0);
