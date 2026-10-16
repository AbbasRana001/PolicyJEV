// `npm run e2e`: starts its own API servers on free ports (never port 8000) and the built console,
// runs every scenario in a real browser, has tests/e2e_check.py check what the browser saw against the
// CSVs and the policy engine, and writes console/e2e/report.txt.
//
// E2E_ONLY=S03,S06 runs only those scenarios (the report marks the others as skipped).
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";
import { CONSOLE, REPO, changedFiles, freePort, localReady, needlesIn, pythonPath, snapshotRepo, startApi, startConsoleServer, waitForHealth } from "./lib/harness.mjs";
import { Kit } from "./lib/kit.mjs";
import { Scenario, formatReport, runScenarios } from "./lib/runner.mjs";
import { s01, s02, s03 } from "./scenarios/s01_s03.mjs";
import { s04, s05 } from "./scenarios/s04_s05.mjs";
import { s06, s07, s08, s09 } from "./scenarios/s06_s09.mjs";
import { s10, s11, s11tiles } from "./scenarios/s10_s11.mjs";
import { s12, s13 } from "./scenarios/s12_s13.mjs";
import { s14, s15, s16 } from "./scenarios/s14_s16.mjs";

const OUT = join(CONSOLE, "e2e", "out");
const REPORT = join(CONSOLE, "e2e", "report.txt");
const ORDER = [s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s15, s16, s14];
// npm run e2e -- --only S07,S10   (or E2E_ONLY=S07,S10)
const flag = process.argv.indexOf("--only");
const selection = flag >= 0 ? process.argv[flag + 1] : process.env.E2E_ONLY;
const only = selection ? selection.split(",").map((id) => id.trim().toUpperCase()) : null;

if (!existsSync(join(CONSOLE, "dist", "index.html"))) {
  console.error("console/dist is missing: run `npm run build` first (npm run e2e does this).");
  process.exit(2);
}
mkdirSync(OUT, { recursive: true });

const started = new Date();
const repoStart = snapshotRepo();
const apis = {};
const servers = {};
let browser = null;
let scenarios = [];

try {
  const [portA, portB] = [await freePort(), await freePort()];
  console.log(`starting two API servers (ports ${portA} and ${portB}; the real server's port 8000 is not used)`);
  apis.a = startApi({ name: "A: real models, JEV unavailable (no key)", port: portA, jev: "off" });
  apis.b = startApi({ name: "B: real models, JEV stub", port: portB, jev: "stub", jevDelay: 0.4 });
  await Promise.all([waitForHealth(apis.a, localReady), waitForHealth(apis.b, localReady)]);
  servers.a = await startConsoleServer({ upstreamPort: portA, label: "A" });
  servers.b = await startConsoleServer({ upstreamPort: portB, label: "B" });
  browser = await chromium.launch();
  const kit = new Kit(browser);
  const ctx = {
    kit,
    apis,
    servers,
    obs: {},
    repoStart,
    allowedHosts: new Set([servers.a.host, servers.b.host]),
  };
  console.log("running scenarios");
  scenarios = await runScenarios(ORDER, ctx, only);

  // write what the browser saw, and the server output, for the Python checks and the final sweep
  writeFileSync(join(OUT, "observations.json"), JSON.stringify(ctx.obs));
  for (const api of Object.values(apis)) writeFileSync(join(OUT, `api-${api.port}.log`), api.text());

  // sweep the whole run for the privacy sentence, now that every file the harness writes exists
  const s05Run = scenarios.find((scenario) => scenario.id === "S05" && !scenario.skipped);
  if (s05Run && ctx.privacy) {
    const changed = changedFiles(repoStart, snapshotRepo());
    const leaking = changed.filter((path) => needlesIn(path, ctx.privacy.needles).length > 0);
    s05Run.check(leaking.length === 0, "end of run: it is in no file created or changed anywhere under the repository since the run began (logs and observations included)", `${changed.length} files changed, ${leaking.length} contain it`);
  }

  // Python checks of the recorded observations
  const python = pythonPath();
  const junit = join(OUT, "junit.xml");
  let pytestText = "";
  const result = spawnSync(python, ["-m", "pytest", "tests/e2e_check.py", "-q", "-p", "no:cacheprovider", `--junitxml=${junit}`], { cwd: REPO, encoding: "utf-8", env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
  pytestText = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  const xml = existsSync(junit) ? readFileSync(junit, "utf8") : "";
  const cases = [...xml.matchAll(/<testcase\b([^>]*?)(\/>|>([\s\S]*?)<\/testcase>)/g)];
  const decode = (text) => text.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&#10;/g, " ").replace(/&amp;/g, "&");
  for (const match of cases) {
    const attributes = match[1];
    const body = match[3] ?? "";
    const name = /[ ]name="([^"]*)"/.exec(attributes)?.[1] ?? "";
    const id = /^test_s(\d\d)_/.exec(name);
    const target = id ? scenarios.find((scenario) => scenario.id === `S${id[1]}`) : null;
    if (!target || target.skipped) continue;
    const failure = /<(failure|error)\b[^>]*message="([^"]*)"/.exec(body);
    const skipped = /<skipped\b/.test(body);
    const observed = /<property name="observed" value="([^"]*)"/.exec(body)?.[1] ?? "";
    const label = decode(name).replace(/^test_s\d\d_/, "").replace(/_/g, " ");
    if (skipped) continue;
    target.python.push({ ok: failure === null, description: label, observed: failure ? decode(failure[2]).slice(0, 400) : decode(observed) });
  }
  if (cases.length === 0 && !only) {
    const broken = new Scenario("PY", "Python checks of the recorded observations");
    broken.error = `no results from tests/e2e_check.py\n${pytestText.slice(-1500)}`;
    scenarios.push(broken);
  }
  const pythonCount = scenarios.reduce((sum, scenario) => sum + scenario.python.length, 0);
  console.log(`python checks: ${pythonCount} results${result.status === 0 ? "" : " (some failed)"}`);
} catch (error) {
  const failed = new Scenario("RUN", "The run itself");
  failed.error = error instanceof Error ? `${error.message}\n${(error.stack ?? "").split("\n").slice(1, 5).join("\n")}` : String(error);
  scenarios.push(failed);
} finally {
  await browser?.close().catch(() => {});
  for (const server of Object.values(servers)) await server.close().catch(() => {});
  for (const api of Object.values(apis)) api.stop();
}

const versions = (() => {
  const python = spawnSync(pythonPath(), ["--version"], { encoding: "utf-8" });
  return `${(python.stdout || python.stderr || "").trim()}; node ${process.version}; playwright chromium`;
})();
const header = [
  "PolicyJEV console end-to-end report",
  `Started ${started.toISOString()}, finished ${new Date().toISOString()}`,
  `Environment: ${versions}`,
  "Servers: two API processes started by tests/e2e_api.py on free ports (not 8000), with the real calibrated models.",
  "  A: JEV unavailable (the key lookup is replaced, no key file is read).  B: JEV answered by a stub that returns fixed probabilities.",
  "  Nothing in this run contacts TypeSafe or any other host; the browser reaches only the test servers on 127.0.0.1.",
  only ? `Only scenarios ${only.join(", ")} were selected.` : "All scenarios were selected.",
];
scenarios.sort((a, b) => a.id.localeCompare(b.id));
const text = formatReport(scenarios, header);
writeFileSync(REPORT, text);
const ran = scenarios.filter((scenario) => !scenario.skipped);
const failed = ran.filter((scenario) => !scenario.passed);
console.log(`\n${ran.length - failed.length} of ${ran.length} scenarios passed. Report: ${REPORT}`);
for (const scenario of failed) console.log(`  FAIL ${scenario.id} ${scenario.title}`);
process.exit(failed.length === 0 ? 0 : 1);
