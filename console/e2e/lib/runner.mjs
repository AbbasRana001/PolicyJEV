// A small scenario runner: each scenario records checks (pass or fail, with what was observed) and
// the report is plain text. Nothing typed into the console is ever written here.

export class Scenario {
  constructor(id, title) {
    this.id = id;
    this.title = title;
    this.checks = [];
    this.notes = [];
    this.error = null;
    this.python = [];
    this.seconds = 0;
  }

  /** Record one assertion. `observed` is what was seen, in words or numbers. */
  check(condition, description, observed = "") {
    this.checks.push({ ok: Boolean(condition), description, observed: String(observed) });
    return Boolean(condition);
  }

  /** Record a value that is reported but not judged. */
  note(text) {
    this.notes.push(text);
  }

  get failures() {
    const own = this.checks.filter((entry) => !entry.ok).length;
    const py = this.python.filter((entry) => !entry.ok).length;
    return own + py + (this.error === null ? 0 : 1);
  }

  get passed() {
    return this.failures === 0 && (this.checks.length + this.python.length) > 0;
  }
}

/** Run scenarios in order; an exception in one is recorded as its failure and the run goes on. */
export async function runScenarios(list, ctx, only) {
  const scenarios = [];
  for (const { id, title, run } of list) {
    const scenario = new Scenario(id, title);
    scenarios.push(scenario);
    if (only && !only.includes(id)) {
      scenario.note("skipped (not selected)");
      scenario.skipped = true;
      continue;
    }
    const started = Date.now();
    process.stdout.write(`  ${id} ${title} ...`);
    try {
      await run(scenario, ctx);
    } catch (error) {
      scenario.error = error instanceof Error ? `${error.message}\n${(error.stack ?? "").split("\n").slice(1, 4).join("\n")}` : String(error);
    }
    scenario.seconds = (Date.now() - started) / 1000;
    process.stdout.write(` ${scenario.passed ? "PASS" : "FAIL"} (${scenario.seconds.toFixed(1)}s)\n`);
  }
  return scenarios;
}

const indent = (text, by) => text.split("\n").map((line) => `${" ".repeat(by)}${line}`).join("\n");

export function formatReport(scenarios, header) {
  const lines = [...header, ""];
  const ran = scenarios.filter((s) => !s.skipped);
  lines.push(`Summary: ${ran.filter((s) => s.passed).length} of ${ran.length} scenarios passed`);
  const totalChecks = ran.reduce((sum, s) => sum + s.checks.length + s.python.length, 0);
  const failedChecks = ran.reduce((sum, s) => sum + s.failures, 0);
  lines.push(`Checks: ${totalChecks} run, ${failedChecks} failed`);
  lines.push("");
  for (const s of scenarios) {
    if (s.skipped) {
      lines.push(`SKIP  ${s.id}  ${s.title}`, "");
      continue;
    }
    const total = s.checks.length + s.python.length;
    lines.push(`${s.passed ? "PASS" : "FAIL"}  ${s.id}  ${s.title}  (${total} checks, ${s.seconds.toFixed(1)}s)`);
    if (s.error !== null) lines.push(indent(`ERROR ${s.error}`, 4));
    for (const entry of s.checks) {
      lines.push(indent(`${entry.ok ? "ok  " : "FAIL"}  ${entry.description}${entry.observed === "" ? "" : `: ${entry.observed}`}`, 4));
    }
    for (const entry of s.python) {
      lines.push(indent(`${entry.ok ? "ok  " : "FAIL"}  [python] ${entry.description}${entry.observed === "" ? "" : `: ${entry.observed}`}`, 4));
    }
    for (const note of s.notes) lines.push(indent(`note  ${note}`, 4));
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}
