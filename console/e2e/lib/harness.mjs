// Infrastructure for the end-to-end run: free ports, the API processes, the console server (built
// files plus a same-origin proxy to one API), and a snapshot of the repository's files.
import { spawn, spawnSync } from "node:child_process";
import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createServer as createNetServer } from "node:net";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const CONSOLE = resolve(fileURLToPath(new URL("../..", import.meta.url)));
export const REPO = resolve(CONSOLE, "..");
export const DIST = join(CONSOLE, "dist");
export const REAL_SERVER_PORT = 8000;

export const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

/** The Python that runs the repository's code: E2E_PYTHON, the repository's .venv, or python on the path. */
export function pythonPath() {
  if (process.env.E2E_PYTHON) return process.env.E2E_PYTHON;
  for (const candidate of [join(REPO, ".venv", "Scripts", "python.exe"), join(REPO, ".venv", "bin", "python")]) {
    if (existsSync(candidate)) return candidate;
  }
  return "python";
}

/** A free port on the loopback interface, never the port of the real server. */
export async function freePort() {
  for (;;) {
    const port = await new Promise((resolvePort, reject) => {
      const probe = createNetServer();
      probe.once("error", reject);
      probe.listen(0, "127.0.0.1", () => {
        const { port: chosen } = probe.address();
        probe.close(() => resolvePort(chosen));
      });
    });
    if (port !== REAL_SERVER_PORT) return port;
  }
}

function killTree(child) {
  if (child.exitCode !== null || child.killed) return;
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else child.kill("SIGKILL");
}

/** Start tests/e2e_api.py. The process output is kept in memory (never printed with typed text in it). */
export function startApi({ name, port, mode = "real", jev = "off", warmDelay = 0, jevDelay = 0 }) {
  const args = [
    join(REPO, "tests", "e2e_api.py"),
    "--port", String(port), "--mode", mode, "--jev", jev,
    "--warm-delay", String(warmDelay), "--jev-delay", String(jevDelay),
  ];
  const child = spawn(pythonPath(), args, {
    cwd: REPO,
    env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  const logs = [];
  const take = (stream) => (chunk) => logs.push({ stream, text: chunk.toString("utf8") });
  child.stdout.on("data", take("stdout"));
  child.stderr.on("data", take("stderr"));
  const api = {
    name,
    port,
    mode,
    jev,
    child,
    logs,
    url: `http://127.0.0.1:${port}`,
    text: () => logs.map((entry) => entry.text).join(""),
    exited: () => child.exitCode !== null,
    stop: () => killTree(child),
    /** How many times the JEV stub has been called (it logs only the length of the text). */
    jevCalls: () => (api.text().match(/\[e2e\] jev stub called/g) ?? []).length,
  };
  return api;
}

export async function getJson(url, init) {
  const response = await fetch(url, init);
  const text = await response.text();
  let body = null;
  try {
    body = JSON.parse(text);
  } catch {
    // keep null
  }
  return { status: response.status, body, text };
}

/** Wait until the API answers health and the predicate accepts the answer. */
export async function waitForHealth(api, accept, timeoutMs = 240_000) {
  const started = Date.now();
  let last = null;
  while (Date.now() - started < timeoutMs) {
    if (api.exited()) throw new Error(`${api.name} exited early:\n${api.text().slice(-1500)}`);
    try {
      const { status, body } = await getJson(`${api.url}/api/health`);
      if (status === 200 && body) {
        last = body;
        if (accept(body)) return body;
      }
    } catch {
      // not listening yet
    }
    await sleep(500);
  }
  throw new Error(`${api.name} did not become ready: ${JSON.stringify(last)}`);
}

export const localReady = (body) =>
  body.systems.logreg_word.status === "available" && body.systems.distilbert.status === "available";

const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2",
  ".woff": "font/woff", ".svg": "image/svg+xml", ".json": "application/json", ".ico": "image/x-icon",
};

/**
 * Serve the built console and forward /api/* to one API, as the development server does.
 * The upstream can be switched off (connections are dropped) to look like a dead server.
 */
export async function startConsoleServer({ upstreamPort, label }) {
  const state = { upstreamPort, requests: [] };
  const server = createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    state.requests.push({ method: request.method, path: url.pathname });
    if (url.pathname.startsWith("/api/")) {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      if (state.upstreamPort === null) {
        request.socket.destroy();
        return;
      }
      const hasBody = request.method !== "GET" && request.method !== "HEAD";
      try {
        const upstream = await fetch(`http://127.0.0.1:${state.upstreamPort}${url.pathname}${url.search}`, {
          method: request.method,
          headers: { "content-type": request.headers["content-type"] ?? "application/json" },
          body: hasBody ? Buffer.concat(chunks) : undefined,
        });
        response.writeHead(upstream.status, { "content-type": upstream.headers.get("content-type") ?? "application/json" });
        response.end(Buffer.from(await upstream.arrayBuffer()));
      } catch {
        request.socket.destroy();
      }
      return;
    }
    let file = normalize(join(DIST, url.pathname === "/" ? "index.html" : url.pathname));
    if (!file.startsWith(DIST + sep) && file !== DIST) file = join(DIST, "index.html");
    if (!existsSync(file) || statSync(file).isDirectory()) file = join(DIST, "index.html");
    response.setHeader("Content-Type", TYPES[extname(file)] ?? "application/octet-stream");
    createReadStream(file).pipe(response);
  });
  const port = await freePort();
  await new Promise((done) => server.listen(port, "127.0.0.1", done));
  return {
    label,
    port,
    origin: `http://127.0.0.1:${port}`,
    host: `127.0.0.1:${port}`,
    requests: state.requests,
    setUpstream: (next) => {
      state.upstreamPort = next;
    },
    close: () => new Promise((done) => {
      server.closeAllConnections?.();
      server.close(done);
    }),
  };
}

const SKIP_DIRS = new Set([".git", "node_modules", ".venv", "dist"]);

/** path -> {mtimeMs, size} for every file under the repository (stat only). */
export function snapshotRepo(root = REPO) {
  const files = new Map();
  const walk = (dir) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) walk(path);
      } else if (entry.isFile()) {
        try {
          const info = statSync(path);
          files.set(path, { mtimeMs: info.mtimeMs, size: info.size });
        } catch {
          // vanished
        }
      }
    }
  };
  walk(root);
  return files;
}

/** Files that are new, or whose time or size changed, between two snapshots. */
export function changedFiles(before, after) {
  const changed = [];
  for (const [path, info] of after) {
    const old = before.get(path);
    if (old === undefined || old.mtimeMs !== info.mtimeMs || old.size !== info.size) changed.push(path);
  }
  return changed;
}

/** The needles that occur in a file's bytes (UTF-8 and UTF-16), reading at most 200 MB. */
export function needlesIn(path, needles) {
  let data;
  try {
    if (statSync(path).size > 200 * 1024 * 1024) return ["(file too large to scan)"];
    data = readFileSync(path);
  } catch {
    return [];
  }
  const hits = [];
  for (const needle of needles) {
    if (data.includes(Buffer.from(needle, "utf8")) || data.includes(Buffer.from(needle, "utf16le"))) hits.push("sentinel");
  }
  return hits;
}
