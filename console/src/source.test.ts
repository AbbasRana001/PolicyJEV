import { renderHook } from "@testing-library/react";
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useTweenedNumber } from "./hooks";
import { mockMedia } from "./test-utils";

// Every source file under src, as text, except the captured data and this check itself.
const sources = import.meta.glob(["./**/*.ts", "./**/*.tsx", "./**/*.css", "../index.html", "../package.json", "../README.md"], {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const checked = Object.entries(sources).filter(([path]) => !path.includes("/fixtures/") && !path.endsWith("source.test.ts"));

const css = sources["./index.css"] ?? "";
const strip = (text: string): string => text.replace(/\/\*[\s\S]*?\*\//g, "");

describe("the source", () => {
  it("covers the files it should", () => {
    const names = checked.map(([path]) => path);
    expect(names).toContain("./App.tsx");
    expect(names).toContain("./index.css");
    expect(names).toContain("../index.html");
    expect(names).toContain("../package.json");
    expect(checked.length).toBeGreaterThan(20);
  });

  it("mentions no placeholder, prototype or simulation wording", () => {
    // "prototype" is flagged as a word; the JavaScript property (Storage.prototype) is not a mention of one
    const banned = /(?<!\.)\bprototype\b|simulat|lorem|TODO|FIXME/i;
    for (const [path, text] of checked) expect(banned.test(text), path).toBe(false);
  });

  it("makes no request to anything but the local API: no external addresses, fonts or scripts", () => {
    for (const [path, text] of checked) {
      const code = strip(text);
      expect(/https?:\/\/(?!127\.0\.0\.1|localhost)/.test(code), path).toBe(false);
      expect(/googleapis|gstatic|cdn\.|unpkg|jsdelivr/i.test(code), path).toBe(false);
      expect(/@import\s+url\(/.test(code), path).toBe(false);
    }
  });

  it("calls the API only through relative /api paths", () => {
    const api = sources["./api.ts"] ?? "";
    const paths = [...api.matchAll(/exchange\(\s*"([^"]+)"/g)].map((match) => match[1]);
    expect(paths.length).toBeGreaterThanOrEqual(4);
    for (const path of paths) expect(path?.startsWith("/api/")).toBe(true);
    // the only fetch call takes its address from exchange(), whose callers are the ones above
    expect(api.match(/\bfetch\(/g)?.length).toBe(1);
  });

  it("does not hold the data it draws: no copied threshold, count or probability tables", () => {
    const bannedNames = /\b(THR|FLOW|MET|CMP|PREC|REC|AP|RB)\s*=\s*[[{]/;
    for (const [path, text] of checked) {
      if (path.endsWith(".test.ts") || path.endsWith(".test.tsx")) continue;
      expect(bannedNames.test(text), path).toBe(false);
      expect(/function\s+(simulate|hash|sig)\b/.test(text), path).toBe(false);
    }
  });

  it("does not write the support floor into the page: the number of comments comes from the API", () => {
    for (const [path, text] of checked) {
      if (/.test.tsx?$/.test(path)) continue;
      expect(/30 comments/.test(text), path).toBe(false);
    }
  });

  it("has no footer note about the design and no switch to preview a key", () => {
    for (const [path, text] of checked) {
      expect(/Preview with a JEV key/i.test(text), path).toBe(false);
      expect(/Design prototype/i.test(text), path).toBe(false);
    }
  });

  it("writes the sentence that says why a comment got its decision in one place, used by Score and Policy", () => {
    const owners = checked.filter(([path, text]) => !/\.test\.tsx?$/.test(path) && text.includes("triggered by")).map(([path]) => path);
    expect(owners).toEqual(["./decision.ts"]);
    expect(sources["./components/Explorer.tsx"]).toContain("decisionWhy(");
    expect(sources["./components/ScoreView.tsx"]).toContain("decisionWhy(");
  });

  it("loads the two fonts from packages, not from a font service", () => {
    const main = sources["./main.tsx"] ?? "";
    expect(main).toContain("@fontsource-variable/bricolage-grotesque");
    expect(main).toContain("@fontsource/instrument-sans/400.css");
    expect(main).toContain("@fontsource/instrument-sans/500.css");
    expect(main).toContain("@fontsource/instrument-sans/600.css");
    const pkg = JSON.parse(sources["../package.json"] ?? "{}") as { dependencies?: Record<string, string> };
    expect(Object.keys(pkg.dependencies ?? {})).toEqual(
      expect.arrayContaining(["@fontsource-variable/bricolage-grotesque", "@fontsource/instrument-sans"]),
    );
  });
});

describe("the design tokens and responsive rules", () => {
  it("has the colour tokens of the design", () => {
    const tokens = {
      "--bg": "#121316",
      "--surface": "#191a1f",
      "--raised": "#23252c",
      "--allow": "#38d6a0",
      "--review": "#ffb224",
      "--action": "#ff5468",
      "--sys-lr": "#5b93ff",
      "--sys-db": "#b18cff",
      "--sys-jev": "#d6ff4d",
      "--accent": "#d6ff4d",
    };
    for (const [name, value] of Object.entries(tokens)) {
      expect(css.toLowerCase()).toContain(`${name}: ${value}`);
    }
  });

  it("changes layout at 1180, 1060, 900 and 760 pixels", () => {
    for (const width of [1180, 1060, 900, 760]) expect(css).toContain(`@media (max-width: ${width}px)`);
  });

  it("turns animations and transitions off when motion is reduced", () => {
    const block = css.slice(css.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(block).toContain("animation: none !important");
    expect(block).toContain("transition: none !important");
  });

  it("shows keyboard focus with an accent outline", () => {
    expect(css).toMatch(/:focus-visible\s*\{[^}]*outline:\s*2px solid var\(--accent\)/);
    expect(css).toMatch(/textarea:focus\s*\{[^}]*outline:\s*2px solid var\(--accent\)/);
  });

  it("does not let the page scroll sideways", () => {
    const layout = [...checked].filter(([path]) => path.endsWith("App.tsx") || path.includes("/components/"));
    for (const [path, text] of layout) {
      expect(/\boverflow-x-(scroll|auto)\b/.test(text), path).toBe(false);
      expect(/(^|\s)w-\[\d{4,}px\]/.test(text), path).toBe(false);
    }
    expect(css).not.toMatch(/body\s*\{[^}]*min-width/);
  });

  it("keeps the only wide-content rule on the table-like charts inside their own grid", () => {
    expect(css).toMatch(/\.wrap\s*\{[^}]*max-width:\s*1240px/);
  });
});

describe("motion", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("shows a number at once, without a count-up, when motion is reduced", () => {
    mockMedia({ reduced: true });
    const { result, rerender } = renderHook(({ value }) => useTweenedNumber(value, true), { initialProps: { value: 0 as number | null } });
    expect(result.current).toBe(0);
    rerender({ value: 0.75 });
    expect(result.current).toBe(0.75);
    rerender({ value: null });
    expect(result.current).toBeNull();
  });

  it("counts up to the target, ending exactly on it, when motion is allowed", async () => {
    vi.useFakeTimers({ toFake: ["requestAnimationFrame", "cancelAnimationFrame", "performance"] });
    const { result, rerender } = renderHook(({ value }) => useTweenedNumber(value, false, 100), { initialProps: { value: 0 as number | null } });
    rerender({ value: 1 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(40);
    });
    expect(result.current).toBeGreaterThan(0);
    expect(result.current).toBeLessThan(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(result.current).toBe(1);
  });
});
