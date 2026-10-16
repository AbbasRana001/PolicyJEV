import { describe, expect, it } from "vitest";

import css from "./index.css?raw";

/*
 * Small-text colours must reach 4.5:1 on every surface they are used on. The surfaces, including the
 * lightest the page glow and the tinted tiles make, are worked out from the tokens in index.css, and
 * every rule that sets a text colour from one of the neutral tokens is pinned to the surfaces it sits on,
 * so adding a new use without saying where it sits fails here.
 */

type Rgb = [number, number, number];

function tokens(): Record<string, string> {
  const found: Record<string, string> = {};
  for (const match of css.matchAll(/--([a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) found[match[1] ?? ""] = match[2] ?? "";
  return found;
}

const colour = tokens();

function rgb(name: string): Rgb {
  const hex = colour[name];
  if (hex === undefined) throw new Error(`no token --${name}`);
  return [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16)) as Rgb;
}

function mix(top: Rgb, below: Rgb, alpha: number): Rgb {
  return top.map((value, i) => value * alpha + (below[i] ?? 0) * (1 - alpha)) as Rgb;
}

function luminance(c: Rgb): number {
  const linear = (v: number): number => {
    const x = v / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * linear(c[0]) + 0.7152 * linear(c[1]) + 0.0722 * linear(c[2]);
}

function ratio(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return ((hi ?? 0) + 0.05) / ((lo ?? 0) + 0.05);
}

const SURFACES: Record<string, Rgb[]> = {
  // the page, at its plainest and under the strongest part of each background glow (7% lime, 8% blue)
  page: [rgb("bg"), mix([214, 255, 77], rgb("bg"), 0.07), mix([91, 147, 255], rgb("bg"), 0.08)],
  panel: [rgb("surface")],
  raised: [rgb("raised")],
  // a measure track: raised with the accent at 10% where "better" is printed
  track: [mix(rgb("accent"), rgb("raised"), 0.1)],
  // tiles and the outcome capsule are tinted by the decision colour
  tinted: ["allow", "review", "action"].flatMap((zone) => [
    mix(rgb(zone), rgb("surface"), 0.11),
    mix(rgb(zone), rgb("raised"), 0.13),
  ]),
};

/** Rules that set a colour from a neutral token, with the surface each sits on. */
const FAINT_USES: Record<string, (keyof typeof SURFACES)[]> = {
  "textarea::placeholder": ["raised"],
  ".primary:disabled": ["raised"],
  ".notice": ["panel"],
  ".vt.na .word": ["page"],
  ".vt.idle .word,\n  .vt.pend .word,\n  .vt.wait .word": ["panel", "page"],
  ".cell.na .val": ["page"],
  ".cell .th": ["page"],
  ".note": ["page"],
  ".xsc .ro": ["panel"],
  ".crhead": ["page"],
  ".axis": ["page"],
  ".apg .gn": ["page"],
};

function rulesUsing(token: string): string[] {
  const selectors: string[] = [];
  for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (new RegExp(`(^|[;\\s])color:\\s*var\\(--${token}\\)`).test(match[2] ?? "")) {
      selectors.push((match[1] ?? "").trim().replace(/\n\s*/g, "\n  "));
    }
  }
  return selectors;
}

describe("text colour contrast", () => {
  it("reads the tokens it needs from the stylesheet", () => {
    for (const name of ["bg", "surface", "raised", "text", "muted", "faint", "accent", "allow", "review", "action"]) {
      expect(colour[name], name).toMatch(/^#[0-9a-fA-F]{6}$/);
    }
  });

  it("has faint at the agreed value", () => {
    expect(colour["faint"]?.toLowerCase()).toBe("#8a8c9c");
  });

  it("pins every rule that uses --faint as a text colour to the surface it sits on", () => {
    expect(rulesUsing("faint").sort()).toEqual(Object.keys(FAINT_USES).sort());
  });

  it("keeps --faint at 4.5:1 or more on each surface it is used on", () => {
    for (const [selector, surfaces] of Object.entries(FAINT_USES)) {
      for (const surface of surfaces) {
        for (const background of SURFACES[surface] ?? []) {
          expect(ratio(rgb("faint"), background), `${selector} on ${surface}`).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });

  it("keeps --muted and --text at 4.5:1 or more on every surface, including the tinted ones", () => {
    for (const token of ["muted", "text"]) {
      for (const [surface, backgrounds] of Object.entries(SURFACES)) {
        for (const background of backgrounds) {
          expect(ratio(rgb(token), background), `${token} on ${surface}`).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });

  it("puts the labels on the measure tracks, which sit on a tinted surface, in a colour that passes there", () => {
    const uses = rulesUsing("faint");
    expect(uses).not.toContain(".dtrack .end");
    expect(rulesUsing("muted")).toContain(".dtrack .end");
    for (const background of SURFACES["track"] ?? []) expect(ratio(rgb("muted"), background)).toBeGreaterThanOrEqual(4.5);
  });

  it("would fail if --faint dropped back to the old value", () => {
    const old: Rgb = [0x6e, 0x70, 0x80];
    const failing = Object.values(FAINT_USES)
      .flat()
      .some((surface) => (SURFACES[surface] ?? []).some((background) => ratio(old, background) < 4.5));
    expect(failing).toBe(true);
  });

  it("agrees with a known pair (white on black is 21:1)", () => {
    expect(ratio([255, 255, 255], [0, 0, 0])).toBeCloseTo(21, 5);
  });
});
