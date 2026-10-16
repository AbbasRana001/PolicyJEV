// Checks the two self-hosted font families in Chromium: that each loads from its package files,
// that the weight axis works, and whether tabular figures change digit widths (so numbers that
// change do not jitter). Run from console/: node scripts/fontcheck.mjs
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const families = [
  {
    name: "Instrument Sans",
    file: "node_modules/@fontsource/instrument-sans/files/instrument-sans-latin-400-normal.woff2",
    weights: "400",
    low: 400,
    high: 600,
    extraFiles: ["500", "600"].map((weight) => ({
      weight,
      file: `node_modules/@fontsource/instrument-sans/files/instrument-sans-latin-${weight}-normal.woff2`,
    })),
  },
  {
    name: "Bricolage Grotesque Variable",
    file: "node_modules/@fontsource-variable/bricolage-grotesque/files/bricolage-grotesque-latin-opsz-normal.woff2",
    weights: "200 800",
    low: 500,
    high: 700,
    extraFiles: [],
  },
];

function face({ name, file, weights }, weight = weights) {
  const data = readFileSync(file).toString("base64");
  return `@font-face{font-family:"${name}";src:url(data:font/woff2;base64,${data}) format("woff2");font-weight:${weight}}`;
}

const css = families
  .flatMap((family) => [
    face(family),
    ...family.extraFiles.map((extra) =>
      face({ ...family, file: extra.file }, extra.weight),
    ),
  ])
  .join("");

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<style>${css} span{font-size:40px;white-space:nowrap}</style><body></body>`);

let failed = false;
for (const family of families) {
  const result = await page.evaluate(
    async ({ name, low, high }) => {
      await document.fonts.load(`${low} 40px "${name}"`);
      await document.fonts.load(`${high} 40px "${name}"`);
      const measure = (text, features, weight) => {
        const el = document.createElement("span");
        el.textContent = text;
        el.style.fontFamily = `"${name}", monospace`;
        el.style.fontVariantNumeric = features;
        el.style.fontWeight = String(weight);
        document.body.appendChild(el);
        const width = el.getBoundingClientRect().width;
        el.remove();
        return width;
      };
      return {
        loaded: document.fonts.check(`${low} 40px "${name}"`),
        defaultOnes: measure("1111111111", "normal", low),
        defaultZeros: measure("0000000000", "normal", low),
        tabularOnes: measure("1111111111", "tabular-nums", low),
        tabularZeros: measure("0000000000", "tabular-nums", low),
        lowWeightWidth: measure("Human review", "normal", low),
        highWeightWidth: measure("Human review", "normal", high),
      };
    },
    family,
  );
  const changes =
    Math.abs(result.defaultOnes - result.tabularOnes) > 0.5 || Math.abs(result.defaultZeros - result.tabularZeros) > 0.5;
  const equal = Math.abs(result.tabularOnes - result.tabularZeros) < 0.5;
  const alreadyEqual = Math.abs(result.defaultOnes - result.defaultZeros) < 0.5;
  const weightWorks = Math.abs(result.lowWeightWidth - result.highWeightWidth) > 0.5;
  console.log(
    family.name,
    JSON.stringify(result),
    `\n  loaded: ${result.loaded}; weights ${family.low} and ${family.high} differ: ${weightWorks};` +
      ` tabular-nums changes widths: ${changes}; equal-width digits with tnum: ${equal}; equal-width digits by default: ${alreadyEqual}`,
  );
  if (!result.loaded || !weightWorks) failed = true;
}
await browser.close();
process.exit(failed ? 1 : 0);
