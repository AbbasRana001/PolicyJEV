// Functions that run inside the page (passed to page.evaluate) and return what the screen shows.
// Each is self-contained: it may not refer to anything outside its own body.

/** The Score page: verdict tiles, per-label values, meters, alert and counters. */
export function readScoreUi() {
  const text = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : null);
  const out = { tiles: {}, cells: {}, alert: null, loadingNote: null, count: null, scoreDisabled: null, text: null };
  for (const tile of document.querySelectorAll("#view-score .vt")) {
    out.tiles[tile.dataset.sys] = {
      kind: tile.dataset.tile,
      decision: tile.dataset.decision ?? null,
      word: text(tile.querySelector(".word")),
      why: text(tile.querySelector(".why")),
      ms: text(tile.querySelector(".ms")),
      pressed: tile.getAttribute("aria-pressed"),
      role: tile.getAttribute("role"),
    };
  }
  for (const cell of document.querySelectorAll("#view-score .cell")) {
    const bars = [...cell.querySelectorAll(".meter .c")];
    const zone = (c) => (c.classList.contains("zx") ? "zx" : c.classList.contains("zr") ? "zr" : "za");
    out.cells[`${cell.dataset.sys}|${cell.dataset.l}`] = {
      val: text(cell.querySelector(".val")),
      lit: bars.filter((c) => c.classList.contains("lit")).length,
      peak: bars.filter((c) => c.classList.contains("peak")).length,
      zones: bars.map(zone),
      th: text(cell.querySelector(".th")),
      display: getComputedStyle(cell).display,
    };
  }
  const alert = document.querySelector("#view-score .alert");
  out.alert = alert ? text(alert) : null;
  out.loadingNote = text(document.querySelector("#view-score .loading-note"));
  out.limit = text(document.querySelector("#view-score .limit"));
  out.count = text(document.querySelector("#score-count"));
  const button = document.querySelector("#view-score button.primary");
  out.scoreDisabled = button ? button.disabled : null;
  out.scoreBusy = button ? button.getAttribute("aria-busy") : null;
  out.text = document.querySelector("#score-text")?.value.length ?? null;
  out.selected = document.querySelector("#view-score .matrix")?.dataset.sel ?? null;
  return out;
}

/** The threshold explorer on the Policy page. */
export function readExplorer() {
  const text = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : null);
  const outcome = document.querySelector(".explore .outcome");
  const slider = document.querySelector("#probe");
  const rows = [...document.querySelectorAll(".explore .xr[data-l]")].map((row) => ({
    label: row.dataset.l,
    decision: row.dataset.decision,
    chip: text(row.querySelector(".dchip")),
    scale: text(row.querySelector(".xsc")),
    note: text(row.querySelector(".nt")),
    widths: [...row.querySelectorAll(".zs > div")].map((z) => ({ zone: z.dataset.z, width: parseFloat(z.style.width), hit: z.classList.contains("hit") })),
    marker: row.querySelector(".mk")?.style.left ?? null,
  }));
  return {
    system: text(document.querySelector('.explore [role="tab"][aria-selected="true"]')),
    probabilityText: text(document.querySelector("[data-probability]")),
    slider: slider ? Number(slider.value) : null,
    valuetext: slider?.getAttribute("aria-valuetext") ?? null,
    outcome: outcome ? { decision: outcome.dataset.outcome ?? null, text: text(outcome), live: outcome.getAttribute("aria-live") } : null,
    rows,
  };
}

/** Everything on the Policy page below the explorer (the visible dumbbell plot only). */
export function readPolicyCharts() {
  const text = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : null);
  const secs = [...document.querySelectorAll("#view-policy .sec")];
  const out = {};
  out.sections = secs.map((sec) => ({ h2: text(sec.querySelector("h2")), sub: text(sec.querySelector(".sub")) }));
  out.flows = [...document.querySelectorAll("#flow .fl")].map((fl) => ({
    system: fl.dataset.system,
    legend: text(fl.querySelector(".fl-n")),
    bars: [...fl.querySelectorAll(".fl-bar > div")].map((bar) => ({ flow: bar.dataset.flow, width: parseFloat(bar.style.width), text: text(bar) })),
  }));
  out.measures = [...document.querySelectorAll("#metrics .mr")].map((mr) => ({
    key: mr.dataset.measure,
    scale: [...mr.querySelectorAll(".scale span")].map((t) => text(t)),
    name: text(mr.querySelector(".mn b")),
    values: [...mr.querySelectorAll(".vc")].map((vc) => ({ system: vc.dataset.system, text: text(vc), best: vc.dataset.best === "true" })),
    dots: [...mr.querySelectorAll(".pd")].map((dot) => ({ system: dot.dataset.system, f: parseFloat(dot.style.getPropertyValue("--f")), best: dot.classList.contains("best") })),
  }));
  out.comparison = [...document.querySelectorAll("#cmp .cr")].map((cr) => ({
    measure: cr.dataset.measure,
    name: text(cr.querySelector(".cn b")),
    hint: text(cr.querySelector(".cn span")),
    lines: [...cr.querySelectorAll(".nn p")].map((p) => ({ b: text(p.querySelector("b")), full: text(p) })),
    dots: [...cr.querySelectorAll(".lane .dt")].map((d) => parseFloat(d.style.left)),
    intervals: [...cr.querySelectorAll(".lane .iv")].map((iv) => ({ left: parseFloat(iv.style.left), width: parseFloat(iv.style.width) })),
  }));
  out.reads = [...document.querySelectorAll("#view-policy .reads p")].map((p) => ({ tone: p.dataset.tone, text: text(p) }));
  out.crhead = text(document.querySelector("#view-policy .crhead"));
  const dumbbells = (id) =>
    [...document.querySelectorAll(`#${id} .fp:not(.gh)`)].map((row) => ({
      key: row.dataset.row,
      label: text(row.querySelector(".rl")),
      validation: row.querySelector(".d.v")?.getAttribute("title") ?? null,
      test: row.querySelector(".d.s")?.getAttribute("title") ?? null,
      change: text(row.querySelector(".dl")),
      up: row.querySelector(".dl")?.classList.contains("up") ?? null,
      target: row.querySelector(".tgt") ? parseFloat(row.querySelector(".tgt").style.left) : null,
      positions: [row.querySelector(".d.v"), row.querySelector(".d.s")].map((d) => (d ? parseFloat(d.style.left) : null)),
    }));
  out.dumbbells = { precision: dumbbells("fp-precision"), recall: dumbbells("fp-recall") };
  out.dumbbellTicks = [...document.querySelectorAll("#view-policy .axis .t span")].map((s) => text(s));
  out.trust = text(document.querySelector("#view-policy .finding .sub"));
  out.notes = [...document.querySelectorAll("#view-policy .note[data-note]")].map((n) => ({ note: n.dataset.note, text: text(n) }));
  out.flags = [...document.querySelectorAll("#metrics .mr")].filter((mr) => mr.querySelector("[data-flag]")).map((mr) => mr.dataset.measure);
  out.comparisonHeader = text(document.querySelector("#view-policy .crhead .signed"));
  out.headings = [...document.querySelectorAll("#view-policy h2")].map((h) => text(h));
  out.scales = Object.fromEntries([...document.querySelectorAll("#metrics .mr")].map((mr) => [mr.dataset.measure, [...mr.querySelectorAll(".scale span")].map((t) => text(t))]));
  out.sliderLabel = text(document.querySelector("#view-policy .xscrub .lbl"));
  out.pillNotes = [...new Set([...document.querySelectorAll("#view-policy .explore .xend .nt")].map((n) => text(n)))];
  out.pillNoteColour = document.querySelector("#view-policy .explore .xend .nt") ? getComputedStyle(document.querySelector("#view-policy .explore .xend .nt")).color : null;
  out.legendAction = text(document.querySelector("#view-policy .zlegend p.zx"));
  return out;
}

/** The Results page: bars, intervals, line-chart points and the sentences. */
export function readResultsPage() {
  const text = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : null);
  const out = {};
  const secs = [...document.querySelectorAll("#view-results .sec")];
  out.sections = secs.map((sec) => ({ h2: text(sec.querySelector("h2")), sub: text(sec.querySelector(".sub")) }));
  out.lede = text(document.querySelector("#view-results .lede"));
  out.spacingNote = text(document.querySelector('#view-results [data-note="budget-spacing"]')) ?? "";
  out.yTitle = text(document.querySelector("#lc [data-axis-title]"));
  out.bars = [...document.querySelectorAll("#apc .apb")].map((bar) => {
    const group = bar.closest(".apg");
    const whisker = bar.querySelector(".wh");
    return {
      label: group.dataset.label,
      system: bar.dataset.system,
      title: bar.getAttribute("title"),
      tag: text(bar.querySelector(".vv")),
      height: parseFloat(bar.style.height),
      whiskerBottom: whisker ? parseFloat(whisker.style.bottom) : null,
      whiskerHeight: whisker ? parseFloat(whisker.style.height) : null,
    };
  });
  out.groups = [...document.querySelectorAll("#apc .apg")].map((g) => ({ label: g.dataset.label, name: text(g.querySelector(".gl")), positives: text(g.querySelector(".gn")) }));
  const svg = document.querySelector("#lc svg");
  const view = svg ? svg.viewBox.baseVal : null;
  out.svg = view ? { width: view.width, height: view.height } : null;
  out.points = [...document.querySelectorAll("#lc circle")].map((c) => ({
    system: c.closest("g").dataset.system,
    level: Number(c.dataset.level),
    value: Number(c.dataset.value),
    title: c.querySelector("title")?.textContent ?? null,
    cx: Number(c.getAttribute("cx")),
    cy: Number(c.getAttribute("cy")),
  }));
  out.axisTexts = [...document.querySelectorAll("#lc text")].filter((t) => !t.dataset.directLabel).map((t) => text(t));
  out.leaders = [...document.querySelectorAll("#lc line[data-leader]")].map((l) => ({ system: l.dataset.leader, x1: Number(l.getAttribute("x1")), y1: Number(l.getAttribute("y1")) }));
  out.directLabels = [...document.querySelectorAll("#lc text[data-direct-label]")].map((t) => ({ system: t.dataset.directLabel, text: text(t) }));
  return out;
}

/** Text that is cut off, runs past the screen, or sits outside a box that hides what overflows it. */
export function findLayoutProblems() {
  const problems = [];
  const visible = (el) => {
    const style = getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) !== 0;
  };
  const isHiddenOnPurpose = (el) => {
    for (let node = el; node && node !== document.body; node = node.parentElement) {
      if (node.classList?.contains("sr-only")) return true;
      if (node.hasAttribute?.("hidden")) return true;
      if (node.getAttribute?.("aria-hidden") === "true" && node.tagName === "svg") return false;
    }
    return false;
  };
  const clipAncestors = (el) => {
    const list = [];
    for (let node = el.parentElement; node && node !== document.documentElement; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.overflowX !== "visible" || style.overflowY !== "visible") list.push(node);
    }
    return list;
  };
  const describe = (el) => `${el.tagName.toLowerCase()}${el.id ? `#${el.id}` : ""}.${String(el.className?.baseVal ?? el.className).split(" ").filter(Boolean).slice(0, 2).join(".")}`;
  const seen = new Set();
  const report = (el, why, rect, extra = "") => {
    const key = `${describe(el)}|${why}`;
    if (seen.has(key)) return;
    seen.add(key);
    problems.push({ element: describe(el), why, text: (el.textContent ?? "").replace(/\s+/g, " ").trim().slice(0, 40), left: Math.round(rect.left), right: Math.round(rect.right), extra });
  };
  const check = (el, rect) => {
    if (rect.width === 0 || rect.height === 0) return;
    if (rect.right > window.innerWidth + 1) report(el, "runs past the right edge of the screen", rect, `innerWidth ${window.innerWidth}`);
    if (rect.left < -1) report(el, "runs past the left edge of the screen", rect);
    for (const box of clipAncestors(el)) {
      const b = box.getBoundingClientRect();
      const style = getComputedStyle(box);
      const clipsX = style.overflowX !== "visible";
      const clipsY = style.overflowY !== "visible";
      if (clipsX && (rect.left < b.left - 1 || rect.right > b.right + 1)) report(el, `cut off horizontally by ${describe(box)}`, rect);
      if (clipsY && (rect.top < b.top - 1 || rect.bottom > b.bottom + 1)) report(el, `cut off vertically by ${describe(box)}`, rect);
    }
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!node.nodeValue || node.nodeValue.trim() === "") continue;
    const el = node.parentElement;
    if (!el || !visible(el) || isHiddenOnPurpose(el)) continue;
    if (["SCRIPT", "STYLE", "TITLE"].includes(el.tagName)) continue;
    const isSvgText = el.namespaceURI === "http://www.w3.org/2000/svg";
    let rect;
    if (isSvgText) {
      rect = el.getBoundingClientRect();
    } else {
      const range = document.createRange();
      range.selectNodeContents(node);
      rect = range.getBoundingClientRect();
    }
    check(el, rect);
    // the text does not fit the box that holds it and the box hides the rest
    if (!isSvgText) {
      const style = getComputedStyle(el);
      if ((style.overflowX === "hidden" || style.overflowX === "clip") && el.scrollWidth > el.clientWidth + 1) {
        report(el, "wider than its own box, which hides the rest", rect, `scrollWidth ${el.scrollWidth} > clientWidth ${el.clientWidth}`);
      }
      if (style.textOverflow === "ellipsis" && el.scrollWidth > el.clientWidth + 1) report(el, "shortened with an ellipsis", rect);
    }
  }
  return {
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    bodyScrollWidth: document.body.scrollWidth,
    problems,
  };
}

/** Every focusable thing now on screen, with whether it can be reached by Tab. */
export function listFocusable() {
  const selector = 'button, input, textarea, select, a[href], [role="tab"], [role="switch"], [role="button"], [tabindex]';
  return [...document.querySelectorAll(selector)]
    .filter((el) => {
      const style = getComputedStyle(el);
      const box = el.getBoundingClientRect();
      return style.display !== "none" && style.visibility !== "hidden" && box.width > 0 && box.height > 0 && !el.closest("[hidden]");
    })
    .map((el) => ({
      id: el.id || null,
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute("role"),
      label: (el.getAttribute("aria-label") ?? el.textContent ?? "").replace(/\s+/g, " ").trim().slice(0, 40),
      tabIndex: el.tabIndex,
      disabled: Boolean(el.disabled),
      sel: el.id ? `#${el.id}` : null,
    }));
}

/** What the browser focus is on, and how it shows. */
export function describeFocus() {
  const el = document.activeElement;
  if (!el || el === document.body) return { body: true };
  const style = getComputedStyle(el);
  const box = el.getBoundingClientRect();
  return {
    body: false,
    tag: el.tagName.toLowerCase(),
    id: el.id || null,
    role: el.getAttribute("role"),
    label: (el.getAttribute("aria-label") ?? el.textContent ?? "").replace(/\s+/g, " ").trim().slice(0, 40),
    outlineStyle: style.outlineStyle,
    outlineWidth: style.outlineWidth,
    outlineColor: style.outlineColor,
    boxShadow: style.boxShadow,
    width: Math.round(box.width),
    height: Math.round(box.height),
    visibleOnScreen: box.bottom > 0 && box.top < window.innerHeight,
  };
}
