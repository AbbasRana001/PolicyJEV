import type { CSSProperties } from "react";

import type { SystemId } from "./api";

/** Colour of each system in charts and chips (CSS variables from index.css). */
export const SYSTEM_COLOUR: Record<SystemId, string> = {
  logreg_word: "var(--sys-lr)",
  distilbert: "var(--sys-db)",
  jev: "var(--sys-jev)",
};

/** Short system names for the narrow verdict tiles. */
export const SYSTEM_SHORT: Record<SystemId, string> = {
  logreg_word: "Logistic",
  distilbert: "DistilBERT",
  jev: "JEV",
};

/** Inline style that sets the --c custom property used by the chart classes. */
export function colourStyle(colour: string, extra: CSSProperties = {}): CSSProperties {
  return { "--c": colour, ...extra } as CSSProperties;
}

export function systemStyle(system: SystemId, extra: CSSProperties = {}): CSSProperties {
  return colourStyle(SYSTEM_COLOUR[system], extra);
}

/** A CSS custom property set inline, typed so the call sites stay free of casts. */
export function cssVars(vars: Record<string, string | number>): CSSProperties {
  return vars as unknown as CSSProperties;
}
