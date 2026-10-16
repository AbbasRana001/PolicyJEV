import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

import { SYSTEM_IDS, getHealth } from "./api";
import type { Health } from "./api";

/** True while the media query matches; false where matchMedia is not available. */
export function useMediaQuery(query: string): boolean {
  const read = (): boolean =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(query).matches
      : false;
  const [matches, setMatches] = useState<boolean>(read);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const list = window.matchMedia(query);
    const onChange = (): void => setMatches(list.matches);
    onChange();
    list.addEventListener("change", onChange);
    return () => list.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

export function useReducedMotion(): boolean {
  return useMediaQuery("(prefers-reduced-motion: reduce)");
}

/** Whether the API answered its last health check; "checking" until the first answer. */
export type ApiState = "checking" | "online" | "offline";

export interface HealthState {
  api: ApiState;
  /** Per-system state from the health response; null until the API has answered. */
  systems: Health | null;
  /** Check again now, for example after a request failed. */
  refresh: () => void;
}

const POLL_LOADING_MS = 2000;
const POLL_OFFLINE_MS = 4000;
const POLL_STEADY_MS = 15000;

/** API reachability and per-system health: polled quickly while loading or unreachable, slowly otherwise. */
export function useHealth(): HealthState {
  const [api, setApi] = useState<ApiState>("checking");
  const [systems, setSystems] = useState<Health | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const active = useRef(true);
  const pollRef = useRef<() => Promise<void>>(async () => {});

  pollRef.current = async (): Promise<void> => {
    window.clearTimeout(timer.current);
    const result = await getHealth();
    if (!active.current) return;
    let delay = POLL_STEADY_MS;
    if (!result.reachable) {
      setApi("offline");
      delay = POLL_OFFLINE_MS;
    } else {
      setApi("online");
      setSystems(result.systems);
      if (result.systems === null || SYSTEM_IDS.some((id) => result.systems?.[id].status === "loading")) {
        delay = POLL_LOADING_MS;
      }
    }
    timer.current = window.setTimeout(() => void pollRef.current(), delay);
  };

  useEffect(() => {
    active.current = true;
    void pollRef.current();
    return () => {
      active.current = false;
      window.clearTimeout(timer.current);
    };
  }, []);

  const refresh = useCallback(() => {
    void pollRef.current();
  }, []);

  return { api, systems, refresh };
}

/**
 * Load a resource once the API is reachable. The value is undefined while loading, null if the
 * request failed, and the data otherwise; a failure is retried when the API comes back.
 */
export function useLoaded<T>(load: () => Promise<T | null>, api: ApiState): T | null | undefined {
  const [value, setValue] = useState<T | null | undefined>(undefined);
  const have = useRef(false);

  useEffect(() => {
    if (have.current) return;
    let current = true;
    void load().then((result) => {
      if (!current) return;
      if (result !== null) have.current = true;
      setValue(result);
    });
    return () => {
      current = false;
    };
  }, [api, load]);

  return value;
}

/** The value counted up from its last displayed value; immediate when motion is reduced. */
export function useTweenedNumber(target: number | null, reduced: boolean, duration = 520): number | null {
  const [shown, setShown] = useState<number | null>(target);
  const last = useRef<number>(target ?? 0);

  useEffect(() => {
    if (target === null) {
      last.current = 0;
      setShown(null);
      return;
    }
    if (reduced || duration <= 0) {
      last.current = target;
      setShown(target);
      return;
    }
    const from = last.current;
    const start = performance.now();
    let frame = 0;
    const step = (now: number): void => {
      const k = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - k) ** 3;
      const value = from + (target - from) * eased;
      last.current = value;
      setShown(value);
      if (k < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, reduced, duration]);

  return shown;
}

/** Width of an element in pixels, kept up to date; 0 until it is measured. */
export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (element === null) return;
    const read = (): void => setWidth(element.clientWidth);
    read();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", read);
      return () => window.removeEventListener("resize", read);
    }
    const observer = new ResizeObserver(read);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
