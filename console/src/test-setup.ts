import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  try {
    window.sessionStorage.clear();
    window.localStorage.clear();
  } catch {
    // storage may be replaced by a test
  }
});

// jsdom does not implement scrolling
window.scrollTo = (() => {}) as typeof window.scrollTo;
