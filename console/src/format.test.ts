import { describe, expect, it } from "vitest";

import { f3, f4 } from "./format";

describe("rounding of printed numbers", () => {
  it("rounds an exact tie up, the one rule for the page (the checker uses the same)", () => {
    expect(f4(0.03125)).toBe("0.0313");
    expect(f3(0.0625)).toBe("0.063");
    expect(f4(0.03124)).toBe("0.0312");
  });
});
