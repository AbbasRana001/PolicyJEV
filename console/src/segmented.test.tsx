import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Segmented } from "./components/Segmented";

const options = [
  { id: "a", text: "First" },
  { id: "b", text: "Second" },
  { id: "c", text: "Third" },
] as const;

type Id = (typeof options)[number]["id"];

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Segmented", () => {
  it("marks the chosen option and calls back with the one clicked", () => {
    const onChange = vi.fn();
    render(<Segmented<Id> label="Pick" options={options} value="a" onChange={onChange} />);
    expect(screen.getByRole("tab", { name: "First" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Second" }).getAttribute("aria-selected")).toBe("false");
    fireEvent.click(screen.getByRole("tab", { name: "Third" }));
    expect(onChange).toHaveBeenCalledWith("c");
  });

  it("moves the choice with the arrow keys and wraps around", () => {
    const onChange = vi.fn();
    render(<Segmented<Id> label="Pick" options={options} value="c" onChange={onChange} />);
    const group = screen.getByRole("tablist", { name: "Pick" });
    fireEvent.keyDown(group, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith("a");
    fireEvent.keyDown(group, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenLastCalledWith("b");
  });

  it("only the chosen option is in the tab order", () => {
    render(<Segmented<Id> label="Pick" options={options} value="b" onChange={() => {}} />);
    expect(screen.getByRole("tab", { name: "Second" }).tabIndex).toBe(0);
    expect(screen.getByRole("tab", { name: "First" }).tabIndex).toBe(-1);
  });

  it("measures the highlight again when the control is shown after being hidden", () => {
    // A control inside a hidden section is measured at width 0; it must be measured again once it has a size.
    let size = { left: 0, width: 0 };
    vi.spyOn(HTMLElement.prototype, "offsetLeft", "get").mockImplementation(() => size.left);
    vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockImplementation(() => size.width);
    let notify: () => void = () => {};
    vi.stubGlobal(
      "ResizeObserver",
      class {
        private readonly callback: () => void;
        constructor(callback: () => void) {
          this.callback = callback;
        }
        // The browser only reports sizes of elements that were observed.
        observe(): void {
          notify = this.callback;
        }
        disconnect(): void {
          notify = () => {};
        }
      },
    );

    const { container } = render(<Segmented<Id> label="Pick" options={options} value="a" onChange={() => {}} />);
    const pill = container.querySelector(".pill") as HTMLElement;
    expect(pill.style.width).toBe("0px");

    size = { left: 4, width: 120 };
    act(() => notify());
    expect(pill.style.width).toBe("120px");
    expect(pill.style.transform).toBe("translateX(4px)");
  });

  it("has no highlight in the simple variant", () => {
    const { container } = render(<Segmented<Id> label="Pick" variant="simple" options={options} value="a" onChange={() => {}} />);
    expect(container.querySelector(".pill")).toBeNull();
    expect(container.querySelector(".seg.simple")).not.toBeNull();
  });
});
