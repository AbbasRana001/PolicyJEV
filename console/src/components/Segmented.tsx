import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

interface Option<T extends string> {
  id: T;
  text: string;
}

interface SegmentedProps<T extends string> {
  label: string;
  options: readonly Option<T>[];
  value: T;
  onChange: (id: T) => void;
  /** "pill" slides a highlight between options; "simple" just fills the chosen one. */
  variant?: "pill" | "simple";
  idPrefix?: string;
}

/** A segmented control (a tab list): one option is chosen, arrow keys move the choice. */
export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  variant = "pill",
  idPrefix,
}: SegmentedProps<T>) {
  const buttons = useRef<Map<T, HTMLButtonElement>>(new Map());
  const container = useRef<HTMLDivElement | null>(null);
  const [pill, setPill] = useState({ left: 0, width: 0 });

  const measure = useCallback((): void => {
    const button = buttons.current.get(value);
    if (button !== undefined) setPill({ left: button.offsetLeft, width: button.offsetWidth });
  }, [value]);

  useLayoutEffect(measure, [measure, options.length]);
  useEffect(() => {
    if (variant !== "pill") return;
    window.addEventListener("resize", measure);
    void document.fonts?.ready.then(measure);
    // The control may sit in a section that is hidden when it first renders (width 0); measure again
    // as soon as it is shown or changes size.
    const element = container.current;
    const observer = typeof ResizeObserver === "undefined" || element === null ? null : new ResizeObserver(measure);
    if (element !== null) observer?.observe(element);
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [measure, variant]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
    if (step === 0) return;
    event.preventDefault();
    const index = options.findIndex((option) => option.id === value);
    const next = options[(index + step + options.length) % options.length];
    if (next === undefined) return;
    onChange(next.id);
    buttons.current.get(next.id)?.focus();
  };

  return (
    <div
      ref={container}
      className={variant === "simple" ? "seg simple" : "seg"}
      role="tablist"
      aria-label={label}
      onKeyDown={onKeyDown}
    >
      {variant === "pill" && (
        <span
          className="pill"
          aria-hidden="true"
          style={{ width: pill.width, transform: `translateX(${pill.left}px)` }}
        />
      )}
      {options.map((option) => (
        <button
          key={option.id}
          ref={(element) => {
            if (element === null) buttons.current.delete(option.id);
            else buttons.current.set(option.id, element);
          }}
          type="button"
          role="tab"
          id={idPrefix === undefined ? undefined : `${idPrefix}-${option.id}`}
          aria-selected={option.id === value}
          tabIndex={option.id === value ? 0 : -1}
          onClick={() => onChange(option.id)}
        >
          {option.text}
        </button>
      ))}
    </div>
  );
}
