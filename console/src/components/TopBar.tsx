import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import type { HealthState } from "../hooks";
import { SystemStatus } from "./SystemStatus";

export const TABS = [
  { id: "score", label: "Score" },
  { id: "policy", label: "Policy" },
  { id: "results", label: "Results" },
] as const;
export type TabId = (typeof TABS)[number]["id"];

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 28 28" aria-hidden="true">
      <rect x=".75" y=".75" width="26.5" height="26.5" rx="8" fill="#191A1F" stroke="#33353E" strokeWidth="1.5" />
      <rect x="5" y="14" width="8" height="6" rx="2" fill="#38D6A0" />
      <rect x="14" y="14" width="6" height="6" rx="2" fill="#FFB224" />
      <rect x="21" y="14" width="2.5" height="6" rx="1.25" fill="#FF5468" />
      <path d="M16.5 7.5v4.5" stroke="#F2F1EE" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

interface TopBarProps {
  tab: TabId;
  onTab: (tab: TabId) => void;
  health: HealthState;
}

/** Brand, section tabs with a sliding highlight, and the status chips. */
export function TopBar({ tab, onTab, health }: TopBarProps) {
  const buttons = useRef<Map<TabId, HTMLButtonElement>>(new Map());
  const [ink, setInk] = useState({ left: 0, width: 0 });

  const measure = useCallback((): void => {
    const button = buttons.current.get(tab);
    if (button !== undefined) setInk({ left: button.offsetLeft, width: button.offsetWidth });
  }, [tab]);

  useLayoutEffect(measure, [measure]);
  useEffect(() => {
    window.addEventListener("resize", measure);
    void document.fonts?.ready.then(measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  return (
    <header className="bar">
      <div className="wrap">
        <div className="brand">
          <Logo />
          <div className="mark">
            Policy<span>JEV</span>
          </div>
        </div>
        <nav className="tabs" role="tablist" aria-label="Sections">
          <span
            className="tab-ink"
            aria-hidden="true"
            style={{ width: ink.width, transform: `translateX(${ink.left}px)` }}
          />
          {TABS.map((item) => (
            <button
              key={item.id}
              ref={(element) => {
                if (element === null) buttons.current.delete(item.id);
                else buttons.current.set(item.id, element);
              }}
              type="button"
              className="tab"
              role="tab"
              id={`tab-${item.id}`}
              aria-controls={`view-${item.id}`}
              aria-selected={tab === item.id}
              onClick={() => onTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <SystemStatus api={health.api} systems={health.systems} />
      </div>
    </header>
  );
}
