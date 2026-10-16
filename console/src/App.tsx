import { useState } from "react";

import { getResults, getThresholds } from "./api";
import { PolicyView } from "./components/PolicyView";
import { ResultsView } from "./components/ResultsView";
import { ScoreView } from "./components/ScoreView";
import { TABS, TopBar } from "./components/TopBar";
import type { TabId } from "./components/TopBar";
import { useHealth, useLoaded } from "./hooks";

export function App() {
  const [tab, setTab] = useState<TabId>("score");
  const health = useHealth();
  const thresholds = useLoaded(getThresholds, health.api);
  const results = useLoaded(getResults, health.api);

  const choose = (next: TabId): void => {
    setTab(next);
    if (typeof window.scrollTo === "function") window.scrollTo({ top: 0 });
  };

  return (
    <>
      <TopBar tab={tab} onTab={choose} health={health} />
      <main className="wrap">
        {TABS.map((item) => (
          <section
            key={item.id}
            id={`view-${item.id}`}
            role="tabpanel"
            aria-labelledby={`tab-${item.id}`}
            className={`view${tab === item.id ? " on" : ""}`}
            hidden={tab !== item.id}
          >
            {item.id === "score" && (
              <ScoreView health={health.systems} thresholds={thresholds} results={results} onApiFailure={health.refresh} />
            )}
            {item.id === "policy" && <PolicyView thresholds={thresholds} results={results} />}
            {item.id === "results" && <ResultsView results={results} />}
          </section>
        ))}
      </main>
    </>
  );
}
